import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from replay_buffer import ReplayMemory
from collections import deque
from Game import GameEnvironment
from model import QNetwork, get_network_input
import numpy as np
import matplotlib.pyplot as plt
import argparse

GAMMA = 0.9
epsilon = 1.0

# Setup DDP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_episode(rank, model, board, memory, decay_rate, num_games):
    model.train()

    rand = np.random.uniform(0, 1)
    global epsilon
    epsilon = max(epsilon * decay_rate, 0.01)
    games_played = 0
    while True:
        state = get_network_input(board.snake, board.apple).to(rank)
        action_0 = model(state)
        if rand > epsilon:
            action = torch.argmax(action_0).item()
        else:
            action = np.random.randint(0, 5)
        reward, done, len_of_snake = board.update_boardstate(action)
        next_state = get_network_input(board.snake, board.apple).to(rank)
        memory.push(state, action, reward, next_state, done)
        if board.game_over:
            games_played += 1
            board.resetgame()
            if num_games == games_played:
                return games_played, reward

def train(rank, 
          world_size, 
          num_episodes, 
          num_updates, 
          batch_size, 
          num_games, 
          gridsize,
          checkpoint_dir):
    setup(rank, world_size)
    
    model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5).to(rank)
    model = DDP(model, device_ids=[rank])
    
    board = GameEnvironment(gridsize=gridsize, nothing=0, dead=-1, apple=1)
    memory = ReplayMemory(1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    

    decay_rate = 0.995
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []
    if rank == 0 and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for episode in range(num_episodes + 1):
        games_played, score = run_episode(rank, model, board, memory, decay_rate, num_games)
        scores_deque.append(score)
        scores_array.append(score)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        # Learning phase
        total_loss = learn(rank, model, memory, optimizer, num_updates, batch_size)
        if episode % 10 == 0:
            print(f"Rank {rank} | Episode {episode}: Total Score: {score} Loss: {total_loss}")
        memory.truncate()
        # Synchronize the model weights
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= world_size

        # Save checkpoint every 250 episodes
        if rank == 0 and episode % 250 == 0 and episode > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"snake_episode{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': model.module.state_dict(),  # Save the underlying model's state_dict
                'optimizer_state_dict': optimizer.state_dict(),
                'scores_array': scores_array,
                'epsilon': epsilon
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    if rank == 0:
        # Only rank 0 will plot and save the graph
        plot_scores(scores_array, avg_scores_array, checkpoint_dir)

    cleanup()

def learn(rank, model, memory, optimizer, num_updates, batch_size):
    total_loss = 0
    MSE = nn.MSELoss()

    for i in range(num_updates):
        optimizer.zero_grad()
        sample = memory.sample(batch_size)

        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x.unsqueeze(0) for x in states], dim=0).to(rank)
        actions = torch.LongTensor(actions).to(rank)
        rewards = torch.FloatTensor(rewards).to(rank)
        next_states = torch.cat([x.unsqueeze(0) for x in next_states]).to(rank)
        dones = torch.FloatTensor(dones).to(rank)

        q_local = model(states)
        next_q_value = model(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)
        Q_targets_next = torch.max(next_q_value, 1)[0] * (1 - dones)
        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss

def plot_scores(scores_array, avg_scores_array, checkpoint_dir):
    """
    Plot and save the scores to a PNG file.
    This function is called only on rank 0.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_array) + 1), scores_array, label="Score")
    plt.plot(np.arange(1, len(avg_scores_array) + 1), avg_scores_array, label="Avg score on 100 episodes")
    plt.xlabel('Episodes #')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'./{checkpoint_dir}/scores.png')
    print("Scores plot saved as 'scores.png'")

def main():
    parser = argparse.ArgumentParser(description="Distributed DQN Training")
    parser.add_argument('--gridsize', type=int, default=15, help='Size of the game grid')
    parser.add_argument('--num_episodes', type=int, default=251, help='Number of episodes to train')
    parser.add_argument('--num_updates', type=int, default=500, help='Number of updates per episode')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    parser.add_argument('--num_games', type=int, default=30, help='Number of games per episode')
    parser.add_argument('--checkpoint_dir', type=str, default='exp_ddp')
    args = parser.parse_args()

    world_size = 8  # Number of GPUs

    # Spawn processes for DDP
    mp.spawn(train, args=(world_size, 
                          args.num_episodes, 
                          args.num_updates, 
                          args.batch_size, 
                          args.num_games,
                          args.gridsize,
                          args.checkpoint_dir), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
