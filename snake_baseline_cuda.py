import matplotlib.pyplot as plt
import sys
import time
from replay_buffer import ReplayMemory
from collections import deque
from Game import GameEnvironment
from model import QNetwork, get_network_input
import os
import random
import numpy as np
import torch
import torch.nn as nn

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up directories and logging
dir = "./dir_chk_reproduce"
if not os.path.exists(dir):
    os.mkdir(dir)
log = open(os.path.join(dir, "log.txt"), "w+", buffering=1)
sys.stdout = log
sys.stderr = log

# Initialize model, move it to device
model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5).to(device)
epsilon = 0.1
gridsize = 15
GAMMA = 0.9

board = GameEnvironment(gridsize, nothing=0, dead=-1, apple=1)
memory = ReplayMemory(1000)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

epsilon_start = 1.0
epsilon_min = 0.01
decay_rate = 0.995
epsilon = 1.0

def run_episode(num_games):
    run = True
    move = 0
    games_played = 0
    total_reward = 0
    episode_games = 0
    len_array = []
    global epsilon
    epsilon = max(epsilon * decay_rate, epsilon_min)
    while run:
        state = get_network_input(board.snake, board.apple).to(device)
        action_0 = model(state)
        rand = np.random.uniform(0, 1)
        if rand > epsilon:
            action = torch.argmax(action_0).item()
        else:
            action = np.random.randint(0, 5)

        reward, done, len_of_snake = board.update_boardstate(action)
        next_state = get_network_input(board.snake, board.apple).to(device)

        memory.push(state, action, reward, next_state, done)

        total_reward += reward
        episode_games += 1

        if board.game_over:
            games_played += 1
            len_array.append(len_of_snake)
            board.resetgame()

            if num_games == games_played:
                run = False

    avg_len_of_snake = np.mean(len_array)
    max_len_of_snake = np.max(len_array)
    return total_reward, avg_len_of_snake, max_len_of_snake

# Initialize target model, move to device
target_model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()

# Define other constants
target_update_frequency = 100
MSE = nn.MSELoss()

def learn(num_updates, batch_size):
    total_loss = 0

    for i in range(num_updates):
        optimizer.zero_grad()
        sample = memory.sample(batch_size)

        states, actions, rewards, next_states, dones = sample
        states = torch.cat([x.unsqueeze(0) for x in states], dim=0).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.cat([x.unsqueeze(0) for x in next_states]).to(device)
        dones = torch.FloatTensor(dones).to(device)

        q_local = model(states)
        next_q_value = target_model(next_states)

        Q_expected = q_local.gather(1, actions.unsqueeze(0).transpose(0, 1)).transpose(0, 1).squeeze(0)
        Q_targets_next = torch.max(next_q_value, 1)[0] * (1 - dones)
        Q_targets = rewards + GAMMA * Q_targets_next

        loss = MSE(Q_expected, Q_targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

    return total_loss

num_episodes = 60000
num_updates = 500
print_every = 10
games_in_episode = 30
batch_size = 20

def train():
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    avg_len_array = []
    avg_max_len_array = []

    time_start = time.time()

    for i_episode in range(num_episodes + 1):
        score, avg_len, max_len = run_episode(games_in_episode)

        scores_deque.append(score)
        scores_array.append(score)
        avg_len_array.append(avg_len)
        avg_max_len_array.append(max_len)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        total_loss = learn(num_updates, batch_size)

        dt = int(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Ep.: {:6}, Loss: {:.3f}, Avg.Score: {:.2f}, Avg.LenOfSnake: {:.2f}, Max.LenOfSnake: {:.2f}, Time: {:02}:{:02}:{:02}'.
                  format(i_episode, total_loss, avg_score, avg_len, max_len, dt // 3600, dt % 3600 // 60, dt % 60))

        memory.truncate()

        if i_episode % 250 == 0 and i_episode > 0:
            torch.save(model.state_dict(), os.path.join(dir, f"Snake_{i_episode}"))

    return scores_array, avg_scores_array, avg_len_array, avg_max_len_array

scores, avg_scores, avg_len_of_snake, max_len_of_snake = train()
print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg score on 100 episodes")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel('Score')
plt.xlabel('Episodes #')
plt.savefig(os.path.join(dir, "scores.png"))
ax1 = fig.add_subplot(121)
plt.plot(np.arange(1, len(avg_len_of_snake) + 1), avg_len_of_snake, label="Avg Len of Snake")
plt.plot(np.arange(1, len(max_len_of_snake) + 1), max_len_of_snake, label="Max Len of Snake")
plt.legend(bbox_to_anchor=(1.05, 1))
plt.ylabel('Length of Snake')
plt.xlabel('Episodes #')
plt.savefig(os.path.join(dir, "Length.png"))
n, bins, patches = plt.hist(max_len_of_snake, 45, density=1, facecolor='green', alpha=0.75)
mu = round(np.mean(max_len_of_snake), 2)
sigma = round(np.std(max_len_of_snake), 2)
median = round(np.median(max_len_of_snake), 2)
print('mu: ', mu, ', sigma: ', sigma, ', median: ', median)
plt.xlabel('Max.Lengths, mu = {:.2f}, sigma={:.2f},  median: {:.2f}'.format(mu, sigma, median))
plt.ylabel('Probability')
plt.title('Histogram of Max.Lengths')
plt.axis([4, 44, 0, 0.15])
plt.grid(True)
plt.savefig(os.path.join(dir, "Max Length.png"))