# num_episodes: epoch
# target_update_frequency: tgt freq
# num_updates: iters in each epoch
# batch_size: bs
# num_games: number of games to run for each iter
# baseline: 5e-3=num_updates*lr
# ours: 1e-3*20=2e-2

python snake_baseline_cuda.py \
--gridsize 15 \
--num_episodes 4000 \
--target_update_frequency 5 \
--lr 1e-3 \
--num_updates 20 \
--batch_size 512 \
--num_games 30 \
--checkpoint_dir exp/reproduce_lr1e-3_tgt5_iter20

# python snake_baseline_cuda.py \
# --gridsize 15 \
# --num_episodes 2000 \
# --target_update_frequency 100 \
# --lr 1e-5 \
# --num_updates 500 \
# --batch_size 20 \
# --num_games 30 \
# --checkpoint_dir exp/reproduce