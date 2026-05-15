#!/bin/bash
set -e

mkdir -p outputs/validation
mkdir -p outputs/models

for seed in 0 1 2 3 4
do
  echo "Running seed ${seed}"

  python3 train_tcp_rl_cubic_tuned.py \
    --env local \
    --algo ppo \
    --mode all \
    --timesteps 100000 \
    --episodes 5 \
    --seed ${seed} \
    --model-path outputs/models/ppo_seed_${seed}.zip \
    > outputs/validation/log_seed_${seed}.txt

  cp outputs/evaluation.csv outputs/validation/evaluation_seed_${seed}.csv
  cp outputs/evaluation_metrics.png outputs/validation/evaluation_seed_${seed}.png
done