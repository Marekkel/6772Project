#!/bin/bash
set -e

mkdir -p outputs/preset_validation
mkdir -p outputs/models

for preset in clean bottleneck long-rtt lossy aqm
do
  for seed in 0 1 2
  do
    echo "Running preset=${preset}, seed=${seed}"

    python3 train_tcp_rl_cubic_tuned.py \
      --env local \
      --algo ppo \
      --mode all \
      --timesteps 100000 \
      --episodes 5 \
      --seed ${seed} \
      --network-preset ${preset} \
      --model-path outputs/models/ppo_${preset}_seed_${seed}.zip \
      > outputs/preset_validation/log_${preset}_seed_${seed}.txt

    cp outputs/evaluation.csv outputs/preset_validation/evaluation_${preset}_seed_${seed}.csv
    cp outputs/evaluation_metrics.png outputs/preset_validation/evaluation_${preset}_seed_${seed}.png
  done
done