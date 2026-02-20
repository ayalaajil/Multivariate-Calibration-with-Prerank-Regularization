#!/bin/bash

# Configuration
PYTHON=/home/ubuntu/miniconda3/envs/calibenv/bin/python
LAMBDA=10
SEEDS=(0 42 866 12 4)
GPUS=(0 1 2 3 4)

# Launch Density seeds on GPUs 0-4
echo "Launching Density parallel seeds on GPUs 0-4..."
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU=${GPUS[$i]}
    echo "Starting Density Seed $SEED on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU SEARCH_PRERANK=density SEARCH_SEED=$SEED SEARCH_LAMBDA=$LAMBDA $PYTHON -u multiple-preranks.py > "logs/log_density_seed${SEED}.txt" 2>&1 &
    sleep 2
done

# Launch CDF seeds on GPUs 5-7 (using 3 GPUs for 5 seeds, they will share or we can use more)
# Let's use 5-7 and reuse 0-1 for the last two CDF seeds
echo "Launching CDF parallel seeds on GPUs 5-7 and 0-1..."
CDF_GPUS=(5 6 7 0 1)
for i in "${!SEEDS[@]}"; do
    SEED=${SEEDS[$i]}
    GPU=${CDF_GPUS[$i]}
    echo "Starting CDF Seed $SEED on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU SEARCH_PRERANK=cdf SEARCH_SEED=$SEED SEARCH_LAMBDA=$LAMBDA $PYTHON -u multiple-preranks.py > "logs/log_cdf_seed${SEED}.txt" 2>&1 &
    sleep 2
done

echo "All seed-parallel processes launched."
