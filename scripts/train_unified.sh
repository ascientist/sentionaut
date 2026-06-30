#!/bin/bash
#SBATCH --job-name=sentionaut-train
#SBATCH --account=def-someuser          # adapt to your DRAC allocation
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out

# module load python/3.11 cuda
set -euo pipefail

DATASET=${DATASET:-data/world_full.h5}
CONFIG=${CONFIG:-configs/train.yaml}

uv run sentionaut-train train \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --mode shared \
    --device cuda
