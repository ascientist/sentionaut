#!/bin/bash
#SBATCH --job-name=sentionaut-ablate
#SBATCH --account=def-someuser          # adapt to your DRAC allocation
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ablate_%j.out

# module load python/3.11 cuda
set -euo pipefail

DATASET=${DATASET:-data/world_full.h5}
CONFIG=${CONFIG:-configs/ablation.yaml}

uv run sentionaut-train ablate \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --device cuda
