#!/bin/bash
#SBATCH --job-name=sentionaut-ablate
#SBATCH --account=def-someuser
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ablate_%j.out

set -euo pipefail

DATASET=${DATASET:-data/world_full.h5}
CONFIG=${CONFIG:-configs/ablation.yaml}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "$LOG_DIR"

uv run sentionaut-train ablate \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --device cuda \
    --log-dir "$LOG_DIR"
