#!/bin/bash
#SBATCH --job-name=sentionaut-train
#SBATCH --account=def-someuser
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out

set -euo pipefail

DATASET=${DATASET:-data/world_full.h5}
CONFIG=${CONFIG:-configs/train.yaml}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "$LOG_DIR"

uv run sentionaut-train train \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --mode shared \
    --device cuda \
    --log "$LOG_DIR/eval.json"
