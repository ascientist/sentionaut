#!/bin/bash
# Multi-seed NeuroStim baseline suite: one array task per (seed, regime).
#
#   sbatch --array=0-19 scripts/slurm_neurostim_baselines.sh   # 5 seeds x 4 regimes
#   uv run python scripts/aggregate_neurostim_seeds.py results/neurostim_baselines
#
# CPU-bound (small models, Python control loops): no GPU requested. A task takes
# ~30 min on 4 cores. On Mila, submit from a login node; pick the partition
# that fits (e.g. `long` or `long-cpu`).
#SBATCH --job-name=neurostim-baselines
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/neurostim_%A_%a.out

set -euo pipefail

REGIMES=(A B C D)
SEED=$((SLURM_ARRAY_TASK_ID / ${#REGIMES[@]}))
REGIME=${REGIMES[$((SLURM_ARRAY_TASK_ID % ${#REGIMES[@]}))]}
OUT=${OUT:-results/neurostim_baselines}/seed${SEED}_${REGIME}

mkdir -p logs "$OUT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

uv run python examples/neurostim_baselines_tutorial.py \
    --regimes "$REGIME" \
    --seed "$SEED" \
    --outdir "$OUT"
