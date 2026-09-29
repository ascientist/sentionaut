#!/bin/bash
# JEPA diagnostics (collapse, information, control): one task per (variant, regime).
#
#   sbatch --array=0-31 scripts/slurm_neurostim_jepa.sh          # 8 variants x 4 regimes
#   SEED=1 sbatch --array=0-31 scripts/slurm_neurostim_jepa.sh   # another seed
#
# Results: results/neurostim_jepa/<regime>_<variant>_seed<k>.json. CPU-bound.
#SBATCH --job-name=neurostim-jepa
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/jepa_%A_%a.out

set -euo pipefail

VARIANTS=(old vicreg-linprobe vicreg vicreg-ema vicreg-L32 vicreg-long vicreg-H20k4 vicreg-H30k6)
REGIMES=(A B C D)
V=${VARIANTS[$((SLURM_ARRAY_TASK_ID % ${#VARIANTS[@]}))]}
R=${REGIMES[$((SLURM_ARRAY_TASK_ID / ${#VARIANTS[@]}))]}

mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
uv run python examples/neurostim_jepa_diagnostics.py \
    --regime "$R" --variants "$V" --seed "${SEED:-0}"
