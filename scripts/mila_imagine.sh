#!/bin/bash
#SBATCH --job-name=sentionaut-imagine
#SBATCH --partition=main
#SBATCH --array=0-23%8
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=01:15:00
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%A_%a.out
# Train a vision encoder four ways on two teachers, three seeds: 24 array tasks.
#
#   task id -> preset (axonmap, dynaphos) x arm (4) x seed (0..2)
#
# Every task gets the same GPU type (L40S), so the 30-minute wall-clock budgets
# are comparable across arms. `main` jobs are not preempted by other `main` jobs.
# Each task asks for 1 GPU, 4 CPUs and 24 GB, inside the main partition's
# per-job cap (2 GPUs, 8 CPUs, 48 GB) and the L40S node's per-GPU bundle
# (12 CPUs, 250 GB). `%8` keeps at most 8 tasks running at once.
#
# Submit from a login node, from $HOME/sentionaut on this branch:
#   sbatch scripts/mila_imagine.sh
# Override the budget with:  BUDGET_S=600 sbatch scripts/mila_imagine.sh

set -euo pipefail
source "$HOME/sentionaut/scripts/mila_prepare.sh"

PRESETS=(axonmap dynaphos)
ARMS=(teacher-bptt wm-bptt reinforce-sim reinforce-wm)
ID=${SLURM_ARRAY_TASK_ID:-0}
PRESET=${PRESETS[$((ID / 12))]}
ARM=${ARMS[$(((ID % 12) / 3))]}
SEED=$((ID % 3))
BUDGET_S=${BUDGET_S:-1800}
OUT="$SLURM_TMPDIR/imagine"
DEST="$ROOT/imagine/runs"
mkdir -p "$OUT" "$DEST"
echo "task $ID: preset=$PRESET arm=$ARM seed=$SEED budget=${BUDGET_S}s"

srun uv run sentionaut-imagine \
    --arm "$ARM" --preset "$PRESET" --seed "$SEED" --budget-s "$BUDGET_S" \
    --device cuda --threads "$SLURM_CPUS_PER_TASK" --mnist "$MNIST_DIR" --out "$OUT"

# The baselines need no training: compute them once per preset (seed 0, first arm).
if [ "$SEED" = 0 ] && [ "$ARM" = teacher-bptt ]; then
    srun uv run sentionaut-imagine --arm baselines --preset "$PRESET" \
        --device cuda --mnist "$MNIST_DIR" --out "$OUT"
fi
cp "$OUT"/* "$DEST"/
echo "results in $DEST"
