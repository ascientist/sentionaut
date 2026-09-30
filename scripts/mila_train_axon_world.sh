#!/bin/bash
#SBATCH --job-name=sentionaut-distill
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%j.out
# Student on CUDA. Read the HDF5 from node-local disk, write checkpoints to scratch.
# https://docs.mila.quebec/examples/distributed/single_gpu/
# https://docs.mila.quebec/examples/good_practices/checkpointing/

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCALE=${SCALE:-smoke}
EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
BF16_FLAG=""
if [[ "${BF16:-0}" == "1" ]]; then
    BF16_FLAG="--bf16"
fi
ROOT=/network/scratch/j/jacob.lavoie/sentionaut
export UV_CACHE_DIR="$ROOT/uv-cache"
export SENTIONAUT_AXON_CACHE="$ROOT/axon_cache"
mkdir -p "$ROOT/logs" "$ROOT/axonmap/$SCALE/logs" "$ROOT/axonmap/$SCALE/ckpt" "$HOME/sentionaut/checkpoints"
cd "$HOME/sentionaut"
uv run python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA required"
print("gpu", torch.cuda.get_device_name(0))
PY
cp "$ROOT/axonmap/$SCALE/transitions.h5" "$SLURM_TMPDIR/transitions.h5"
uv run sentionaut-distill-axon \
    --dataset "$SLURM_TMPDIR/transitions.h5" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --device cuda \
    --ckpt "$ROOT/axonmap/$SCALE/ckpt/axon_world.pt" \
    --timing "$ROOT/axonmap/$SCALE/logs/timing_train.json" \
    --num-workers "$NUM_WORKERS" \
    $BF16_FLAG
cp "$ROOT/axonmap/$SCALE/ckpt/axon_world.pt" "$HOME/sentionaut/checkpoints/axon_world_${SCALE}.pt"
uv run python - <<PY
import json
t = json.load(open("$ROOT/axonmap/$SCALE/logs/timing_train.json"))
parts = {"data_load_s": t["data_load_s"], "forward_backward_s": t["forward_backward_s"]}
slow = max(parts, key=parts.get)
print("slowest", slow, parts, "val_mse", t["val_mse"])
PY
