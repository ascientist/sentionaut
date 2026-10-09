#!/bin/bash
#SBATCH --job-name=sentionaut-mnist
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%j.out
# Fine-tune the random-stimulation student on MNIST-derived Argus II stimulation.
# Teacher rollouts stay on CUDA. Data and the new checkpoint stay on scratch.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT=/network/scratch/j/jacob.lavoie/sentionaut
export UV_CACHE_DIR="$ROOT/uv-cache"
export SENTIONAUT_AXON_CACHE="$ROOT/axon_cache"
INIT=${INIT:-$ROOT/axonmap/full/ckpt/axon_world.pt}
mkdir -p "$ROOT/logs" "$ROOT/axonmap/mnist/logs" "$ROOT/axonmap/mnist/ckpt" "$ROOT/uv-cache"
cd "$HOME/sentionaut"
uv sync
uv run python - <<PY
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() > 0
print("gpu", torch.cuda.get_device_name(0))
from pathlib import Path
from sentionaut.learned.mnist_world import generate_mnist_dataset
generate_mnist_dataset(
    Path("$ROOT/axonmap/mnist/transitions.h5"),
    episodes=512,
    device=torch.device("cuda"),
    cache_dir=Path("$ROOT/mnist"),
)
print("wrote", "$ROOT/axonmap/mnist/transitions.h5")
PY
cp "$ROOT/axonmap/mnist/transitions.h5" "$SLURM_TMPDIR/transitions.h5"
uv run sentionaut-distill-axon \
    --dataset "$SLURM_TMPDIR/transitions.h5" \
    --init-ckpt "$INIT" \
    --epochs 8 \
    --batch-size 128 \
    --num-workers 4 \
    --device cuda \
    --ckpt "$ROOT/axonmap/mnist/ckpt/axon_world.pt" \
    --timing "$ROOT/axonmap/mnist/logs/timing_train.json"
cp "$ROOT/axonmap/mnist/ckpt/axon_world.pt" "$HOME/sentionaut/checkpoints/axon_world_mnist.pt"
