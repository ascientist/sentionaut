#!/bin/bash
#SBATCH --job-name=sentionaut-env
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:20:00
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%j.out
# Stage 1. uv sync on a GPU node so the torch wheel is the CUDA build.
# https://docs.mila.quebec/examples/frameworks/pytorch_setup/

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/network/scratch/j/jacob.lavoie/sentionaut/uv-cache
export SENTIONAUT_AXON_CACHE=/network/scratch/j/jacob.lavoie/sentionaut/axon_cache
mkdir -p "$UV_CACHE_DIR" "$SENTIONAUT_AXON_CACHE" \
    /network/scratch/j/jacob.lavoie/sentionaut/logs \
    /network/scratch/j/jacob.lavoie/sentionaut/axonmap/smoke
cd "$HOME/sentionaut"
uv sync
uv run python - <<'PY'
import torch
assert torch.backends.cuda.is_built(), "torch was not built with CUDA"
assert torch.cuda.is_available() and torch.cuda.device_count() > 0
print("gpu", torch.cuda.get_device_name(0))
PY
echo ok > /network/scratch/j/jacob.lavoie/sentionaut/axonmap/smoke/envcheck.txt
cat /network/scratch/j/jacob.lavoie/sentionaut/axonmap/smoke/envcheck.txt
