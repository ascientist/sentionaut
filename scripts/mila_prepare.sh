#!/bin/bash
# Sourced by the Mila job scripts. Builds the shared caches once, under a lock,
# so concurrent array tasks do not race on the same files:
#   - the project environment (uv sync, on the GPU node: the CUDA torch wheel),
#   - the MNIST cache used for encoder targets,
#   - the axon-map topographies for the imagination grid and the benchmark grid.
# https://docs.mila.quebec/ (Slurm guide: install GPU frameworks from a GPU node)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT=${SENTIONAUT_ROOT:-$SCRATCH/sentionaut}
export UV_CACHE_DIR="$ROOT/uv-cache"
export SENTIONAUT_AXON_CACHE="$ROOT/axon_cache"
export MNIST_DIR="$ROOT/mnist"
mkdir -p "$ROOT/logs" "$UV_CACHE_DIR" "$SENTIONAUT_AXON_CACHE" "$MNIST_DIR"
# The checkout the job was submitted from (e.g. $SCRATCH/sentionaut/repo).
REPO=${SENTIONAUT_REPO:-${SLURM_SUBMIT_DIR:-$HOME/sentionaut}}
cd "$REPO"

(
    flock -w 3600 9
    uv sync --locked
    uv run python - <<PY
import torch
assert torch.cuda.is_available(), "CUDA required: submit from a GPU partition"
print("gpu", torch.cuda.get_device_name(0))
from sentionaut.neurostim.percept import load_mnist
load_mnist("$MNIST_DIR")
from sentionaut.core.config import Config
from sentionaut.core.registry import build_components
for kw in (
    dict(model="axonmap", implant="argusii", xrange=(-15, 15), yrange=(-10, 10), xystep=1.0),
):
    build_components(Config(**kw), device=torch.device("cpu"))
print("caches ready")
PY
) 9>"$ROOT/prepare.lock"
