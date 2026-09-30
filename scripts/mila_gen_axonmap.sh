#!/bin/bash
#SBATCH --job-name=sentionaut-gen
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:30:00
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%j.out
# Axon-map teacher on CUDA. Dataset, cache, and timing stay on scratch.
# https://docs.mila.quebec/technical_reference/clusters/mila/storage/
# https://docs.mila.quebec/examples/distributed/single_gpu/

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCALE=${SCALE:-smoke}
EPISODES=${EPISODES:-1}
SEQ_LEN=${SEQ_LEN:-2}
SILENT_TAIL=${SILENT_TAIL:-2}
XYSTEP=${XYSTEP:-1.0}
XRANGE=${XRANGE:--2 2}
YRANGE=${YRANGE:--2 2}
ROOT=/network/scratch/j/jacob.lavoie/sentionaut
export UV_CACHE_DIR="$ROOT/uv-cache"
export SENTIONAUT_AXON_CACHE="$ROOT/axon_cache"
mkdir -p "$UV_CACHE_DIR" "$SENTIONAUT_AXON_CACHE" "$ROOT/logs" "$ROOT/axonmap/$SCALE/logs"
cd "$HOME/sentionaut"
uv run python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA required"
print("gpu", torch.cuda.get_device_name(0))
PY
# shellcheck disable=SC2086
uv run sentionaut-world \
    --output "$ROOT/axonmap/$SCALE/transitions.h5" \
    --model axonmap \
    --episodes "$EPISODES" \
    --sequence-length "$SEQ_LEN" \
    --silent-tail "$SILENT_TAIL" \
    --xrange $XRANGE \
    --yrange $YRANGE \
    --xystep "$XYSTEP" \
    --device cuda \
    --timing "$ROOT/axonmap/$SCALE/logs/timing_gen.json"
uv run python - <<PY
import json
p = "$ROOT/axonmap/$SCALE/logs/timing_gen.json"
t = json.load(open(p))
parts = {"topography_s": t["topography_s"], "gpu_step_s": t["gpu_step_s"], "h5_write_s": t["h5_write_s"]}
slow = max(parts, key=parts.get)
print("slowest", slow, parts)
PY
