#!/bin/bash
#SBATCH --job-name=sentionaut-video
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%j.out
# Genie-style video student on axon-map streams, plus the exact-fade student as
# baseline, then the report. Every stage is resumable, so a preempted job that is
# requeued picks up where it stopped: the dataset is written under a temporary name
# and renamed when complete, and both trainings resume from their per-epoch checkpoints.
# https://docs.mila.quebec/examples/good_practices/checkpointing/
#
#   sbatch scripts/mila_axon_video.sh                        # full 97x97 grid
#   XYSTEP=0.5 SCALE=half sbatch scripts/mila_axon_video.sh  # grid of the CPU stand-in

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
SCALE=${SCALE:-full}
XYSTEP=${XYSTEP:-0.25}
EPISODES=${EPISODES:-512}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
ROOT=/network/scratch/j/jacob.lavoie/sentionaut
OUT="$ROOT/axonvideo/$SCALE"
export UV_CACHE_DIR="$ROOT/uv-cache"
export SENTIONAUT_AXON_CACHE="$ROOT/axon_cache"
mkdir -p "$ROOT/logs" "$OUT/logs" "$OUT/ckpt" "$OUT/report"
cd "$HOME/sentionaut"
uv sync
uv run python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA required"
print("gpu", torch.cuda.get_device_name(0))
PY

if [[ ! -f "$OUT/streams.h5" ]]; then
    uv run sentionaut-world \
        --output "$OUT/streams.partial.h5" \
        --model axonmap --stream \
        --episodes "$EPISODES" --sequence-length 32 --silent-tail 8 \
        --xrange -12 12 --yrange -12 12 --xystep "$XYSTEP" \
        --device cuda \
        --timing "$OUT/logs/timing_gen.json"
    mv "$OUT/streams.partial.h5" "$OUT/streams.h5"
fi
cp "$OUT/streams.h5" "$SLURM_TMPDIR/streams.h5"
DATA="$SLURM_TMPDIR/streams.h5"

uv run sentionaut-axon-video train \
    --dataset "$DATA" --device cuda \
    --epochs "$EPOCHS" --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --ckpt "$OUT/ckpt/axon_video.pt" \
    --timing "$OUT/logs/timing_train_video.json"

uv run sentionaut-distill-axon \
    --dataset "$DATA" --device cuda \
    --epochs "$EPOCHS" --batch-size "$BATCH" --num-workers "$NUM_WORKERS" \
    --lr 5e-4 --rollout-k 16 --train-stride 8 --val-stride 16 --dim 128 --depth 4 \
    --ckpt "$OUT/ckpt/axon_stream_markov.pt" \
    --timing "$OUT/logs/timing_train_markov.json"

uv run sentionaut-axon-video report \
    --dataset "$DATA" --device cuda \
    --ckpt "$OUT/ckpt/axon_video.pt" \
    --baseline-ckpt "$OUT/ckpt/axon_stream_markov.pt" \
    --timing-gen "$OUT/logs/timing_gen.json" \
    --timing-train "$OUT/logs/timing_train_video.json" \
    --timing-baseline "$OUT/logs/timing_train_markov.json" \
    --out-dir "$OUT/report"
echo "report in $OUT/report; copy it to docs/assets/axon-video to publish"
