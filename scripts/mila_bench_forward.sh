#!/bin/bash
#SBATCH --job-name=sentionaut-bench
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=01:30:00
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%j.out
# Throughput of axon-map forward models on one L40S node (scripts/bench_forward.py).
#
#   third party: pulse2percept cython (CPU, 1 and 8 threads) and pulse2percept jax (GPU)
#   this repo:   torch port exact and sparse (GPU), learned world model (GPU)
#
# 8 CPUs is the main partition's per-job maximum; it lets the cython engine use
# all of them. Your own implementation can be timed on the same stimuli with
#   EXTRA=my_module:my_function sbatch scripts/mila_bench_forward.sh
# (f(amp) with amp a float32 numpy array (B, 60), returning (B, H, W)).

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$HOME/sentionaut}/scripts/mila_prepare.sh"
OUT="$ROOT/bench/$SLURM_JOB_ID"
mkdir -p "$OUT"
BENCH="uv run python scripts/bench_forward.py --batches 1 8 32 128 512 2048"

$BENCH --engines p2p-cython --threads 1 --max-p2p-batch 128 | tee "$OUT/p2p_cython_t1.jsonl"
$BENCH --engines p2p-cython --threads 8 --max-p2p-batch 128 | tee "$OUT/p2p_cython_t8.jsonl"
$BENCH --engines torch-exact torch-sparse wm --device cuda --threads 8 | tee "$OUT/torch_cuda.jsonl"
# pulse2percept's jax engine needs jax with CUDA, and setuptools<81 for pkg_resources.
uv run --with "jax[cuda12]" --with "setuptools<81" python scripts/bench_forward.py \
    --engines p2p-jax --threads 8 --batches 1 8 32 128 --max-p2p-batch 128 \
    | tee "$OUT/p2p_jax_cuda.jsonl"
if [ -n "${EXTRA:-}" ]; then
    $BENCH --engines extra --extra "$EXTRA" | tee "$OUT/extra.jsonl"
fi
echo "results in $OUT"
