#!/bin/bash
#SBATCH --job-name=sentionaut-drawings
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00
#SBATCH --output=/network/scratch/j/jacob.lavoie/sentionaut/logs/%x_%j.out
# Fit the axon map (rho, axlambda, drawing threshold) to the Beyeler et al. 2019
# phosphene drawings of the three Argus II users, by gradient descent on GPU.
# The dataset (69 MB, OSF) is downloaded once to scratch with its SHA-256 checked.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$HOME/sentionaut}/scripts/mila_prepare.sh"
mkdir -p "$ROOT/p2p"
srun uv run sentionaut-fit-drawings --data-dir "$ROOT/p2p" --device cuda \
    --subjects S2 S3 S4 --iters 150 --out "$ROOT/p2p/drawing_fits.json"
echo "results in $ROOT/p2p/drawing_fits.json"
