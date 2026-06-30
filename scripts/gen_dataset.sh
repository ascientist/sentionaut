#!/bin/bash
#SBATCH --job-name=sentionaut-gen
#SBATCH --account=def-someuser          # adapt to your DRAC allocation
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gen_%j.out

# Digital Research Alliance of Canada launcher. Adapt module/venv lines.
# module load python/3.11 cuda
set -euo pipefail

OUTPUT=${OUTPUT:-data/world_full.h5}
EPISODES=${EPISODES:-512}
SEQ_LEN=${SEQ_LEN:-8}
XRANGE=${XRANGE:-"-12 12"}
YRANGE=${YRANGE:-"-12 12"}
XYSTEP=${XYSTEP:-0.25}

uv run sentionaut-world \
    --output "$OUTPUT" \
    --model axonmap --model scoreboard --model dynaphos \
    --episodes "$EPISODES" --sequence-length "$SEQ_LEN" \
    --xrange $XRANGE --yrange $YRANGE --xystep "$XYSTEP" \
    --device cuda
