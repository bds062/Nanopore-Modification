#!/usr/bin/env bash
#SBATCH --job-name=deepmod
#SBATCH --time=1-4:00:00
#SBATCH --output=logs/deepmod_%j.out
#SBATCH --qos=huge-long
#SBATCH --mem=50G
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
INPUT_GLOB=${INPUT_GLOB:-$REPO_ROOT/results/features/*.h5}
OUT_DIR=${OUT_DIR:-$REPO_ROOT/results/training}
BATCH=${BATCH:-128}
NUM_WORKERS=${NUM_WORKERS:-6}
EXTRA_ARGS=${EXTRA_ARGS:-}

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

# shellcheck disable=SC2086
"$PYTHON" -m deepmod.model \
  --input $INPUT_GLOB \
  --out-dir "$OUT_DIR" \
  --num-workers "$NUM_WORKERS" \
  --batch "$BATCH" \
  $EXTRA_ARGS
