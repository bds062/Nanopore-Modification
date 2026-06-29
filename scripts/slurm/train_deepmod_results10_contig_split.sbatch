#!/usr/bin/env bash
# Train DeepMod on results9 HDF5 tensors with a contig-level holdout.
#
# Default split:
#   - 30% of contigs are held out for final testing.
#   - 10% of contigs are held out for validation/early stopping.
#   - the remaining contigs are used for weight updates.
#
# Override any variable at submission time, for example:
#   sbatch --export=ALL,TEST_FRAC=0.30,VAL_FRAC=0.10 scripts/slurm/train_deepmod_results10_contig_split.sbatch

#SBATCH --job-name=deepmod_r10_contig
#SBATCH --time=1-04:00:00
#SBATCH --output=/fs/nexus-scratch/bds062/logs/deepmod_results10_%j.out
#SBATCH --qos=huge-long
#SBATCH --mem=50G
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/fs/nexus-scratch/bds062/Nanopore-Modification}
PYTHON=${PYTHON:-/fs/nexus-scratch/bds062/envs/mod/bin/python}
INPUT_GLOB=${INPUT_GLOB:-/fs/nexus-scratch/bds062/results/deep_modification/results9/*.h5}
OUT_DIR=${OUT_DIR:-/fs/nexus-scratch/bds062/results/deep_modification/results10}

BATCH=${BATCH:-128}
NUM_WORKERS=${NUM_WORKERS:-6}
EPOCHS=${EPOCHS:-50}
SEED=${SEED:-42}
VAL_FRAC=${VAL_FRAC:-0.10}
TEST_FRAC=${TEST_FRAC:-0.30}

SKIP_CHANNEL_IMPORTANCE=${SKIP_CHANNEL_IMPORTANCE:-1}
SKIP_LODO=${SKIP_LODO:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}

OPTIONAL_ARGS=()
if [[ "$SKIP_CHANNEL_IMPORTANCE" == "1" ]]; then
  OPTIONAL_ARGS+=(--skip-channel-importance)
fi
if [[ "$SKIP_LODO" == "1" ]]; then
  OPTIONAL_ARGS+=(--skip-lodo)
fi

export MPLCONFIGDIR=${MPLCONFIGDIR:-$OUT_DIR/matplotlib_cache}

mkdir -p /fs/nexus-scratch/bds062/logs "$OUT_DIR" "$MPLCONFIGDIR"
cd "$REPO_ROOT"

echo "=== DeepMod results10 contig-split training ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "Python: $PYTHON"
"$PYTHON" --version
echo "Input glob: $INPUT_GLOB"
echo "Output dir: $OUT_DIR"
echo "Split mode: contig"
echo "Validation contig fraction: $VAL_FRAC"
echo "Test contig fraction: $TEST_FRAC"
echo "Batch size: $BATCH"
echo "Epochs: $EPOCHS"
echo "Num workers: $NUM_WORKERS"
echo "Seed: $SEED"
echo "Skip channel importance: $SKIP_CHANNEL_IMPORTANCE"
echo "Skip LODO: $SKIP_LODO"
echo "Extra args: ${EXTRA_ARGS:-none}"
echo ""

# shellcheck disable=SC2086
"$PYTHON" -m deepmod.model \
  --input $INPUT_GLOB \
  --out-dir "$OUT_DIR" \
  --split-mode contig \
  --val-frac "$VAL_FRAC" \
  --test-frac "$TEST_FRAC" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  "${OPTIONAL_ARGS[@]}" \
  $EXTRA_ARGS

echo ""
echo "DeepMod results10 contig-split training finished: $(date)"
echo "Outputs:"
find "$OUT_DIR" -maxdepth 2 -type f | sort
