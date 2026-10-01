#!/bin/bash
# Inference-mode featurization of one candidate chunk on one strand (STRAND exported).
# Flags reproduce the strand-resolved training features: MAD --normalize, 15 reads,
# 21-base window, L=10, one image per site; minus strand is mirrored with --orient read.
source "${GM_DIR:?}/steps/common.sh"   # sbatch runs a spooled copy, so not $0
set +u; source "$CONDA_SH"; conda activate "$PYTHON_ENV"; set -u
s=$(sdir "$STRAND"); i=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
cand="$OUT/candidates_$s/chunk_$i.tsv"
[[ -s "$cand" ]] || { echo "no chunk $cand (fewer sites than CHUNKS); nothing to do"; exit 0; }
if [[ "$STRAND" == "+" ]]; then tb=$TARGET_BASE; orient=(); else tb=$(comp "$TARGET_BASE"); orient=(--orient read); fi
mkdir -p "$OUT/feat_$s"
python "$REPO/rawmod/featurization.py" \
  --pod5 "$POD5" --bam "$PRE/reads_refined.bam" --peaks "$PRE/peaks_refined.tsv" \
  --output "$OUT/feat_$s/chunk_$i.h5" --level-table "$LEVELS" \
  --candidate-bed "$cand" --target-base "$tb" --strand "$STRAND" "${orient[@]}" \
  --max-reads "$MAX_READS" --min-reads "$MIN_READS" --min-mapq "$MIN_MAPQ" \
  --half-window 10 --L 10 --normalize --max-images-per-base 1
