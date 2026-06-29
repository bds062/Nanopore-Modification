#!/usr/bin/env bash
set -euo pipefail

# End-to-end DeepMod inference on one dataset:
#   POD5 + BAM + peaks/moves -> HDF5 pileup features -> reference predictions
#   -> score summary + thresholded BED calls.

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORK_ROOT=${WORK_ROOT:-/fs/nexus-scratch/bds062}
PYTHON=${PYTHON:-$WORK_ROOT/envs/rockfish/bin/python}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_deepmod_pipeline.sh \
    --dataset NAME \
    --pod5 READS.pod5 \
    --bam READS.bam \
    (--peaks peaks.tsv | --moves moves.tsv) \
    --model best_model.pt \
    --out-dir results/deepmod_NAME \
    [options]

Required:
  --dataset NAME             Dataset label used in output tables.
  --pod5 PATH                POD5 file or directory.
  --bam PATH                 Aligned BAM file.
  --peaks PATH               Per-read segmentation boundaries TSV.
  --moves PATH               Dorado move-table TSV. Use either --peaks or --moves.
  --model PATH               DeepMod checkpoint.
  --out-dir PATH             Output directory.

Optional:
  --level-table PATH         K-mer level table for the reference row.
  --reference PATH           FASTA for annotating ref_base in prediction TSV.
  --gt-bed PATH              Optional BED labels for evaluation/known sites.
  --features-out PATH        Override default HDF5 path.
  --predictions-out PATH     Override default prediction TSV path.
  --summary-dir PATH         Override default summary directory.
  --device NAME              auto, cpu, cuda, cuda:0, etc. Default: auto.
  --batch-size INT           Prediction batch size. Default: 512.
  --threshold FLOAT          Modified-call threshold. Default: 0.5.
  --high-confidence FLOAT    High-confidence summary threshold. Default: 0.9.
  --min-reads INT            Minimum reads per reference position. Default: 5.
  --min-mapq INT             Minimum mapping quality. Default: 0.
  --max-reads INT            Reads per pileup image. Default: 30.
  --max-images-per-base INT  Cap images per reference base. Default: 5.
  --target-base BASE         Featurize only one reference base, e.g. C.
  --target-bases BASES       Featurize only these bases, e.g. AC.
  --half-window INT          Bases on each side of candidate base. Default: 10.
  --samples-per-base INT     Resampled signal length per base. Default: 10.
  --normalize / --no-normalize
                             MAD-normalize read signal. Default: normalize.
  --force                    Recompute features and predictions if outputs exist.
  -h, --help                 Show this help.

Outputs by default:
  OUT_DIR/features/DATASET.h5
  OUT_DIR/predictions/DATASET.deepmod_reference_predictions.tsv
  OUT_DIR/summary/prediction_summary.tsv
  OUT_DIR/summary/score_histogram.{svg,png if rsvg-convert exists}
  OUT_DIR/summary/modified_calls_threshold_<threshold>.bed
  OUT_DIR/logs/{featurize,predict,summarize}.log
EOF
}

DATASET=""
POD5=""
BAM=""
PEAKS=""
MOVES=""
MODEL=""
OUT_DIR=""
LEVEL_TABLE=""
REFERENCE=""
GT_BED=""
FEATURES_OUT=""
PREDICTIONS_OUT=""
SUMMARY_DIR=""
DEVICE="auto"
BATCH_SIZE=512
THRESHOLD=0.5
HIGH_CONFIDENCE_THRESHOLD=0.9
MIN_READS=5
MIN_MAPQ=0
MAX_READS=30
MAX_IMAGES_PER_BASE=5
TARGET_BASE=""
TARGET_BASES=""
HALF_WINDOW=10
SAMPLES_PER_BASE=10
NORMALIZE=1
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET=$2; shift 2 ;;
    --pod5) POD5=$2; shift 2 ;;
    --bam) BAM=$2; shift 2 ;;
    --peaks) PEAKS=$2; shift 2 ;;
    --moves) MOVES=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    --level-table) LEVEL_TABLE=$2; shift 2 ;;
    --reference) REFERENCE=$2; shift 2 ;;
    --gt-bed) GT_BED=$2; shift 2 ;;
    --features-out) FEATURES_OUT=$2; shift 2 ;;
    --predictions-out) PREDICTIONS_OUT=$2; shift 2 ;;
    --summary-dir) SUMMARY_DIR=$2; shift 2 ;;
    --device) DEVICE=$2; shift 2 ;;
    --batch-size) BATCH_SIZE=$2; shift 2 ;;
    --threshold) THRESHOLD=$2; shift 2 ;;
    --high-confidence) HIGH_CONFIDENCE_THRESHOLD=$2; shift 2 ;;
    --min-reads) MIN_READS=$2; shift 2 ;;
    --min-mapq) MIN_MAPQ=$2; shift 2 ;;
    --max-reads) MAX_READS=$2; shift 2 ;;
    --max-images-per-base) MAX_IMAGES_PER_BASE=$2; shift 2 ;;
    --target-base) TARGET_BASE=$2; shift 2 ;;
    --target-bases) TARGET_BASES=$2; shift 2 ;;
    --half-window) HALF_WINDOW=$2; shift 2 ;;
    --samples-per-base) SAMPLES_PER_BASE=$2; shift 2 ;;
    --normalize) NORMALIZE=1; shift ;;
    --no-normalize) NORMALIZE=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

fail() {
  echo "ERROR: $*" >&2
  usage >&2
  exit 2
}

[[ -n "$DATASET" ]] || fail "--dataset is required"
[[ -n "$POD5" ]] || fail "--pod5 is required"
[[ -n "$BAM" ]] || fail "--bam is required"
[[ -n "$MODEL" ]] || fail "--model is required"
[[ -n "$OUT_DIR" ]] || fail "--out-dir is required"
if [[ -n "$PEAKS" && -n "$MOVES" ]]; then
  fail "Use only one of --peaks or --moves"
fi
if [[ -z "$PEAKS" && -z "$MOVES" ]]; then
  fail "One of --peaks or --moves is required"
fi

FEATURES_OUT=${FEATURES_OUT:-$OUT_DIR/features/$DATASET.h5}
PREDICTIONS_OUT=${PREDICTIONS_OUT:-$OUT_DIR/predictions/$DATASET.deepmod_reference_predictions.tsv}
SUMMARY_DIR=${SUMMARY_DIR:-$OUT_DIR/summary}
LOG_DIR=$OUT_DIR/logs

mkdir -p "$(dirname "$FEATURES_OUT")" "$(dirname "$PREDICTIONS_OUT")" "$SUMMARY_DIR" "$LOG_DIR"

test -e "$POD5"
test -f "$BAM"
test -f "$MODEL"
if [[ -n "$PEAKS" ]]; then test -f "$PEAKS"; fi
if [[ -n "$MOVES" ]]; then test -f "$MOVES"; fi
if [[ -n "$LEVEL_TABLE" ]]; then test -f "$LEVEL_TABLE"; fi
if [[ -n "$REFERENCE" ]]; then test -f "$REFERENCE"; fi
if [[ -n "$GT_BED" ]]; then test -f "$GT_BED"; fi

echo "DeepMod pipeline"
echo "  dataset:        $DATASET"
echo "  pod5:           $POD5"
echo "  bam:            $BAM"
echo "  peaks:          ${PEAKS:-none}"
echo "  moves:          ${MOVES:-none}"
echo "  model:          $MODEL"
echo "  level_table:    ${LEVEL_TABLE:-none}"
echo "  features_out:   $FEATURES_OUT"
echo "  predictions:    $PREDICTIONS_OUT"
echo "  summary_dir:    $SUMMARY_DIR"

if [[ "$FORCE" == "1" || ! -f "$FEATURES_OUT" ]]; then
  featurize_cmd=(
    "$PYTHON" -m deepmod.featurization
    --pod5 "$POD5"
    --bam "$BAM"
    --output "$FEATURES_OUT"
    --min-reads "$MIN_READS"
    --min-mapq "$MIN_MAPQ"
    --max-reads "$MAX_READS"
    --max-images-per-base "$MAX_IMAGES_PER_BASE"
    --half-window "$HALF_WINDOW"
    --L "$SAMPLES_PER_BASE"
  )
  if [[ -n "$PEAKS" ]]; then
    featurize_cmd+=(--peaks "$PEAKS")
  else
    featurize_cmd+=(--moves "$MOVES")
  fi
  if [[ "$NORMALIZE" == "1" ]]; then
    featurize_cmd+=(--normalize)
  fi
  if [[ -n "$LEVEL_TABLE" ]]; then
    featurize_cmd+=(--level-table "$LEVEL_TABLE")
  fi
  if [[ -n "$TARGET_BASE" ]]; then
    featurize_cmd+=(--target-base "$TARGET_BASE")
  fi
  if [[ -n "$TARGET_BASES" ]]; then
    featurize_cmd+=(--target-bases "$TARGET_BASES")
  fi
  if [[ -n "$GT_BED" ]]; then
    featurize_cmd+=(--gt "$GT_BED")
  fi

  echo "[$DATASET] featurizing"
  FEATURES_TMP="$FEATURES_OUT.tmp.$$"
  rm -f "$FEATURES_TMP"
  featurize_cmd_tmp=("${featurize_cmd[@]}")
  for i in "${!featurize_cmd_tmp[@]}"; do
    if [[ "${featurize_cmd_tmp[$i]}" == "$FEATURES_OUT" ]]; then
      featurize_cmd_tmp[$i]="$FEATURES_TMP"
      break
    fi
  done
  {
    echo "[$(date '+%F %T')] command:"
    printf ' %q' "${featurize_cmd_tmp[@]}"
    echo
  } > "$LOG_DIR/featurize.log"
  if /usr/bin/time -vpo "$LOG_DIR/featurize.time" "${featurize_cmd_tmp[@]}" >> "$LOG_DIR/featurize.log" 2>&1; then
    mv "$FEATURES_TMP" "$FEATURES_OUT"
  else
    status=$?
    echo "[$DATASET] featurization failed with exit code $status" >&2
    echo "[$DATASET] tail of $LOG_DIR/featurize.log:" >&2
    tail -80 "$LOG_DIR/featurize.log" >&2 || true
    exit "$status"
  fi
else
  echo "[$DATASET] using existing features: $FEATURES_OUT"
fi

if [[ "$FORCE" == "1" || ! -f "$PREDICTIONS_OUT" ]]; then
  predict_cmd=(
    "$PYTHON" "$REPO_ROOT/scripts/evaluation/predict_deepmod_by_reference.py"
    --model "$MODEL"
    --h5 "$FEATURES_OUT"
    --dataset "$DATASET"
    --output "$PREDICTIONS_OUT"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --threshold "$THRESHOLD"
  )
  if [[ -n "$REFERENCE" ]]; then
    predict_cmd+=(--reference "$REFERENCE")
  fi

  echo "[$DATASET] predicting"
  {
    echo "[$(date '+%F %T')] command:"
    printf ' %q' "${predict_cmd[@]}"
    echo
  } > "$LOG_DIR/predict.log"
  if ! /usr/bin/time -vpo "$LOG_DIR/predict.time" "${predict_cmd[@]}" >> "$LOG_DIR/predict.log" 2>&1; then
    status=$?
    echo "[$DATASET] prediction failed with exit code $status" >&2
    tail -80 "$LOG_DIR/predict.log" >&2 || true
    exit "$status"
  fi
else
  echo "[$DATASET] using existing predictions: $PREDICTIONS_OUT"
fi

summary_cmd=(
  "$PYTHON" "$REPO_ROOT/scripts/evaluation/summarize_deepmod_predictions.py"
  --predictions "$PREDICTIONS_OUT"
  --out-dir "$SUMMARY_DIR"
  --threshold "$THRESHOLD"
  --high-confidence-threshold "$HIGH_CONFIDENCE_THRESHOLD"
)
{
  echo "[$(date '+%F %T')] command:"
  printf ' %q' "${summary_cmd[@]}"
  echo
} > "$LOG_DIR/summarize.log"
if ! /usr/bin/time -vpo "$LOG_DIR/summarize.time" "${summary_cmd[@]}" >> "$LOG_DIR/summarize.log" 2>&1; then
  status=$?
  echo "[$DATASET] summarization failed with exit code $status" >&2
  tail -80 "$LOG_DIR/summarize.log" >&2 || true
  exit "$status"
fi

cat > "$OUT_DIR/README.md" <<EOF
# DeepMod Pipeline Output: $DATASET

Generated by \`scripts/run_deepmod_pipeline.sh\`.

## Inputs

- POD5: \`$POD5\`
- BAM: \`$BAM\`
- peaks: \`${PEAKS:-none}\`
- moves: \`${MOVES:-none}\`
- level table: \`${LEVEL_TABLE:-none}\`
- reference FASTA: \`${REFERENCE:-none}\`
- GT BED: \`${GT_BED:-none}\`
- target base: \`${TARGET_BASE:-none}\`
- target bases: \`${TARGET_BASES:-none}\`
- model: \`$MODEL\`

## Outputs

- HDF5 features: \`$FEATURES_OUT\`
- reference predictions: \`$PREDICTIONS_OUT\`
- summary metrics: \`$SUMMARY_DIR/prediction_summary.tsv\`
- score histogram: \`$SUMMARY_DIR/score_histogram.svg\`
- thresholded BED calls: \`$SUMMARY_DIR/modified_calls_threshold_$THRESHOLD.bed\`
- logs: \`$LOG_DIR\`
EOF

echo "[$DATASET] done"
