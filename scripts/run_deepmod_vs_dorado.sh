#!/usr/bin/env bash
set -euo pipefail

# Run DeepMod on a dataset, run compatible Dorado DNA modified-base models,
# convert Dorado MM/ML tags into reference-level labels, and compare DeepMod
# scores against Dorado labels.

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORK_ROOT=${WORK_ROOT:-/fs/nexus-scratch/bds062}
PYTHON=${PYTHON:-$WORK_ROOT/envs/rockfish/bin/python}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DEFAULT_DORADO_DIR=/fs/cbcb-lab/storm/shared/rawhash2/basecallers/dorado-1.4.0-linux-x64/bin

usage() {
  cat <<'EOF'
Usage:
  scripts/run_deepmod_vs_dorado.sh \
    --dataset NAME \
    --pod5 READS.pod5 \
    --bam READS.bam \
    (--peaks peaks.tsv | --moves moves.tsv | --deepmod-predictions predictions.tsv) \
    --reference ref.fa \
    --deepmod-model best_model.pt \
    --out-dir results/deepmod_vs_dorado_NAME \
    [options]

Required:
  --dataset NAME             Dataset label.
  --pod5 PATH                POD5 file/directory used by Dorado and DeepMod.
  --bam PATH                 Existing aligned BAM for DeepMod featurization.
  --peaks PATH               Peak boundaries TSV for DeepMod. Use --peaks or --moves.
  --moves PATH               Dorado move-table TSV for DeepMod. Use --peaks or --moves.
  --deepmod-predictions PATH Reuse existing DeepMod reference predictions instead of
                             running DeepMod featurization/inference.
  --reference PATH           Reference FASTA. Required because Dorado output must be aligned.
  --deepmod-model PATH       DeepMod checkpoint, unless --deepmod-predictions is supplied.
  --out-dir PATH             Output directory.

Optional:
  --level-table PATH         K-mer level table for DeepMod reference channel.
  --dorado-bin PATH          Dorado binary.
  --dorado-model PATH        Basecalling model name/path. Default: R10.4.1 sup@v5.2.0.
  --dorado-model-dir PATH    Directory containing Dorado model folders.
  --mod-model PATH           Explicit Dorado modified-base model. May be repeated.
  --all-compatible-mod-models
                             Run every compatible mod model for the base Dorado model.
                             Default: latest version per compatible mod-model family.
  --device NAME              auto, cpu, cuda:0, cuda:all. Default: auto.
  --deepmod-device NAME      auto, cpu, cuda, cuda:0. Default: same as --device.
  --dorado-threshold FLOAT   Dorado per-reference label threshold. Default: 0.5.
  --dorado-emit-threshold FLOAT
                             Dorado MM/ML emission threshold. Default: 0, so
                             low-probability candidate calls remain available
                             for downstream binary labeling.
  --deepmod-threshold FLOAT  DeepMod threshold for confusion matrices. Default: 0.5.
  --min-mapq INT             Minimum MAPQ for Dorado label aggregation and DeepMod. Default: 0.
  --min-calls INT            Minimum Dorado MM/ML calls per reference position. Default: 1.
  --min-reads INT            Minimum reads per DeepMod pileup position. Default: 5.
  --max-reads INT            Reads per DeepMod pileup image. Default: 30.
  --max-images-per-base INT  DeepMod image cap per reference position. Default: 5.
  --force                    Recompute existing outputs.
  -h, --help                 Show this help.

Outputs:
  OUT_DIR/deepmod/                         DeepMod pipeline output, unless reused.
  OUT_DIR/dorado/MODEL/reads.bam           Dorado BAM with MM/ML calls.
  OUT_DIR/dorado/MODEL/*.labels.tsv        Dorado reference-level labels.
  OUT_DIR/comparison/MODEL/MOD_TYPE/       Joined table, metrics, PR/F1, confusion.
  OUT_DIR/comparison/metrics_by_mod.tsv    One metrics row per model/mod type.
  OUT_DIR/comparison/plots/summary.svg     Accuracy/F1/AUPRC summary plot.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  usage >&2
  exit 2
}

sanitize() {
  basename "$1" | tr -c 'A-Za-z0-9._-' '_'
}

discover_mod_types() {
  local name="$1"
  local found=0
  if [[ "$name" == *"4mC"* ]]; then echo "4mC"; found=1; fi
  if [[ "$name" == *"5mC"* || "$name" == *"5mCG"* ]]; then echo "5mC"; found=1; fi
  if [[ "$name" == *"5hmC"* || "$name" == *"5hmCG"* ]]; then echo "5hmC"; found=1; fi
  if [[ "$name" == *"6mA"* ]]; then echo "6mA"; found=1; fi
  [[ "$found" == "1" ]]
}

discover_mod_models() {
  local base_name="$1"
  local mode="$2"
  local model
  local family
  declare -A latest=()
  while IFS= read -r model; do
    family=$(basename "$model")
    family=${family#${base_name}_}
    family=${family%@v*}
    if [[ "$mode" == "all" ]]; then
      echo "$model"
    else
      latest["$family"]="$model"
    fi
  done < <(
    find "$DORADO_MODEL_DIR" -maxdepth 1 -type d -name "${base_name}_*" \
      | grep -E '_(4mC_5mC|5mCG_5hmCG|5mC_5hmC|6mA)@v' \
      | sort -V
  )
  if [[ "$mode" != "all" ]]; then
    for family in "${!latest[@]}"; do
      echo "${latest[$family]}"
    done | sort
  fi
}

DATASET=""
POD5=""
BAM=""
PEAKS=""
MOVES=""
REFERENCE=""
DEEPMOD_MODEL=""
DEEPMOD_PREDICTIONS=""
LEVEL_TABLE=""
OUT_DIR=""
DORADO_BIN="$DEFAULT_DORADO_DIR/dorado"
DORADO_MODEL_DIR="$DEFAULT_DORADO_DIR"
DORADO_MODEL="$DEFAULT_DORADO_DIR/dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
DEVICE="auto"
DEEPMOD_DEVICE=""
DORADO_THRESHOLD=0.5
DORADO_EMIT_THRESHOLD=0
DEEPMOD_THRESHOLD=0.5
MIN_MAPQ=0
MIN_CALLS=1
MIN_READS=5
MAX_READS=30
MAX_IMAGES_PER_BASE=5
FORCE=0
ALL_COMPATIBLE=0
MOD_MODELS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET=$2; shift 2 ;;
    --pod5) POD5=$2; shift 2 ;;
    --bam) BAM=$2; shift 2 ;;
    --peaks) PEAKS=$2; shift 2 ;;
    --moves) MOVES=$2; shift 2 ;;
    --reference) REFERENCE=$2; shift 2 ;;
    --deepmod-model|--model) DEEPMOD_MODEL=$2; shift 2 ;;
    --deepmod-predictions) DEEPMOD_PREDICTIONS=$2; shift 2 ;;
    --level-table) LEVEL_TABLE=$2; shift 2 ;;
    --out-dir) OUT_DIR=$2; shift 2 ;;
    --dorado-bin) DORADO_BIN=$2; shift 2 ;;
    --dorado-model) DORADO_MODEL=$2; shift 2 ;;
    --dorado-model-dir) DORADO_MODEL_DIR=$2; shift 2 ;;
    --mod-model) MOD_MODELS+=("$2"); shift 2 ;;
    --all-compatible-mod-models) ALL_COMPATIBLE=1; shift ;;
    --device) DEVICE=$2; shift 2 ;;
    --deepmod-device) DEEPMOD_DEVICE=$2; shift 2 ;;
    --dorado-threshold) DORADO_THRESHOLD=$2; shift 2 ;;
    --dorado-emit-threshold) DORADO_EMIT_THRESHOLD=$2; shift 2 ;;
    --deepmod-threshold) DEEPMOD_THRESHOLD=$2; shift 2 ;;
    --min-mapq) MIN_MAPQ=$2; shift 2 ;;
    --min-calls) MIN_CALLS=$2; shift 2 ;;
    --min-reads) MIN_READS=$2; shift 2 ;;
    --max-reads) MAX_READS=$2; shift 2 ;;
    --max-images-per-base) MAX_IMAGES_PER_BASE=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

[[ -n "$DATASET" ]] || fail "--dataset is required"
[[ -n "$POD5" ]] || fail "--pod5 is required"
[[ -n "$BAM" ]] || fail "--bam is required"
[[ -n "$REFERENCE" ]] || fail "--reference is required for reference-level Dorado comparison"
[[ -n "$OUT_DIR" ]] || fail "--out-dir is required"
if [[ -z "$DEEPMOD_PREDICTIONS" ]]; then
  [[ -n "$DEEPMOD_MODEL" ]] || fail "--deepmod-model is required unless --deepmod-predictions is supplied"
  if [[ -n "$PEAKS" && -n "$MOVES" ]]; then fail "Use only one of --peaks or --moves"; fi
  if [[ -z "$PEAKS" && -z "$MOVES" ]]; then fail "Use --peaks, --moves, or --deepmod-predictions"; fi
fi

test -e "$POD5"
test -f "$BAM"
test -f "$REFERENCE"
test -f "$DORADO_BIN"
test -d "$DORADO_MODEL_DIR"
if [[ -n "$DEEPMOD_MODEL" ]]; then test -f "$DEEPMOD_MODEL"; fi
if [[ -n "$DEEPMOD_PREDICTIONS" ]]; then test -f "$DEEPMOD_PREDICTIONS"; fi
if [[ -n "$PEAKS" ]]; then test -f "$PEAKS"; fi
if [[ -n "$MOVES" ]]; then test -f "$MOVES"; fi
if [[ -n "$LEVEL_TABLE" ]]; then test -f "$LEVEL_TABLE"; fi

if [[ -z "$DEEPMOD_DEVICE" ]]; then
  if [[ "$DEVICE" == "cuda:all" ]]; then
    DEEPMOD_DEVICE="cuda"
  else
    DEEPMOD_DEVICE="$DEVICE"
  fi
fi
mkdir -p "$OUT_DIR"

if [[ "${#MOD_MODELS[@]}" -eq 0 ]]; then
  BASE_MODEL_NAME=$(basename "$DORADO_MODEL")
  DISCOVERY_MODE="latest"
  if [[ "$ALL_COMPATIBLE" == "1" ]]; then
    DISCOVERY_MODE="all"
  fi
  mapfile -t MOD_MODELS < <(discover_mod_models "$BASE_MODEL_NAME" "$DISCOVERY_MODE")
fi
[[ "${#MOD_MODELS[@]}" -gt 0 ]] || fail "No compatible Dorado DNA modified-base models found"

echo "DeepMod vs Dorado"
echo "  dataset:          $DATASET"
echo "  pod5:             $POD5"
echo "  deepmod bam:      $BAM"
echo "  reference:        $REFERENCE"
echo "  output:           $OUT_DIR"
echo "  dorado binary:    $DORADO_BIN"
echo "  dorado model:     $DORADO_MODEL"
echo "  dorado threshold: $DORADO_THRESHOLD"
echo "  dorado emit thr.: $DORADO_EMIT_THRESHOLD"
echo "  deepmod threshold:$DEEPMOD_THRESHOLD"
echo "  mod models:"
printf '    %s\n' "${MOD_MODELS[@]}"

if [[ -z "$DEEPMOD_PREDICTIONS" ]]; then
  DEEPMOD_OUT="$OUT_DIR/deepmod"
  DEEPMOD_PREDICTIONS="$DEEPMOD_OUT/predictions/$DATASET.deepmod_reference_predictions.tsv"
  deepmod_cmd=(
    "$REPO_ROOT/scripts/run_deepmod_pipeline.sh"
    --dataset "$DATASET"
    --pod5 "$POD5"
    --bam "$BAM"
    --model "$DEEPMOD_MODEL"
    --out-dir "$DEEPMOD_OUT"
    --reference "$REFERENCE"
    --device "$DEEPMOD_DEVICE"
    --threshold "$DEEPMOD_THRESHOLD"
    --min-mapq "$MIN_MAPQ"
    --min-reads "$MIN_READS"
    --max-reads "$MAX_READS"
    --max-images-per-base "$MAX_IMAGES_PER_BASE"
  )
  if [[ -n "$PEAKS" ]]; then deepmod_cmd+=(--peaks "$PEAKS"); fi
  if [[ -n "$MOVES" ]]; then deepmod_cmd+=(--moves "$MOVES"); fi
  if [[ -n "$LEVEL_TABLE" ]]; then deepmod_cmd+=(--level-table "$LEVEL_TABLE"); fi
  if [[ "$FORCE" == "1" ]]; then deepmod_cmd+=(--force); fi
  echo "[$DATASET] running DeepMod pipeline"
  "${deepmod_cmd[@]}"
else
  echo "[$DATASET] using existing DeepMod predictions: $DEEPMOD_PREDICTIONS"
fi

COMPARISON_DIR="$OUT_DIR/comparison"
METRICS_ALL="$COMPARISON_DIR/metrics_by_mod.tsv"
rm -f "$METRICS_ALL.tmp"

for MOD_MODEL in "${MOD_MODELS[@]}"; do
  test -d "$MOD_MODEL"
  MODEL_NAME=$(basename "$MOD_MODEL")
  MODEL_SAFE=$(sanitize "$MODEL_NAME")
  MODEL_OUT="$OUT_DIR/dorado/$MODEL_SAFE"
  MODEL_COMPARISON="$COMPARISON_DIR/$MODEL_SAFE"
  DORADO_BAM="$MODEL_OUT/reads.bam"
  mkdir -p "$MODEL_OUT" "$MODEL_COMPARISON" "$MODEL_OUT/logs"

  if [[ "$FORCE" == "1" || ! -f "$DORADO_BAM" ]]; then
    dorado_cmd=("$DORADO_BIN" basecaller)
    if [[ "$DEVICE" != "auto" ]]; then dorado_cmd+=("-x" "$DEVICE"); fi
    dorado_cmd+=(
      --emit-moves
      --disable-read-splitting
      --reference "$REFERENCE"
      --modified-bases-models "$MOD_MODEL"
      --modified-bases-threshold "$DORADO_EMIT_THRESHOLD"
      "$DORADO_MODEL"
      "$POD5"
    )
    echo "[$DATASET] running Dorado model: $MODEL_NAME"
    /usr/bin/time -vpo "$MODEL_OUT/basecall.time" "${dorado_cmd[@]}" > "$DORADO_BAM" 2> "$MODEL_OUT/logs/basecall.err"
  else
    echo "[$DATASET] using existing Dorado BAM: $DORADO_BAM"
  fi

  mapfile -t MOD_TYPES < <(discover_mod_types "$MODEL_NAME")
  for MOD_TYPE in "${MOD_TYPES[@]}"; do
    LABELS="$MODEL_OUT/$MOD_TYPE.dorado_reference_labels.tsv"
    KEY_SUMMARY="$MODEL_OUT/$MOD_TYPE.modified_base_keys.tsv"
    MOD_COMPARE_DIR="$MODEL_COMPARISON/$MOD_TYPE"
    if [[ "$FORCE" == "1" || ! -f "$LABELS" ]]; then
      echo "[$DATASET] aggregating Dorado $MOD_TYPE labels from $MODEL_NAME"
      "$PYTHON" "$REPO_ROOT/scripts/evaluation/extract_dorado_mods_by_reference.py" \
        --bam "$DORADO_BAM" \
        --dataset "$DATASET" \
        --mod-model-name "$MODEL_NAME" \
        --mod-type "$MOD_TYPE" \
        --reference "$REFERENCE" \
        --output "$LABELS" \
        --threshold "$DORADO_THRESHOLD" \
        --min-mapq "$MIN_MAPQ" \
        --min-calls "$MIN_CALLS" \
        --key-summary "$KEY_SUMMARY" \
        > "$MODEL_OUT/logs/extract_$MOD_TYPE.log" 2>&1
    fi

    echo "[$DATASET] comparing DeepMod to Dorado $MOD_TYPE for $MODEL_NAME"
    "$PYTHON" "$REPO_ROOT/scripts/evaluation/compare_deepmod_to_dorado.py" \
      --deepmod "$DEEPMOD_PREDICTIONS" \
      --dorado "$LABELS" \
      --out-dir "$MOD_COMPARE_DIR" \
      --dataset "$DATASET" \
      --mod-model-name "$MODEL_NAME" \
      --mod-type "$MOD_TYPE" \
      --deepmod-threshold "$DEEPMOD_THRESHOLD" \
      > "$MOD_COMPARE_DIR.compare.log" 2>&1

    METRICS_FILE="$MOD_COMPARE_DIR/$MOD_TYPE.metrics.tsv"
    if [[ ! -f "$METRICS_ALL.tmp" ]]; then
      head -n 1 "$METRICS_FILE" > "$METRICS_ALL.tmp"
    fi
    tail -n +2 "$METRICS_FILE" >> "$METRICS_ALL.tmp"
  done
done

mkdir -p "$COMPARISON_DIR/plots"
mv "$METRICS_ALL.tmp" "$METRICS_ALL"
"$PYTHON" "$REPO_ROOT/scripts/evaluation/plot_dorado_comparison_summary.py" \
  --metrics "$METRICS_ALL" \
  --output "$COMPARISON_DIR/plots/summary.svg" \
  --title "$DATASET: DeepMod vs Dorado Modified-Base Calls"

cat > "$OUT_DIR/README.md" <<EOF
# DeepMod vs Dorado: $DATASET

Generated by \`scripts/run_deepmod_vs_dorado.sh\`.

## Inputs

- POD5: \`$POD5\`
- DeepMod BAM: \`$BAM\`
- peaks: \`${PEAKS:-none}\`
- moves: \`${MOVES:-none}\`
- reference FASTA: \`$REFERENCE\`
- DeepMod checkpoint: \`${DEEPMOD_MODEL:-reused predictions}\`
- DeepMod predictions: \`$DEEPMOD_PREDICTIONS\`
- Dorado binary: \`$DORADO_BIN\`
- Dorado base model: \`$DORADO_MODEL\`
- Dorado label threshold: \`$DORADO_THRESHOLD\`
- Dorado MM/ML emission threshold: \`$DORADO_EMIT_THRESHOLD\`
- DeepMod threshold: \`$DEEPMOD_THRESHOLD\`

## Outputs

- DeepMod output: \`$OUT_DIR/deepmod\`
- Dorado BAMs and reference labels: \`$OUT_DIR/dorado\`
- Per-model/per-type comparisons: \`$COMPARISON_DIR\`
- Summary metrics: \`$METRICS_ALL\`
- Summary plot: \`$COMPARISON_DIR/plots/summary.svg\`

Interpretation note: Dorado MM/ML calls are treated as reference labels here.
DeepMod is still a binary modified-position model, so 5mC, 5hmC, 6mA, and 4mC
comparisons ask whether DeepMod assigns high modified probability at positions
Dorado labels as that modification type.
EOF

echo "[$DATASET] done"
echo "Metrics: $METRICS_ALL"
echo "Summary plot: $COMPARISON_DIR/plots/summary.svg"
