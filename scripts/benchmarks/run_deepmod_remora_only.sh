#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
WORK_ROOT=${WORK_ROOT:-/fs/nexus-scratch/bds062}
TEST_ROOT=${TEST_ROOT:-$WORK_ROOT/results/testing/remora_msssi_cpg_5mc}
PY_ROCKFISH=${PY_ROCKFISH:-$WORK_ROOT/envs/rockfish/bin/python}
PY_DEEPMOD=${PY_DEEPMOD:-$WORK_ROOT/envs/rockfish/bin/python}
DEEPMOD_MODEL=${DEEPMOD_MODEL:-$WORK_ROOT/results/deep_modification/results6/best_model.pt}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
THREADS=${THREADS:-6}
DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-512}
DATASET=${DATASET:-remora_msssi_5mc_deepmod_all_positions}
FEATURE_DIR=${FEATURE_DIR:-$TEST_ROOT/deepmod_all_positions}
OUT_DIR=${OUT_DIR:-$TEST_ROOT/deepmod_all_positions_eval}

REMORA_DATA=${REMORA_DATA:-$TEST_ROOT/raw/remora/tests/data}
CAN_POD5=${CAN_POD5:-$REMORA_DATA/can_reads.pod5}
MOD_POD5=${MOD_POD5:-$REMORA_DATA/mod_reads.pod5}
CAN_BAM=${CAN_BAM:-$REMORA_DATA/can_mappings.bam}
MOD_BAM=${MOD_BAM:-$REMORA_DATA/mod_mappings.bam}
REMORA_MOD_GT=${REMORA_MOD_GT:-$REMORA_DATA/mod_gt.bed}
LEVEL_TABLE=${LEVEL_TABLE:-$REMORA_DATA/levels.txt}
CPG_BED=${CPG_BED:-$TEST_ROOT/labels/msssi_cpg_5mc.remora_gt.bed}

MOVES_CAN=$FEATURE_DIR/can.moves.tsv
MOVES_MOD=$FEATURE_DIR/mod.moves.tsv
H5_CAN=$FEATURE_DIR/can.h5
H5_MOD=$FEATURE_DIR/mod.h5
PRED_CAN=$FEATURE_DIR/can.reference_predictions.tsv
PRED_MOD=$FEATURE_DIR/mod.reference_predictions.tsv
TABLE_DIR=$OUT_DIR/deepmod
TABLE=$TABLE_DIR/$DATASET.deepmod_reference_predictions.tsv

mkdir -p "$TEST_ROOT/labels" "$FEATURE_DIR" "$TABLE_DIR"
test -f "$CAN_POD5"
test -f "$MOD_POD5"
test -f "$CAN_BAM"
test -f "$MOD_BAM"
test -f "$REMORA_MOD_GT"
test -f "$LEVEL_TABLE"
test -f "$DEEPMOD_MODEL"

if [[ ! -f "$CPG_BED" ]]; then
  cp "$REMORA_MOD_GT" "$CPG_BED"
fi

if [[ ! -f "$MOVES_CAN" ]]; then
  "$PY_DEEPMOD" "$REPO_ROOT/scripts/extract_moves_from_bam.py" \
    --bam "$CAN_BAM" \
    --output "$MOVES_CAN"
fi

if [[ ! -f "$MOVES_MOD" ]]; then
  "$PY_DEEPMOD" "$REPO_ROOT/scripts/extract_moves_from_bam.py" \
    --bam "$MOD_BAM" \
    --output "$MOVES_MOD"
fi

if [[ ! -f "$H5_CAN" ]]; then
  "$PY_DEEPMOD" -m deepmod.featurization \
    --pod5 "$CAN_POD5" \
    --bam "$CAN_BAM" \
    --moves "$MOVES_CAN" \
    --level-table "$LEVEL_TABLE" \
    --output "$H5_CAN" \
    --normalize \
    --gt \
    --min-mapq 0 \
    --min-reads 5 \
    --max-reads 30 \
    --max-images-per-base 5
fi

if [[ ! -f "$H5_MOD" ]]; then
  "$PY_DEEPMOD" -m deepmod.featurization \
    --pod5 "$MOD_POD5" \
    --bam "$MOD_BAM" \
    --moves "$MOVES_MOD" \
    --level-table "$LEVEL_TABLE" \
    --output "$H5_MOD" \
    --normalize \
    --gt "$CPG_BED" \
    --min-mapq 0 \
    --min-reads 5 \
    --max-reads 30 \
    --max-images-per-base 5
fi

if [[ ! -f "$PRED_CAN" ]]; then
  "$PY_DEEPMOD" "$REPO_ROOT/scripts/evaluation/predict_deepmod_by_reference.py" \
    --model "$DEEPMOD_MODEL" \
    --h5 "$H5_CAN" \
    --dataset can \
    --output "$PRED_CAN" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"
fi

if [[ ! -f "$PRED_MOD" ]]; then
  "$PY_DEEPMOD" "$REPO_ROOT/scripts/evaluation/predict_deepmod_by_reference.py" \
    --model "$DEEPMOD_MODEL" \
    --h5 "$H5_MOD" \
    --dataset mod \
    --output "$PRED_MOD" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE"
fi

"$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/make_deepmod_sample_table.py" \
  --dataset "$DATASET" \
  --cpg-bed "$CPG_BED" \
  --can "$PRED_CAN" \
  --mod "$PRED_MOD" \
  --output "$TABLE" \
  --no-cpg-filter \
  --use-source-labels

"$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/plot_method_comparison.py" \
  --deepmod-dir "$TABLE_DIR" \
  --out-dir "$OUT_DIR" \
  --datasets "$DATASET"

"$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/plot_class_breakdown.py" \
  --input "$TABLE" \
  --out-dir "$OUT_DIR/class_breakdown"

echo "Wrote DeepMod-only Remora evaluation to $OUT_DIR"
