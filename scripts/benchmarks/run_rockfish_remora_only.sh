#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
WORK_ROOT=${WORK_ROOT:-/fs/nexus-scratch/bds062}
TEST_ROOT=${TEST_ROOT:-$WORK_ROOT/results/testing/remora_msssi_cpg_5mc}
PY_ROCKFISH=${PY_ROCKFISH:-$WORK_ROOT/envs/rockfish/bin/python}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
THREADS=${THREADS:-6}
DATASET=${DATASET:-remora_msssi_5mc_rockfish_callable_cpg}
CALLABLE_MODE=${CALLABLE_MODE:-paired}
OUT_DIR=${OUT_DIR:-$TEST_ROOT/rockfish_all_positions_eval}

REMORA_DATA=${REMORA_DATA:-$TEST_ROOT/raw/remora/tests/data}
CAN_BAM=${CAN_BAM:-$REMORA_DATA/can_mappings.bam}
MOD_BAM=${MOD_BAM:-$REMORA_DATA/mod_mappings.bam}
REMORA_MOD_GT=${REMORA_MOD_GT:-$REMORA_DATA/mod_gt.bed}
CPG_BED=${CPG_BED:-$TEST_ROOT/labels/msssi_cpg_5mc.remora_gt.bed}

ROCKFISH_DIR=${ROCKFISH_DIR:-$TEST_ROOT/rockfish}
CAN_PREDS=${CAN_PREDS:-$ROCKFISH_DIR/can/predictions.tsv}
MOD_PREDS=${MOD_PREDS:-$ROCKFISH_DIR/mod/predictions.tsv}
CAN_LABELS=${CAN_LABELS:-$ROCKFISH_DIR/can.reference_labels.tsv}
MOD_LABELS=${MOD_LABELS:-$ROCKFISH_DIR/mod.reference_labels.tsv}

TABLE_DIR=$OUT_DIR/rockfish
TABLE=$TABLE_DIR/$DATASET.rockfish_reference_labels.tsv
SUMMARY=$OUT_DIR/callable_summary.tsv

mkdir -p "$TEST_ROOT/labels" "$TABLE_DIR"
test -f "$CAN_BAM"
test -f "$MOD_BAM"
test -f "$REMORA_MOD_GT"
test -f "$CAN_PREDS"
test -f "$MOD_PREDS"

if [[ ! -f "$CPG_BED" ]]; then
  cp "$REMORA_MOD_GT" "$CPG_BED"
fi

if [[ ! -f "$CAN_LABELS" ]]; then
  "$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/aggregate_rockfish_by_reference.py" \
    --dataset can \
    --predictions "$CAN_PREDS" \
    --bam "$CAN_BAM" \
    --output "$CAN_LABELS" \
    --mapq-min 0 \
    --threads "$THREADS" \
    --covered-only
fi

if [[ ! -f "$MOD_LABELS" ]]; then
  "$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/aggregate_rockfish_by_reference.py" \
    --dataset mod \
    --predictions "$MOD_PREDS" \
    --bam "$MOD_BAM" \
    --gt-bed "$CPG_BED" \
    --output "$MOD_LABELS" \
    --mapq-min 0 \
    --threads "$THREADS" \
    --covered-only
fi

"$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/make_rockfish_callable_cpg_table.py" \
  --dataset "$DATASET" \
  --cpg-bed "$CPG_BED" \
  --can "$CAN_LABELS" \
  --mod "$MOD_LABELS" \
  --output "$TABLE" \
  --summary "$SUMMARY" \
  --callable-mode "$CALLABLE_MODE"

"$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/plot_method_comparison.py" \
  --rockfish-dir "$TABLE_DIR" \
  --out-dir "$OUT_DIR" \
  --datasets "$DATASET"

"$PY_ROCKFISH" "$REPO_ROOT/scripts/evaluation/plot_class_breakdown.py" \
  --input "$TABLE" \
  --out-dir "$OUT_DIR/class_breakdown" \
  --method-name Rockfish \
  --require-both-classes

echo "Wrote Rockfish-only callable-CpG Remora evaluation to $OUT_DIR"
