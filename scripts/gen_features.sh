#!/usr/bin/env bash
set -euo pipefail

# Featurize the ONT modified-base benchmark into DeepMod HDF5 pileup tensors.
# Paths default to the current /fs/nexus-scratch layout but can be overridden
# with environment variables.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-/fs/nexus-scratch/bds062}
PYTHON=${PYTHON:-python}

OUT_DIR=${OUT_DIR:-$REPO_ROOT/results/features}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/ont-os}
EVENT_ROOT=${EVENT_ROOT:-$PROJECT_ROOT/results}
LEVEL_TABLE=${LEVEL_TABLE:-$PROJECT_ROOT/results/uncalled_r1041_model_only_means.txt}

MAX_READS=${MAX_READS:-30}
MAX_IMAGES_PER_BASE=${MAX_IMAGES_PER_BASE:-5}
MIN_READS=${MIN_READS:-5}
MIN_MAPQ=${MIN_MAPQ:-60}
HALF_WINDOW=${HALF_WINDOW:-10}
SAMPLES_PER_BASE=${SAMPLES_PER_BASE:-10}
NORMALIZE=${NORMALIZE:-1}

SLURM_CPUS=${SLURM_CPUS:-4}
SLURM_MEM=${SLURM_MEM:-64GB}
SLURM_QOS=${SLURM_QOS:-medium}
SLURM_PARTITION=${SLURM_PARTITION:-cbcb}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-cbcb}
SLURM_TIME=${SLURM_TIME:-04:00:00}
USE_SRUN=${USE_SRUN:-auto}

mkdir -p "$OUT_DIR"

run_featurize() {
  local name=$1
  local pod5=$2
  local bam=$3
  local peaks=$4
  local gt_bed=${5:-}
  local out_h5="$OUT_DIR/${name}.h5"
  local log="$OUT_DIR/${name}.out"

  local cmd=(
    "$PYTHON" -m deepmod.featurization
    --pod5 "$pod5"
    --bam "$bam"
    --peaks "$peaks"
    --level-table "$LEVEL_TABLE"
    --output "$out_h5"
    --half-window "$HALF_WINDOW"
    --L "$SAMPLES_PER_BASE"
    --min-reads "$MIN_READS"
    --min-mapq "$MIN_MAPQ"
    --max-reads "$MAX_READS"
    --max-images-per-base "$MAX_IMAGES_PER_BASE"
  )

  if [[ "$NORMALIZE" == "1" ]]; then
    cmd+=(--normalize)
  fi

  if [[ -n "$gt_bed" ]]; then
    cmd+=(--gt "$gt_bed")
  else
    cmd+=(--gt)
  fi

  echo "[$name] writing $out_h5"
  if [[ "$USE_SRUN" == "1" || ( "$USE_SRUN" == "auto" && -n "${SLURM_JOB_ID:-}" ) ]]; then
    srun -c "$SLURM_CPUS" \
      --mem="$SLURM_MEM" \
      --qos="$SLURM_QOS" \
      --partition="$SLURM_PARTITION" \
      --account="$SLURM_ACCOUNT" \
      --time "$SLURM_TIME" \
      "${cmd[@]}" > "$log" 2>&1
  else
    "${cmd[@]}" > "$log" 2>&1
  fi
}

run_featurize \
  control \
  "$DATA_ROOT/subset_control/control_rep1.pod5" \
  "$EVENT_ROOT/event_clustering_control/basecalled/reads_refined.bam" \
  "$EVENT_ROOT/event_clustering_control/basecalled/peaks_refined.tsv"

run_featurize \
  5mC \
  "$DATA_ROOT/subset_5mC/5mC_rep1.pod5" \
  "$EVENT_ROOT/event_clustering_5mC/basecalled/reads_refined.bam" \
  "$EVENT_ROOT/event_clustering_5mC/basecalled/peaks_refined.tsv" \
  "$DATA_ROOT/references/all_5mers_5mC_sites.bed"

run_featurize \
  5hmC \
  "$DATA_ROOT/subset_5hmC/5hmC_rep1.pod5" \
  "$EVENT_ROOT/event_clustering_5hmC/basecalled/reads_refined.bam" \
  "$EVENT_ROOT/event_clustering_5hmC/basecalled/peaks_refined.tsv" \
  "$DATA_ROOT/references/all_5mers_5hmC_sites.bed"

run_featurize \
  6mA \
  "$DATA_ROOT/subset_6mA/6mA_rep1.pod5" \
  "$EVENT_ROOT/event_clustering_6mA/basecalled/reads_refined.bam" \
  "$EVENT_ROOT/event_clustering_6mA/basecalled/peaks_refined.tsv" \
  "$DATA_ROOT/references/all_5mers_6mA_sites.bed"
