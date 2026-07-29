#!/bin/bash
# rawmod_matched_loco — matched-only causal LOCO with paired-contrastive (SupCon).
# 6 jobs: mixed (in-distribution reference) + loco_{5hmU,4mC,6mA,5mC,5hmC}
# (each holds out one whole chemistry and scores it zero-shot).
#
# Streaming mode (PILEUP_PRELOAD=0): workers read re-chunked HDF5 straight from
# disk, so mem=48G fits qos=high. The matched pool is small (~150k images) so
# even preloading would fit, but streaming keeps it on the plentiful nodes.
#
# Usage (login node):
#   bash run_matched_loco.sh [--dry-run] [--epochs N]
set -euo pipefail

DRY_RUN=false; EPOCHS_ARG=""
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=true ;;
        [0-9]*)   EPOCHS_ARG="--epochs $a" ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${REPO_DIR}/run_matched_loco.py"
OUT=/fs/cbcb-scratch/bds062/results/rawmod_matched_loco
OUTDIR=${OUTDIR:-${OUT}/results1}
LOGDIR=${OUT}/logs
PYTHON=/fs/nexus-scratch/bds062/envs/mod/bin/python
CONDA_INIT="source /nfshomes/bds062/miniconda3/etc/profile.d/conda.sh && conda activate /fs/nexus-scratch/bds062/envs/mod"
mkdir -p "${LOGDIR}"

FOLDS=(mixed loco_5hmU loco_4mC loco_6mA loco_5mC loco_5hmC)
PARTITION=${PARTITION:-cbcb}
# GPU_TYPE empty (default on scavenger) = any GPU type in the partition, for max
# scheduling flexibility; set e.g. GPU_TYPE=rtxa6000 to pin to a specific type.
GPU_TYPE=${GPU_TYPE:-}
GRES="gpu:${GPU_TYPE:+${GPU_TYPE}:}1"
if [ "${PARTITION}" == "scavenger" ]; then
    SLURM_COMMON="--partition=scavenger --account=scavenger --qos=scavenger \
--gres=${GRES} --ntasks=1 --cpus-per-task=10 --mem=48G --time=12:00:00"
else
    SLURM_COMMON="--partition=cbcb --account=cbcb --qos=high \
--gres=gpu:rtxa5000:1 --ntasks=1 --cpus-per-task=10 --mem=48G --time=12:00:00"
fi

submit() { if ${DRY_RUN}; then echo "[dry-run] sbatch $*" >&2; echo 9999; else eval "sbatch --parsable $*"; fi; }

echo "=== rawmod_matched_loco  ->  ${OUTDIR}   epochs=${EPOCHS_ARG:-default}  dry=${DRY_RUN}  partition=${PARTITION} ==="
for FOLD in "${FOLDS[@]}"; do
    WRAP="${CONDA_INIT} && PILEUP_PRELOAD=0 PILEUP_WORKERS=8 \
PILEUP_MASK_BASES=${PILEUP_MASK_BASES:-0} \
SUPCON_DIM=${SUPCON_DIM:-128} SUPCON_WEIGHT=${SUPCON_WEIGHT:-0.1} SUPCON_TEMP=${SUPCON_TEMP:-0.07} \
CURRICULUM=${CURRICULUM:-0} CURRICULUM_EPOCHS=${CURRICULUM_EPOCHS:-15} \
SAD_DIM=${SAD_DIM:-0} SAD_WEIGHT=${SAD_WEIGHT:-1.0} SAD_ETA=${SAD_ETA:-1.0} \
BCE_WEIGHT=${BCE_WEIGHT:-1.0} \
${PYTHON} ${DRIVER} --fold ${FOLD} --out-dir ${OUTDIR} ${EPOCHS_ARG}"
    JID=$(submit "${SLURM_COMMON} --job-name=mloco_${FOLD} \
        --output=${LOGDIR}/${FOLD}_%j.out --error=${LOGDIR}/${FOLD}_%j.out \
        --wrap=\"${WRAP}\"")
    echo "  ${FOLD}: ${JID}"
done
echo "Monitor: squeue -u \$USER   Results: ${OUTDIR}/metrics/"
