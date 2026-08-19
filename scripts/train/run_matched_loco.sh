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

DEFAULT_FOLDS=(mixed loco_5hmU loco_4mC loco_6mA loco_5mC loco_5hmC)
# FOLDS env var overrides the default 6-fold set with a space-separated list,
# e.g. FOLDS="mixed" for a mixed-only sweep run (embedding-space checks only
# need the mixed/in-distribution checkpoint, not the full LOCO battery).
if [ -n "${FOLDS:-}" ]; then
    read -ra FOLDS <<< "${FOLDS}"
else
    FOLDS=("${DEFAULT_FOLDS[@]}")
fi
PARTITION=${PARTITION:-cbcb}
# GPU_TYPE empty (default on scavenger) = any GPU type in the partition, for max
# scheduling flexibility; set e.g. GPU_TYPE=rtxa6000 to pin to a specific type.
GPU_TYPE=${GPU_TYPE:-}
GRES="gpu:${GPU_TYPE:+${GPU_TYPE}:}1"
# Any node CARRYING (not just named legacygpu*) an old pre-Turing card
# (gtxtitanx/gtx1080ti/titanxp/titanxpascal/p6000/p100 -- compute capability <7.5)
# is excluded: those cards are too old for our PyTorch build ("no kernel image
# available" CUDA error), and a node can mix an old card alongside a compatible
# one (e.g. cbcb25 = rtx2080ti + gtx1080ti) so node-name filtering alone missed
# it once. Keyed off GRES GPU model, not node name. Excluded by default whenever
# GPU_TYPE is left open; override EXCLUDE_NODES="" to disable.
EXCLUDE_NODES=${EXCLUDE_NODES-$(sinfo -N -o "%N %G" 2>/dev/null | \
    grep -iE "gtxtitanx|gtx1080ti|titanxp|titanxpascal|p6000|p100" | \
    awk '{print $1}' | sort -u | paste -sd,)}
EXCLUDE_ARG=""
if [ -n "${EXCLUDE_NODES}" ]; then EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"; fi
# DEPENDENCY: SLURM job ID (or colon-separated list) this run's jobs should
# wait on, e.g. a prior featurization job -- handled natively by the SLURM
# scheduler, so it survives this shell/session ending (unlike a local wait
# loop). Default afterok (only starts if the dependency succeeded); override
# via DEPENDENCY_TYPE (e.g. afterany).
DEPENDENCY_ARG=""
if [ -n "${DEPENDENCY:-}" ]; then
    DEPENDENCY_ARG="--dependency=${DEPENDENCY_TYPE:-afterok}:${DEPENDENCY}"
fi
if [ "${PARTITION}" == "scavenger" ]; then
    SLURM_COMMON="--partition=scavenger --account=scavenger --qos=scavenger \
--gres=${GRES} ${EXCLUDE_ARG} ${DEPENDENCY_ARG} --ntasks=1 --cpus-per-task=10 --mem=48G --time=${TIME_LIMIT:-12:00:00}"
else
    # GRES respects GPU_TYPE same as the scavenger branch above (was hardcoded
    # to rtxa5000 only, which stranded jobs when cbcb26 -- the ONLY rtxa5000
    # node on this partition -- was unavailable, while cbcb27 (rtxa6000) and
    # cbcb28-29 (rtx6000ada) sat idle because nothing could request them).
    SLURM_COMMON="--partition=cbcb --account=cbcb --qos=high \
--gres=${GRES} ${EXCLUDE_ARG} ${DEPENDENCY_ARG} --ntasks=1 --cpus-per-task=10 --mem=48G --time=${TIME_LIMIT:-12:00:00}"
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
RAWMOD_DATA_GEN=${RAWMOD_DATA_GEN:-} EXTRA_ORGANISMS=${EXTRA_ORGANISMS:-0} \
${PYTHON} ${DRIVER} --fold ${FOLD} --out-dir ${OUTDIR} ${EPOCHS_ARG}"
    JID=$(submit "${SLURM_COMMON} --job-name=mloco_${FOLD} \
        --output=${LOGDIR}/${FOLD}_%j.out --error=${LOGDIR}/${FOLD}_%j.out \
        --wrap=\"${WRAP}\"")
    echo "  ${FOLD}: ${JID}"
done
echo "Monitor: squeue -u \$USER   Results: ${OUTDIR}/metrics/"
