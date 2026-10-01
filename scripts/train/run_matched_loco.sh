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
#   bash run_matched_loco.sh [--dry-run] [--epochs N] [--seed N]
set -a; source "${RAWMOD_PATHS_FILE:-$HOME/.config/rawmod/paths.env}" 2>/dev/null || true; set +a   # site paths; see paths.env.example
set -euo pipefail

DRY_RUN=false; EPOCHS_ARG=""; SEED_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --epochs)  EPOCHS_ARG="--epochs $2"; shift 2 ;;
        --seed)    SEED_ARG="--seed $2"; shift 2 ;;
        [0-9]*)    EPOCHS_ARG="--epochs $1"; shift ;;  # bare number = epochs (back-compat)
        *)         shift ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${REPO_DIR}/run_matched_loco.py"
OUT=${RAWMOD_RESULTS}/rawmod_matched_loco
OUTDIR=${OUTDIR:-${OUT}/results1}
LOGDIR=${OUT}/logs
PYTHON=${RAWMOD_ENV}/bin/python
CONDA_INIT="source ${CONDA_SH} && conda activate ${RAWMOD_ENV}"
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
PARTITION=${PARTITION:-${RAWMOD_SLURM_PARTITION:-gpu}}
# GPU_TYPE empty (default on scavenger) = any GPU type in the partition, for max
# scheduling flexibility; set e.g. GPU_TYPE=rtxa6000 to pin to a specific type.
GPU_TYPE=${GPU_TYPE:-}
GRES="gpu:${GPU_TYPE:+${GPU_TYPE}:}1"
# Any node CARRYING (not just named legacygpu*) an old pre-Turing card
# (gtxtitanx/gtx1080ti/titanxp/titanxpascal/p6000/p100 -- compute capability <7.5)
# is excluded: those cards are too old for our PyTorch build ("no kernel image
# available" CUDA error), and a node can mix an old card alongside a compatible
# one (e.g. rtx2080ti + gtx1080ti) so node-name filtering alone can miss
# it. Keyed off GRES GPU model, not node name. Excluded by default whenever
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
# BEGIN: hold these jobs until a time SLURM understands (e.g. "now+40minutes").
# Used to stagger a multi-run battery: 18 jobs starting at once all stream random
# reads from the same NFS feature files and that triggered errno-5 read failures.
BEGIN_ARG=""
if [ -n "${BEGIN:-}" ]; then BEGIN_ARG="--begin=${BEGIN}"; fi
# Partition/account/QoS: PARTITION (default $RAWMOD_SLURM_PARTITION), with account and
# QoS from SLURM_ACCOUNT / SLURM_QOS (default: account = partition, QoS = $RAWMOD_SLURM_QOS
# or none). A "scavenger" partition uses its own account and QoS.
if [ "${PARTITION}" == "scavenger" ]; then
    SLURM_ACCOUNT=${SLURM_ACCOUNT:-scavenger}; SLURM_QOS=${SLURM_QOS:-scavenger}
fi
SLURM_ACCOUNT=${SLURM_ACCOUNT:-${PARTITION}}
SLURM_QOS=${SLURM_QOS:-${RAWMOD_SLURM_QOS:-}}
SLURM_COMMON="--partition=${PARTITION} --account=${SLURM_ACCOUNT} ${SLURM_QOS:+--qos=${SLURM_QOS}} \
--gres=${GRES} ${EXCLUDE_ARG} ${DEPENDENCY_ARG} ${BEGIN_ARG} --ntasks=1 --cpus-per-task=${CPUS:-10} --mem=48G --time=${TIME_LIMIT:-12:00:00}"

submit() { if ${DRY_RUN}; then echo "[dry-run] sbatch $*" >&2; echo 9999; else eval "sbatch --parsable $*"; fi; }

echo "=== rawmod_matched_loco  ->  ${OUTDIR}   epochs=${EPOCHS_ARG:-default}  seed=${SEED_ARG:-default}  dry=${DRY_RUN}  partition=${PARTITION} ==="
for FOLD in "${FOLDS[@]}"; do
    WRAP="${CONDA_INIT} && PILEUP_PRELOAD=0 PILEUP_WORKERS=${PILEUP_WORKERS:-8} \
PILEUP_MASK_BASES=${PILEUP_MASK_BASES:-0} \
SUPCON_DIM=${SUPCON_DIM:-128} SUPCON_WEIGHT=${SUPCON_WEIGHT:-0.1} SUPCON_TEMP=${SUPCON_TEMP:-0.07} \
CURRICULUM=${CURRICULUM:-0} CURRICULUM_EPOCHS=${CURRICULUM_EPOCHS:-15} \
SAD_DIM=${SAD_DIM:-0} SAD_WEIGHT=${SAD_WEIGHT:-1.0} SAD_ETA=${SAD_ETA:-1.0} \
BCE_WEIGHT=${BCE_WEIGHT:-1.0} \
RAWMOD_DATA_GEN=${RAWMOD_DATA_GEN:-} EXTRA_ORGANISMS=${EXTRA_ORGANISMS:-0} \
INCLUDE_HUMAN=${INCLUDE_HUMAN:-0} TF_LAYERS=${TF_LAYERS:-2} ROW_EMB=${ROW_EMB:-1} \
RAWMOD_STRANDRES_ROOT=${RAWMOD_STRANDRES_ROOT:-${RAWMOD_STORE}/rawmod_strand_resolved/features} \
RAWMOD_DROP_CH9=${RAWMOD_DROP_CH9:-0} \
${PYTHON} ${DRIVER} --fold ${FOLD} --out-dir ${OUTDIR} ${EPOCHS_ARG} ${SEED_ARG}"
    JID=$(submit "${SLURM_COMMON} --job-name=mloco_${FOLD} \
        --output=${LOGDIR}/${FOLD}_%j.out --error=${LOGDIR}/${FOLD}_%j.out \
        --wrap=\"${WRAP}\"")
    echo "  ${FOLD}: ${JID}"
done
echo "Monitor: squeue -u \$USER   Results: ${OUTDIR}/metrics/"
