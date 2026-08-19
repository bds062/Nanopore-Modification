#!/bin/bash
# rawmod DANN two-head experiment (results7 adv-target=type, results8
# adv-target=dataset). 6 jobs: mixed (in-distribution reference) +
# loco_{5hmU,4mC,6mA,5mC,5hmC} (each holds out one whole chemistry, zero-shot).
#
# Usage (login node):
#   ADV_TARGET=type OUTDIR=.../results7 bash run_dann_loco.sh [--epochs N]
#   ADV_TARGET=dataset OUTDIR=.../results8 bash run_dann_loco.sh [--epochs N]
set -euo pipefail

DRY_RUN=false; EPOCHS_ARG=""
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=true ;;
        [0-9]*)   EPOCHS_ARG="--epochs $a" ;;
    esac
done

ADV_TARGET=${ADV_TARGET:?set ADV_TARGET=type or ADV_TARGET=dataset}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="${REPO_DIR}/run_dann_loco.py"
OUT=/fs/cbcb-scratch/bds062/results/rawmod_matched_loco
OUTDIR=${OUTDIR:?set OUTDIR (e.g. ${OUT}/results7)}
LOGDIR=${OUTDIR}/logs
PYTHON=/fs/nexus-scratch/bds062/envs/mod/bin/python
CONDA_INIT="source /nfshomes/bds062/miniconda3/etc/profile.d/conda.sh && conda activate /fs/nexus-scratch/bds062/envs/mod"
mkdir -p "${LOGDIR}"

FOLDS=(mixed loco_5hmU loco_4mC loco_6mA loco_5mC loco_5hmC)

# Any node CARRYING (not just named legacygpu*) an old pre-Turing card is
# excluded -- see run_matched_loco.sh for the full rationale (cbcb25 mixes an
# rtx2080ti with a gtx1080ti, so name-based filtering alone missed it once).
EXCLUDE_NODES=${EXCLUDE_NODES-$(sinfo -N -o "%N %G" 2>/dev/null | \
    grep -iE "gtxtitanx|gtx1080ti|titanxp|titanxpascal|p6000|p100" | \
    awk '{print $1}' | sort -u | paste -sd,)}
EXCLUDE_ARG=""
if [ -n "${EXCLUDE_NODES}" ]; then EXCLUDE_ARG="--exclude=${EXCLUDE_NODES}"; fi

SLURM_COMMON="--partition=scavenger --account=scavenger --qos=scavenger \
--gres=gpu:1 ${EXCLUDE_ARG} --ntasks=1 --cpus-per-task=10 --mem=48G --time=12:00:00"

submit() { if ${DRY_RUN}; then echo "[dry-run] sbatch $*" >&2; echo 9999; else eval "sbatch --parsable $*"; fi; }

echo "=== rawmod_dann_loco (adv_target=${ADV_TARGET})  ->  ${OUTDIR}   epochs=${EPOCHS_ARG:-default}  dry=${DRY_RUN} ==="
for FOLD in "${FOLDS[@]}"; do
    WRAP="${CONDA_INIT} && PILEUP_PRELOAD=0 PILEUP_WORKERS=8 \
CURRICULUM=${CURRICULUM:-1} CURRICULUM_EPOCHS=${CURRICULUM_EPOCHS:-15} \
ADV_LAMBDA=${ADV_LAMBDA:-1.0} RAWMOD_DATA_GEN=${RAWMOD_DATA_GEN:-} \
${PYTHON} ${DRIVER} --fold ${FOLD} --adv-target ${ADV_TARGET} --out-dir ${OUTDIR} ${EPOCHS_ARG}"
    JID=$(submit "${SLURM_COMMON} --job-name=dann_${ADV_TARGET}_${FOLD} \
        --output=${LOGDIR}/${FOLD}_%j.out --error=${LOGDIR}/${FOLD}_%j.out \
        --wrap=\"${WRAP}\"")
    echo "  ${FOLD}: ${JID}"
done
echo "Monitor: squeue -u \$USER   Results: ${OUTDIR}/metrics/"
