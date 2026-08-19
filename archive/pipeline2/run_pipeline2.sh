#!/bin/bash
# deepmod_full_pipeline2 — 10 fold jobs (mixed + 7 lodo_<dataset> + lomo_6mA +
# lomo_5mC) + a collect job, mirroring deepmod_full_pipeline1's run_convformer_v2.sh.
#
# 7 role=train datasets: HP26695_WGA, Ecoli_DM, Ecoli_DM_MSssI, Ecoli_WT,
# arabidopsis (bacteria/plant, from deepmod_genomes/manifest.tsv) + UMCES
# (SPO1 phage, 7 barcode files) + ONT (synthetic benchmark, 4 files).
#
# train_one_model() preloads its whole split into RAM (uncompressed,
# ~0.234MB/image); the pooled train-role set (~1.85M images across all 7
# datasets) is far bigger than any single dataset results1-6 ever trained
# on, so plan for ~300-430GB RAM per job, not the 96-160G used in pipeline1.
#
# EVERY fold needs ALL 7 datasets featurized: mixed and both lomo folds pool
# all 7, and each lodo_X fold trains on the other 6 AND evaluates on the
# held-out X (so it needs X's features too). There is therefore no per-fold
# dependency shortcut. Set FEAT_DEPS to a SLURM --dependency spec covering any
# featurization jobs that must finish first; leave it empty if all 7
# features.h5 already exist. No fold depends on another fold job (mixed does
# not gate lodo/lomo, and vice versa) — all 10 run in parallel once inputs exist.
#
# Usage (login node):
#   bash run_pipeline2.sh [--dry-run] [--epochs N]
#   FEAT_DEPS="afterok:7086055" bash run_pipeline2.sh   # wait on a featurization job

set -euo pipefail

DRY_RUN=false
EPOCHS_ARG=""
for a in "$@"; do
    case "$a" in
        --dry-run) DRY_RUN=true ;;
        --epochs) : ;;
        [0-9]*)   EPOCHS_ARG="--epochs $a" ;;
    esac
done

PIPE=/fs/cbcb-scratch/bds062/results/deepmod_full_pipeline2
OUTDIR=${OUTDIR:-${PIPE}/results1}
LOGDIR=${PIPE}/logs
# Driver/collect come from THIS repo, not the scratch copies: run_pipeline2.py
# resolves its imports via Path(__file__).parents[2], which is only the repo
# when the file it runs *is* the repo file. (Overridable for debugging.)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER=${DRIVER:-${REPO_DIR}/run_pipeline2.py}
COLLECT=${COLLECT:-${REPO_DIR}/collect2.py}
PYTHON=/fs/nexus-scratch/bds062/envs/mod/bin/python
CONDA_INIT="source /nfshomes/bds062/miniconda3/etc/profile.d/conda.sh && conda activate /fs/nexus-scratch/bds062/envs/mod"

mkdir -p "${LOGDIR}"

# Featurization dependency for ALL folds (empty = every features.h5 already
# exists). Override from the environment, e.g. FEAT_DEPS="afterok:7086055".
FEAT_DEPS=${FEAT_DEPS:-}

FOLDS=(mixed lodo_HP26695_WGA_5kHz lodo_Ecoli_DM_5kHz lodo_Ecoli_DM_MSssI_5kHz
       lodo_Ecoli_WT_5kHz lodo_arabidopsis lodo_UMCES lodo_ONT
       lomo_5mC lomo_5hmC lomo_6mA lomo_5hmU)

# cbcb on rtxa5000 (~501GB nodes). STREAMING mode (PILEUP_PRELOAD=0): instead of
# preloading the ~390GB pooled set into RAM (which forced mem=460G + scarce
# qos=highmem), each DataLoader worker streams images straight from the
# re-chunked HDF5 (1 image/chunk, see scripts/rechunk_features.py) so a random
# read decompresses one image, not 64. RAM drops to a few GB, so mem=48G fits
# qos=high (128G cap) and schedules on plentiful nodes. 10 cpus feed 8 workers.
SLURM_COMMON="--partition=cbcb --account=cbcb --qos=high \
--gres=gpu:rtxa5000:1 --ntasks=1 --cpus-per-task=10 --mem=48G --time=24:00:00"

submit() {
    if ${DRY_RUN}; then echo "[dry-run] sbatch $*" >&2; echo "9999"; else eval "sbatch --parsable $*"; fi
}

echo "================================================================"
echo " deepmod_full_pipeline2  —  10 fold jobs + collect   Output: ${OUTDIR}"
echo " Epochs : ${EPOCHS_ARG:-default(50)}   Dry-run: ${DRY_RUN}"
echo "================================================================"

DEP_ARG=""
[[ -n "${FEAT_DEPS}" ]] && DEP_ARG="--dependency=${FEAT_DEPS}"

JIDS=()
for FOLD in "${FOLDS[@]}"; do
    WRAP="${CONDA_INIT} && MANIFEST=${MANIFEST:-} PILEUP_PRELOAD=0 PILEUP_WORKERS=8 \
DANN_LAMBDA=${DANN_LAMBDA:-0} SUPCON_DIM=${SUPCON_DIM:-0} \
SUPCON_WEIGHT=${SUPCON_WEIGHT:-0.1} SUPCON_TEMP=${SUPCON_TEMP:-0.07} \
${PYTHON} ${DRIVER} --fold ${FOLD} --out-dir ${OUTDIR} ${EPOCHS_ARG}"
    JID=$(submit "${SLURM_COMMON} ${DEP_ARG} \
        --job-name=fp2_${FOLD} \
        --output=${LOGDIR}/${FOLD}_%j.out --error=${LOGDIR}/${FOLD}_%j.out \
        --wrap=\"${WRAP}\"")
    echo "  ${FOLD} job: ${JID}  (dependency=${FEAT_DEPS:-none})"
    JIDS+=("${JID}")
done

DEP=$(printf ",afterok:%s" "${JIDS[@]}"); DEP=${DEP:1}
COLLECT_WRAP="${CONDA_INIT} && ${PYTHON} ${COLLECT} --out-dir ${OUTDIR}"
CJID=$(submit "--partition=cbcb --account=cbcb --qos=high --ntasks=1 --cpus-per-task=2 \
    --mem=16G --time=0:30:00 --dependency=${DEP} \
    --job-name=fp2_collect \
    --output=${LOGDIR}/collect_%j.out --error=${LOGDIR}/collect_%j.out \
    --wrap=\"${COLLECT_WRAP}\"")

echo ""
echo "  collect job: ${CJID}  (dependency=${DEP})"
echo " Monitor : squeue -u \$USER    Results: ${OUTDIR}/{metrics,figures,models}/"
echo "================================================================"
