#!/bin/bash
set -a; source "${RAWMOD_PATHS_FILE:-$HOME/.config/rawmod/paths.env}" 2>/dev/null || true; set +a   # site paths; see paths.env.example
set -euo pipefail
REPO=${RAWMOD_REPO}
SCRIPT=${REPO}/analysis/orca_remake/loco_embedding_cluster.py
LOGDIR=${RAWMOD_RESULTS}/rawmod_matched_loco/results4/embedding_clustering_per_fold/logs
PYENV="source ${CONDA_SH} && conda activate ${RAWMOD_ENV}"
mkdir -p "${LOGDIR}"
sbatch --job-name=loco_embed_cluster \
    --partition=scavenger --account=scavenger --qos=scavenger \
    --gres=gpu:rtxa6000:1 --ntasks=1 --cpus-per-task=6 --mem=48G --time=2:00:00 \
    --output="${LOGDIR}/loco_embed_cluster_%j.out" \
    --error="${LOGDIR}/loco_embed_cluster_%j.out" \
    --wrap="${PYENV} && cd ${REPO} && PILEUP_PRELOAD=0 WANDB_DISABLED=1 python ${SCRIPT}"
