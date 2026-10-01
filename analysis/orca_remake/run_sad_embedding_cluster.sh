#!/bin/bash
# Launch the best-model (mixed Deep SAD) embedding-clustering / ORCA-analog job.
set -a; source "${RAWMOD_PATHS_FILE:-$HOME/.config/rawmod/paths.env}" 2>/dev/null || true; set +a   # site paths; see paths.env.example
set -euo pipefail
REPO=${RAWMOD_REPO}
SCRIPT=${REPO}/analysis/orca_remake/sad_embedding_cluster.py
LOGDIR=${RAWMOD_RESULTS}/rawmod_matched_loco/results4/embedding_clustering/logs
PYENV="source ${CONDA_SH} && conda activate ${RAWMOD_ENV}"
mkdir -p "${LOGDIR}"
sbatch --job-name=embed_cluster \
    ${RAWMOD_SBATCH_ARGS:-} \
    --gres=gpu:rtxa5000:1 --ntasks=1 --cpus-per-task=6 --mem=48G --time=2:00:00 \
    --output="${LOGDIR}/embed_cluster_%j.out" \
    --error="${LOGDIR}/embed_cluster_%j.out" \
    --wrap="${PYENV} && cd ${REPO} && PILEUP_PRELOAD=0 WANDB_DISABLED=1 python ${SCRIPT}"
