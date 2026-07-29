#!/bin/bash
# Launch the best-model (mixed Deep SAD) embedding-clustering / ORCA-analog job.
set -euo pipefail
REPO=/fs/nexus-scratch/bds062/Nanopore-Modification
SCRIPT=${REPO}/analysis/orca_remake/sad_embedding_cluster.py
LOGDIR=/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results4/embedding_clustering/logs
PYENV="source /nfshomes/bds062/miniconda3/etc/profile.d/conda.sh && conda activate /fs/nexus-scratch/bds062/envs/mod"
mkdir -p "${LOGDIR}"
sbatch --job-name=embed_cluster \
    --partition=cbcb --account=cbcb --qos=high \
    --gres=gpu:rtxa5000:1 --ntasks=1 --cpus-per-task=6 --mem=48G --time=2:00:00 \
    --output="${LOGDIR}/embed_cluster_%j.out" \
    --error="${LOGDIR}/embed_cluster_%j.out" \
    --wrap="${PYENV} && cd ${REPO} && PILEUP_PRELOAD=0 WANDB_DISABLED=1 python ${SCRIPT}"
