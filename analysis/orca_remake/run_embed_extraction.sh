#!/bin/bash
# orca_remake — extract RawMod embeddings for ONT (5mC/5hmC/6mA/control) and
# SPO1/UMCES (bc01,bc02-05,bc06,bc07) with a SINGLE checkpoint, so every
# embedding lives in the same space and can be pooled for the annotation/
# type-clustering figure.
#
# Uses deepmod_ont+umces/results5/best_model.pt -- trained EXCLUSIVELY on
# these 7 files (barcode_unmod, barcode06, barcode07, ONT 5mC/5hmC/6mA/
# control; see train5.sh's --input list). Unlike pipeline2's mixed/base
# checkpoint (which also trains on HP26695_WGA/Ecoli_DM/Ecoli_DM_MSssI/
# Ecoli_WT/arabidopsis), this model has never touched any dataset outside
# ONT+SPO1 -- per explicit user direction not to let bacterial/plant genome
# data leak into this study at all.
#
# This is still an in-sample embedding space for ONT/UMCES (the model WAS
# trained on exactly these files) -- documented as a caveat in report.md,
# not hidden.
#
# Usage: bash run_embed_extraction.sh

set -a; source "${RAWMOD_PATHS_FILE:-$HOME/.config/rawmod/paths.env}" 2>/dev/null || true; set +a   # site paths; see paths.env.example
set -euo pipefail

REPO=${RAWMOD_REPO}
SCORE=${REPO}/experiments/pipeline3/score_genome.py
CKPT=${RAWMOD_RESULTS}/deepmod_ont+umces/results5/best_model.pt
OUT=${RAWMOD_RESULTS}/orca_remake/data/embeddings
LOGDIR=${RAWMOD_RESULTS}/orca_remake/logs
PYTHON_ENV="source ${CONDA_SH} && conda activate ${RAWMOD_ENV}"

mkdir -p "${OUT}" "${LOGDIR}"

RESULTS9=${RAWMOD_LEGACY_RESULTS}/deep_modification/results9
UMCES_ONTUM=${RAWMOD_RESULTS}/deepmod_ont+umces/features

# Exactly the 7 files deepmod_ont+umces/results5/best_model.pt was itself
# trained on (see its checkpoint 'args'/'input' list) -- barcode_unmod.h5 is
# bc01-05 (PCR amplicon) merged by scripts/merge_unmod.py, bc06/07 are native
# WGS. No other file is fed to this checkpoint.
declare -A JOBS
JOBS[ONT_5mC]="${RESULTS9}/5mC.h5"
JOBS[ONT_5hmC]="${RESULTS9}/5hmC.h5"
JOBS[ONT_6mA]="${RESULTS9}/6mA.h5"
JOBS[ONT_control]="${RESULTS9}/control.h5"
JOBS[SPO1_bc06]="${UMCES_ONTUM}/barcode06.h5"
JOBS[SPO1_bc07]="${UMCES_ONTUM}/barcode07.h5"
JOBS[SPO1_amplicon]="${UMCES_ONTUM}/barcode_unmod.h5"

for name in "${!JOBS[@]}"; do
    h5="${JOBS[$name]}"
    sbatch --job-name="embed_${name}" \
        --partition=scavenger --account=scavenger --qos=scavenger \
        --gres=gpu:rtxa6000:1 --ntasks=1 --cpus-per-task=4 --mem=32G \
        --time=1:00:00 \
        --output="${LOGDIR}/embed_${name}_%j.out" \
        --error="${LOGDIR}/embed_${name}_%j.out" \
        --wrap="${PYTHON_ENV} && python ${SCORE} --h5 ${h5} --dataset ${name} --checkpoint ${CKPT} --out-dir ${OUT} --save-embeddings"
done
