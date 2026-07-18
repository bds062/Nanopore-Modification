#!/bin/bash
# Dorado modification basecalling for a single dataset -> modkit pileup.
#
# Produces per-site modification fractions so build_screened_candidates.py can
# define "confident unmodified" negatives (a non-motif A/C position that ALL
# active Dorado models call unmodified) rather than assuming every non-motif
# base is unmethylated. Runs the 6mA (A) + 5mC_5hmC (C) specialist models over
# v5.0.0 sup basecalls, aligned to the reference.
#
# Required env: DATASET, POD5_DIR, REF, OUTDIR
# Optional env: MODMODELS (space-sep --modified-bases shorthands, default
#               "6mA 5mC_5hmC")
#
#SBATCH --job-name=dorado_mod
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/cbcb-scratch/bds062/logs/%x_%j.out

set -euo pipefail

: "${DATASET:?}" ; : "${POD5_DIR:?}" ; : "${REF:?}" ; : "${OUTDIR:?}"

DORADO=/fs/nexus-scratch/bds062/programs/dorado-1.3.0-linux-x64/bin/dorado
MODELS_DIR=/fs/cbcb-scratch/bds062/dorado_models
SIMPLEX=${MODELS_DIR}/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
# Pre-downloaded exact model dirs (A-context 6mA + C-context 5mC/5hmC all-context,
# NOT the CpG-only 5mCG variant — E. coli 5mC is at CCWGG/CpG, much of it non-CpG).
MOD_6MA=${MODELS_DIR}/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v1
MOD_5MC=${MODELS_DIR}/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v2.0.1
# --modified-bases-models takes ONE comma-separated arg (paths ok).
MODMODELS=${MODMODELS:-"${MOD_6MA},${MOD_5MC}"}
SAMTOOLS=/fs/nexus-scratch/bds062/envs/campolina/bin/samtools
MODKIT=/fs/nexus-scratch/bds062/envs/mod/bin/modkit

mkdir -p "${OUTDIR}"
MODBAM_UNS="${OUTDIR}/${DATASET}_mod.unsorted.bam"
MODBAM="${OUTDIR}/${DATASET}_mod.bam"
PILEUP="${OUTDIR}/${DATASET}_modkit_pileup.bed"

echo "=== [1/3] dorado basecaller (mods: ${MODMODELS}) — $(date) ==="
"${DORADO}" basecaller "${SIMPLEX}" "${POD5_DIR}" \
    --modified-bases-models "${MODMODELS}" \
    --reference "${REF}" \
    --device cuda:0 \
    > "${MODBAM_UNS}"

echo "=== [2/3] sort + index — $(date) ==="
"${SAMTOOLS}" sort -@ 8 -o "${MODBAM}" "${MODBAM_UNS}"
"${SAMTOOLS}" index "${MODBAM}"
rm -f "${MODBAM_UNS}"

echo "=== [3/3] modkit pileup — $(date) ==="
"${MODKIT}" pileup "${MODBAM}" "${PILEUP}" \
    --ref "${REF}" \
    --threads 8

echo "=== done ${DATASET} — $(date) ==="
wc -l "${PILEUP}"
ls -lh "${OUTDIR}/"
