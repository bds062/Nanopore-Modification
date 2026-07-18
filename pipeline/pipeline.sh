#!/bin/bash
# Nanopore modification detection pipeline: Dorado → Remora → Featurization
#
# Required env vars (set by submit_all.sh or --export):
#   DATASET       - short name for this dataset (used as output subdir)
#   POD5_DIR      - path to directory of pod5 files (may contain subdirs)
#   REF           - path to reference FASTA (may be .fa.gz or .fna.bgz)
#   GT_BED        - path to ground-truth BED (tab: ref_name, 0-based pos).
#                   Must be from an orthogonal method (emseq/bisulfite/motif).
#                   Featurization is skipped if this is missing or empty.
#   CANDIDATE_BED - (optional) BED of high-confidence +/- candidate sites.
#                   When set, only these positions are featurized, preventing
#                   ambiguous sites from being mislabeled as negative.
#   RECURSIVE     - set to "true" to pass --recursive to Dorado (for nested pod5 dirs)

#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/fs/cbcb-scratch/bds062/logs/%x_%j.out
#SBATCH --error=/fs/cbcb-scratch/bds062/logs/%x_%j.err

set -euo pipefail

mkdir -p /fs/cbcb-scratch/bds062/logs

# ── Initialize conda (required for Remora and featurization steps) ─────────────
CONDA_BASE=/nfshomes/bds062/miniconda3
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── Fixed paths ────────────────────────────────────────────────────────────────
SHARED=/fs/cbcb-lab/storm/shared/rawhash2
DORADO_BIN=${SHARED}/basecallers/dorado-0.9.2-linux-x64/bin/dorado
MODEL=${SHARED}/basecallers/dorado-0.9.2-linux-x64/bin/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
RAWHASH2=/fs/nexus-scratch/bds062/rawhash2-env/rawhash2-storm
SCRIPTS=${RAWHASH2}/test/benchmark/scripts
LEVEL_TABLE=${RAWHASH2}/extern/local_kmer_models/uncalled_r1041_model_only_means.txt
FEATURIZE=/fs/nexus-scratch/bds062/Nanopore-Modification/deepmod/featurization.py
FEATURIZE_ENV=/fs/nexus-scratch/bds062/envs/mod

# ── Validate required env vars ─────────────────────────────────────────────────
: "${DATASET:?DATASET env var is required}"
: "${POD5_DIR:?POD5_DIR env var is required}"
: "${REF:?REF env var is required}"

OUTDIR=/fs/cbcb-scratch/bds062/results/benchmark_results/${DATASET}
mkdir -p "${OUTDIR}"

echo "================================================================"
echo " Dataset  : ${DATASET}"
echo " Pod5 dir : ${POD5_DIR}"
echo " Reference: ${REF}"
echo " GT BED   : ${GT_BED:-<none>}"
echo " Recursive: ${RECURSIVE:-false}"
echo " Output   : ${OUTDIR}"
echo " Host     : $(hostname)"
echo " Date     : $(date)"
echo "================================================================"

# ── Step 1: Dorado basecalling ─────────────────────────────────────────────────
echo ""
echo "=== [1/3] Dorado basecalling ==="
if [[ -s "${OUTDIR}/reads.bam" ]]; then
    echo "  [skip] reads.bam already exists ($(du -h "${OUTDIR}/reads.bam" | cut -f1))"
else
    [[ -f "${OUTDIR}/reads.bam" ]] && rm -f "${OUTDIR}/reads.bam"
    module load samtools 2>/dev/null || true

    RECURSIVE_ARG=""
    [[ "${RECURSIVE:-false}" == "true" ]] && RECURSIVE_ARG="--recursive"

    bash "${SCRIPTS}/3_run_dorado.sh" \
        -b "${DORADO_BIN}" \
        -m "${MODEL}" \
        -i "${POD5_DIR}" \
        -o "${OUTDIR}" \
        -t 16 \
        -r "${REF}" \
        --enable-read-splitting \
        ${RECURSIVE_ARG}
fi

# ── Step 2: Remora refinement (reference-anchored, produces peaks_refined.tsv) ─
echo ""
echo "=== [2/3] Remora refinement ==="
if [[ -s "${OUTDIR}/peaks_refined.tsv" ]]; then
    echo "  [skip] peaks_refined.tsv already exists ($(du -h "${OUTDIR}/peaks_refined.tsv" | cut -f1))"
else
    bash "${SCRIPTS}/7_refine_moves_remora.sh" \
        -b "${OUTDIR}/reads.bam" \
        -p "${POD5_DIR}" \
        -l "${LEVEL_TABLE}" \
        -r \
        -o "${OUTDIR}"
fi

# ── Step 3: Featurization ──────────────────────────────────────────────────────
echo ""
echo "=== [3/3] Featurization ==="
if [[ -s "${OUTDIR}/features.h5" ]]; then
    echo "  [skip] features.h5 already exists ($(du -h "${OUTDIR}/features.h5" | cut -f1))"
else
    # GT is required for supervised training. Skip featurization if not yet available
    # so Dorado/Remora results are preserved and featurization can be rerun later.
    if [[ -z "${GT_BED:-}" || ! -f "${GT_BED}" ]]; then
        echo "  [skip] GT_BED not set or file missing — skipping featurization."
        echo "         Set GT_BED and rerun this job to produce features.h5."
        exit 0
    fi

    conda activate "${FEATURIZE_ENV}"

    CANDIDATE_ARG=""
    if [[ -n "${CANDIDATE_BED:-}" && -f "${CANDIDATE_BED}" ]]; then
        CANDIDATE_ARG="--candidate-bed ${CANDIDATE_BED}"
    fi

    # MIN_READS: 25 for bacteria/plants (20x+ coverage), 5 for mouse/human (5-7x)
    : "${MIN_READS:=25}"
    # SAMPLE_N_SITES: leave unset to use every eligible site (fine for small
    # bacterial genomes); set explicitly for large genomes with huge candidate
    # pools (arabidopsis/osativa/human) to stay within the storage budget.
    # MAX_IMAGES_PER_BASE stays at 1 — breadth of distinct sites is worth more
    # than extra images stacked on the same site.
    : "${MAX_IMAGES_PER_BASE:=1}"

    SAMPLE_ARG=""
    [[ -n "${SAMPLE_N_SITES:-}" ]] && SAMPLE_ARG="--sample-n-sites ${SAMPLE_N_SITES}"

    python "${FEATURIZE}" \
        --pod5               "${POD5_DIR}" \
        --bam                "${OUTDIR}/reads_refined.bam" \
        --peaks              "${OUTDIR}/peaks_refined.tsv" \
        --output             "${OUTDIR}/features.h5" \
        --level-table        "${LEVEL_TABLE}" \
        --gt                 "${GT_BED}" \
        ${CANDIDATE_ARG} \
        --half-window        10 \
        --L                  10 \
        --max-reads          30 \
        --min-reads          "${MIN_READS}" \
        --min-mapq           60 \
        --target-bases       AC \
        ${SAMPLE_ARG} \
        --uniform-sampling \
        --max-images-per-base "${MAX_IMAGES_PER_BASE}"
fi

echo ""
echo "================================================================"
echo " Pipeline complete: ${DATASET}"
echo " $(date)"
echo " Output: ${OUTDIR}"
ls -lh "${OUTDIR}/" 2>/dev/null
echo "================================================================"
