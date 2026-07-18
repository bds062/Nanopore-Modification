#!/bin/bash
# Extract 5mC ground-truth BED files from enzymatic/bisulfite methylation-seq BAMs.
#
# Uses modkit pileup to get per-position methylation fraction, then applies
# the Rockfish-style confidence filter:
#   modified  : coverage >= MIN_COV and fraction >= HI_FRAC  (label=1)
#   unmodified: coverage >= MIN_COV and fraction <= LO_FRAC  (label=0)
#   discarded : everything else (ambiguous / low coverage)
#
# Outputs (in OUTDIR):
#   gt_modified.bed     — tab: ref_name, 0-based pos     (positives for --gt)
#   candidate.bed       — tab: ref_name, 0-based pos     (high-conf +/- for --candidate-bed)
#
# Usage:
#   sbatch extract_gt_emseq.sh  (with env vars set via --export)
# OR:
#   bash extract_gt_emseq.sh  (env vars set in calling shell)
#
# Required env vars:
#   DATASET   — short name (used for log and output dir)
#   EMSEQ_BAM — path to deduplicated emseq/bisulfite BAM
#   REF       — reference FASTA (.fa.gz or .fna.bgz)
#   OUTDIR    — output directory
#
# Optional env vars:
#   MIN_COV   — minimum read coverage (default: 30)
#   HI_FRAC   — fraction threshold for modified (default: 0.90)
#   LO_FRAC   — fraction threshold for unmodified (default: 0.10)
#   THREADS   — number of threads for modkit (default: 16)

#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/cbcb-scratch/bds062/logs/extract_gt_%x_%j.out
#SBATCH --error=/fs/cbcb-scratch/bds062/logs/extract_gt_%x_%j.err

set -euo pipefail
mkdir -p /fs/cbcb-scratch/bds062/logs

module load samtools/1.16 2>/dev/null || true

CONDA_BASE=/nfshomes/bds062/miniconda3
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate /fs/nexus-scratch/bds062/envs/mod

MODKIT=/fs/nexus-scratch/bds062/envs/mod/bin/modkit

: "${DATASET:?DATASET env var required}"
: "${EMSEQ_BAM:?EMSEQ_BAM env var required}"
: "${REF:?REF env var required}"
: "${OUTDIR:?OUTDIR env var required}"

MIN_COV="${MIN_COV:-30}"
HI_FRAC="${HI_FRAC:-0.90}"
LO_FRAC="${LO_FRAC:-0.10}"
THREADS="${THREADS:-16}"

mkdir -p "${OUTDIR}"

GT_MOD="${OUTDIR}/gt_modified.bed"
CANDIDATE="${OUTDIR}/candidate.bed"

SCRIPTS_DIR="$(dirname "$(realpath "$0")")"
PYTHON=/fs/nexus-scratch/bds062/envs/mod/bin/python

echo "================================================================"
echo " Dataset   : ${DATASET}"
echo " emseq BAM : ${EMSEQ_BAM}"
echo " Reference : ${REF:-<not required for Bismark>}"
echo " Output    : ${OUTDIR}"
echo " Filters   : coverage>=${MIN_COV}, mod>=${HI_FRAC}, unmod<=${LO_FRAC}"
echo " Date      : $(date)"
echo "================================================================"

# ── Auto-detect BAM format ─────────────────────────────────────────────────────
# Bismark BAMs have @PG records with ID:Bismark; ONT modbam have MM/ML tags.
IS_BISMARK=false
if samtools view -H "${EMSEQ_BAM}" 2>/dev/null | grep -q "ID:Bismark"; then
    IS_BISMARK=true
fi
echo "BAM format: $(${IS_BISMARK} && echo "Bismark bisulfite" || echo "ONT modbam (MM/ML)")"

if ${IS_BISMARK}; then
    # ── Bismark BAM: parse XM tags with pysam ─────────────────────────────────
    echo ""
    echo "Running Bismark XM-tag extractor (pysam)..."
    "${PYTHON}" "${SCRIPTS_DIR}/extract_gt_bismark.py" \
        --bam     "${EMSEQ_BAM}" \
        --outdir  "${OUTDIR}" \
        --context CpG \
        --min-cov "${MIN_COV}" \
        --hi-frac "${HI_FRAC}" \
        --lo-frac "${LO_FRAC}" \
        --threads "${THREADS}"
else
    # ── ONT modbam: use modkit pileup ─────────────────────────────────────────
    PILEUP="${OUTDIR}/modkit_pileup.bed"
    : "${REF:?REF is required for modkit pileup on ONT modbam}"

    if [[ ! -f "${EMSEQ_BAM}.bai" && ! -f "${EMSEQ_BAM%.bam}.bai" ]]; then
        echo "Indexing BAM..."
        samtools index -@ "${THREADS}" "${EMSEQ_BAM}"
    fi

    echo ""
    echo "Running modkit pileup..."
    if [[ -s "${PILEUP}" ]]; then
        echo "  [skip] pileup already exists ($(du -h "${PILEUP}" | cut -f1))"
    else
        "${MODKIT}" pileup \
            --ref "${REF}" \
            --threads "${THREADS}" \
            --log-filepath "${OUTDIR}/modkit.log" \
            "${EMSEQ_BAM}" \
            "${PILEUP}"
    fi

    echo "Pileup complete. Lines: $(wc -l < "${PILEUP}")"

    echo ""
    echo "Extracting GT BEDs (min_cov=${MIN_COV}, hi=${HI_FRAC}, lo=${LO_FRAC})..."
    export PILEUP GT_MOD CANDIDATE MIN_COV HI_FRAC LO_FRAC
    python3 - <<'PYEOF'
import sys, os
pileup  = os.environ["PILEUP"]
gt_mod  = os.environ["GT_MOD"]
cand    = os.environ["CANDIDATE"]
min_cov = int(os.environ["MIN_COV"])
hi_frac = float(os.environ["HI_FRAC"])
lo_frac = float(os.environ["LO_FRAC"])
n_mod = n_unmod = n_ambig = n_lowcov = 0
with open(pileup) as fh, open(gt_mod, 'w') as fmod, open(cand, 'w') as fcand:
    for line in fh:
        if line.startswith('#'): continue
        p = line.rstrip('\n').split('\t')
        if len(p) < 11: continue
        n_valid = int(p[9]); frac = float(p[10])
        if n_valid < min_cov: n_lowcov += 1; continue
        if frac >= hi_frac:
            fmod.write(f"{p[0]}\t{p[1]}\n"); fcand.write(f"{p[0]}\t{p[1]}\n"); n_mod += 1
        elif frac <= lo_frac:
            fcand.write(f"{p[0]}\t{p[1]}\n"); n_unmod += 1
        else: n_ambig += 1
print(f"  Modified  : {n_mod:,}  Unmodified: {n_unmod:,}  Ambig: {n_ambig:,}  LowCov: {n_lowcov:,}")
PYEOF
fi

echo ""
echo "================================================================"
echo " Done: ${DATASET}"
echo " GT modified : ${GT_MOD}  ($(wc -l < "${GT_MOD}") sites)"
echo " Candidate   : ${CANDIDATE}  ($(wc -l < "${CANDIDATE}") sites)"
echo " $(date)"
echo "================================================================"
