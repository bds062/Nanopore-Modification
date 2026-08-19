#!/bin/bash
# Re-featurize ONE motif dataset with balanced labels:
#   1. build_screened_candidates.py -> candidate.bed (motif positives +
#      Dorado-confident-unmodified non-motif A/C negatives)
#   2. featurization.py with the EXISTING refined BAM (no re-basecall) and the
#      new candidate.bed -> v2 features.h5 (positives label 1, screened
#      negatives label 0).
#
# Required env: DATASET, REF, GT_BED, PILEUP, EXCLUDE_MOTIFS, OUTDIR
# Optional env: NEG_PER_POS(1.0) MAX_TOTAL(500000) NEG_MAX_FRAC(10.0)
#               MIN_COV(5) MIN_READS(25)
#
#SBATCH --job-name=refeat
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/cbcb-scratch/bds062/logs/%x_%j.out

set -euo pipefail

: "${DATASET:?}" ; : "${REF:?}" ; : "${GT_BED:?}" ; : "${PILEUP:?}" ; : "${OUTDIR:?}"
EXCLUDE_MOTIFS=${EXCLUDE_MOTIFS:-}
NEG_PER_POS=${NEG_PER_POS:-1.0}
MAX_TOTAL=${MAX_TOTAL:-500000}
NEG_MAX_FRAC=${NEG_MAX_FRAC:-10.0}
MIN_COV=${MIN_COV:-5}
MIN_READS=${MIN_READS:-25}

PY=/fs/nexus-scratch/bds062/envs/mod/bin/python
FEATURIZE=/fs/nexus-scratch/bds062/Nanopore-Modification/deepmod/featurization.py
BUILDCAND=/fs/cbcb-scratch/bds062/scripts/build_screened_candidates.py
LEVEL_TABLE=/fs/nexus-scratch/bds062/rawhash2-env/rawhash2-storm/extern/local_kmer_models/uncalled_r1041_model_only_means.txt
SRCDIR=/fs/cbcb-scratch/bds062/results/benchmark_results/${DATASET}
POD5_DIR=$(dirname "${SRCDIR}")   # placeholder; set below from known layout

# POD5 dir (same layout as submit_all.sh)
case "${DATASET}" in
  arabidopsis) POD5_DIR=/fs/cbcb-scratch/bds062/data/benchmark/arabidopsis/pod5 ;;
  *)           POD5_DIR=/fs/cbcb-scratch/bds062/data/benchmark/bacteria/${DATASET}/pod5 ;;
esac

mkdir -p "${OUTDIR}"
CAND="${OUTDIR}/candidate_screened.bed"

echo "=== [1/2] build screened candidates — $(date) ==="
EXCL_ARG=""
[[ -n "${EXCLUDE_MOTIFS}" ]] && EXCL_ARG="--exclude-motifs ${EXCLUDE_MOTIFS}"
# shellcheck disable=SC2086
"${PY}" "${BUILDCAND}" \
    --ref "${REF}" --gt "${GT_BED}" --pileup "${PILEUP}" --out "${CAND}" \
    --neg-per-pos "${NEG_PER_POS}" --max-total "${MAX_TOTAL}" \
    --neg-max-frac "${NEG_MAX_FRAC}" --min-cov "${MIN_COV}" ${EXCL_ARG}

echo "=== [2/2] featurize with screened candidates — $(date) ==="
"${PY}" "${FEATURIZE}" \
    --pod5               "${POD5_DIR}" \
    --bam                "${SRCDIR}/reads_refined.bam" \
    --peaks              "${SRCDIR}/peaks_refined.tsv" \
    --output             "${OUTDIR}/features.h5" \
    --level-table        "${LEVEL_TABLE}" \
    --gt                 "${GT_BED}" \
    --candidate-bed      "${CAND}" \
    --half-window        10 \
    --L                  10 \
    --max-reads          30 \
    --min-reads          "${MIN_READS}" \
    --min-mapq           60 \
    --target-bases       AC \
    --uniform-sampling \
    --max-images-per-base 1

echo "=== [3/3] rechunk to 1 image/chunk for streaming — $(date) ==="
"${PY}" /fs/cbcb-scratch/bds062/scripts/rechunk_features.py "${OUTDIR}/features.h5"

echo "=== done ${DATASET} — $(date) ==="
"${PY}" - "${OUTDIR}/features.h5" <<'PYEOF'
import h5py, sys
with h5py.File(sys.argv[1], 'r') as h:
    lab = h['labels'][:]
    n = len(lab); pos = int((lab > 0).sum())
    ch = h['tensors'].chunks
    print(f"features.h5: n={n:,}  pos={pos:,} ({100*pos/n:.1f}%)  neg={n-pos:,}  chunks={ch}")
PYEOF
