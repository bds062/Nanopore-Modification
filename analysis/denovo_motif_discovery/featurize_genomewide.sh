#!/bin/bash
# UNBIASED genome-wide featurization for de novo motif re-discovery.
#
# The existing benchmark_results/<DS>/features.h5 for the test genomes were built
# with --candidate-bed == the motif ground truth, so they contain ONLY motif
# positions (100% positive). Scoring those and "re-discovering" the motif would be
# circular: the positions were selected *because* they are motifs.
#
# Here we deliberately omit --candidate-bed. Positions are then chosen only by
# --target-bases AC + uniform sampling, i.e. independent of any motif annotation.
# --gt is still supplied so each emitted site carries a motif label for POST-HOC
# validation, but it does not influence which sites are emitted.
#
# Reuses the existing basecalled/refined BAM (no re-basecalling).
#
# Required env: DATASET, REF_GT (motif gt bed), OUTDIR
# Optional env: SAMPLE_N_SITES (default 500000), MIN_READS (default 10)
#
#SBATCH --job-name=gwfeat
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=highmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=16:00:00
#SBATCH --output=/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline3/logs/%x_%j.out

set -euo pipefail
: "${DATASET:?}" ; : "${REF_GT:?}" ; : "${OUTDIR:?}"
SAMPLE_N_SITES=${SAMPLE_N_SITES:-500000}
MIN_READS=${MIN_READS:-10}

PY=/fs/nexus-scratch/bds062/envs/mod/bin/python
REPO=${RAWMOD_REPO:-/fs/nexus-scratch/bds062/Nanopore-Modification}
FEATURIZE=${REPO}/deepmod/featurization.py
RECHUNK=${REPO}/pipeline/rechunk_features.py
LEVEL_TABLE=/fs/nexus-scratch/bds062/rawhash2-env/rawhash2-storm/extern/local_kmer_models/uncalled_r1041_model_only_means.txt
SRCDIR=/fs/cbcb-scratch/bds062/results/benchmark_results/${DATASET}

case "${DATASET}" in
  arabidopsis) POD5_DIR=/fs/cbcb-scratch/bds062/data/benchmark/arabidopsis/pod5 ;;
  *)           POD5_DIR=/fs/cbcb-scratch/bds062/data/benchmark/bacteria/${DATASET}/pod5 ;;
esac

mkdir -p "${OUTDIR}"
echo "=== genome-wide UNBIASED featurization: ${DATASET} — $(date) ==="
echo "    sites=${SAMPLE_N_SITES}  min_reads=${MIN_READS}  (NO --candidate-bed)"

"${PY}" "${FEATURIZE}" \
    --pod5               "${POD5_DIR}" \
    --bam                "${SRCDIR}/reads_refined.bam" \
    --peaks              "${SRCDIR}/peaks_refined.tsv" \
    --output             "${OUTDIR}/features.h5" \
    --level-table        "${LEVEL_TABLE}" \
    --gt                 "${REF_GT}" \
    --half-window        10 \
    --L                  10 \
    --max-reads          30 \
    --min-reads          "${MIN_READS}" \
    --min-mapq           60 \
    --target-bases       AC \
    --sample-n-sites     "${SAMPLE_N_SITES}" \
    --uniform-sampling \
    --max-images-per-base 1

echo "=== rechunk for streaming — $(date) ==="
"${PY}" "${RECHUNK}" "${OUTDIR}/features.h5"

echo "=== done ${DATASET} — $(date) ==="
"${PY}" - "${OUTDIR}/features.h5" <<'PYEOF'
import h5py, sys, numpy as np
with h5py.File(sys.argv[1], 'r') as h:
    lab = h['labels'][:]; n = len(lab); pos = int((lab > 0).sum())
    print(f"features.h5: n={n:,}  motif-labelled pos={pos:,} ({100*pos/n:.2f}%)  "
          f"chunks={h['tensors'].chunks[0]}")
    print("  (low %pos is EXPECTED and required: positions were chosen genome-wide, "
          "not from the motif set)")
PYEOF
