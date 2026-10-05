#!/bin/bash
# Dataset D4: A. thaliana Col-0 (R10.4.1, 5 kHz) with EM-seq ground truth.
# Rows: arabidopsis_cpg, arabidopsis_noncpg.
# Download size: ~31 GB.
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d4_athaliana

# arabidopsis_chr1: Benchmark reads: primary alignments on chromosome 1 (NC_003070.9) with mean qscore >= 10, at most 100,000.
mkdir -p d4_athaliana/pod5_files
while read -r url; do
  [ -s "d4_athaliana/pod5_files/${url##*/}" ] || aws s3 cp --no-sign-request "$url" d4_athaliana/pod5_files/
done < pod5_urls/arabidopsis_chr1.txt

# reference: TAIR10, GCF_000001735.4.
[ -s d4_athaliana/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/arabidopsis_thaliana.fa.gz" "d4_athaliana/ref.fa.gz" && gunzip -f d4_athaliana/ref.fa.gz; }
[ -s d4_athaliana/ref.fa.fai ] || samtools faidx d4_athaliana/ref.fa 2>/dev/null || true

# ground truth: EM-seq per-strand counts at >= 20x (rows arabidopsis_cpg and arabidopsis_noncpg; bench.bisulfite_rows).
mkdir -p d4_athaliana/ground_truth
[ -s d4_athaliana/ground_truth/Arabidopsis_bisulfite_20x.tsv ] || aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/dataframes/plants/Arabidopsis/Arabidopsis_bisulfite_20x.tsv" "d4_athaliana/ground_truth/Arabidopsis_bisulfite_20x.tsv"

echo "d4_athaliana: done"
