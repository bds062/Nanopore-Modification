#!/bin/bash
# Dataset D5: O. sativa Nipponbare (R10.4.1, 5 kHz) with EM-seq ground truth.
# Row: rice_cpg.
# Download size: ~60 GB.
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d5_osativa

# rice_w: 25 of the run files. Benchmark reads: primary alignments starting in NC_089035.1:0-13,000,000 with mean qscore >= 10, at most 15,000.
mkdir -p d5_osativa/pod5_files
while read -r url; do
  [ -s "d5_osativa/pod5_files/${url##*/}" ] || aws s3 cp --no-sign-request "$url" d5_osativa/pod5_files/
done < pod5_urls/rice_w.txt

# reference: Nipponbare T2T assembly, GCF_034140825.1.
[ -s d5_osativa/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/oryza_sativa_japonica.fa.gz" "d5_osativa/ref.fa.gz" && gunzip -f d5_osativa/ref.fa.gz; }
[ -s d5_osativa/ref.fa.fai ] || samtools faidx d5_osativa/ref.fa 2>/dev/null || true

# ground truth: EM-seq per-strand counts at >= 20x (row rice_cpg).
mkdir -p d5_osativa/ground_truth
[ -s d5_osativa/ground_truth/osjaponica_bisulfite_20x.tsv ] || aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/dataframes/plants/Osativa/osjaponica_bisulfite_20x.tsv" "d5_osativa/ground_truth/osjaponica_bisulfite_20x.tsv"

echo "d5_osativa: done"
