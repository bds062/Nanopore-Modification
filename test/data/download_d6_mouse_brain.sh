#!/bin/bash
# Dataset D6: Mouse brain (R10.4.1, 5 kHz) with EM-seq ground truth.
# Row: mouse_cpg.
# Download size: ~938 GB (the benchmark reads are spread over all 452 files).
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d6_mouse_brain

# mouse: Benchmark reads: read_ids/mouse.txt.gz (reads overlapping candidate CpG sites on chromosome 1).
mkdir -p d6_mouse_brain/pod5_files
while read -r url; do
  [ -s "d6_mouse_brain/pod5_files/${url##*/}" ] || aws s3 cp --no-sign-request "$url" d6_mouse_brain/pod5_files/
done < pod5_urls/mouse.txt

# reference: mm39 (UCSC chromosome names).
[ -s d6_mouse_brain/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/mouse_mm39.fa.gz" "d6_mouse_brain/ref.fa.gz" && gunzip -f d6_mouse_brain/ref.fa.gz; }
[ -s d6_mouse_brain/ref.fa.fai ] || samtools faidx d6_mouse_brain/ref.fa 2>/dev/null || true

# ground truth: EM-seq per-strand counts at >= 20x, chromosome 1 (row mouse_cpg).
mkdir -p d6_mouse_brain/ground_truth
[ -s d6_mouse_brain/ground_truth/mouse_brain_bisulfite_chr1_20x.tsv ] || aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/dataframes/mammals/mouse/mouse_brain/mouse_brain_bisulfite_chr1_20x.tsv" "d6_mouse_brain/ground_truth/mouse_brain_bisulfite_chr1_20x.tsv"

echo "d6_mouse_brain: done"
