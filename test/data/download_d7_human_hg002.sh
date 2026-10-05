#!/bin/bash
# Dataset D7: Human HG002 (GIAB 2023.05, R10.4.1, 5 kHz) with whole-genome bisulfite ground truth.
# Row: hg002_cpg.
# Download size: ~286 GB.
# Source: ONT Open Data giab_2023.05 (https://epi2me.nanoporetech.com/giab-2023.05/)
# Source: ONT Open Data gm24385_mod_2021.09 (https://epi2me.nanoporetech.com/gm24385-5mc/)
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d7_human_hg002

# hg002: Flow cell PAO89685 (pod5_pass). Benchmark reads: read_ids/hg002.txt.gz (the reads of the first 15,000 primary alignments).
mkdir -p d7_human_hg002/pod5_files
while read -r url; do
  [ -s "d7_human_hg002/pod5_files/${url##*/}" ] || aws s3 cp --no-sign-request "$url" d7_human_hg002/pod5_files/
done < pod5_urls/hg002.txt

# reference: GRCh38 no-alt analysis set, the coordinates of the bisulfite ground truth.
[ -s d7_human_hg002/ref.fa ] || aws s3 cp --no-sign-request "s3://ont-open-data/gm24385_mod_2021.09/refs/GCA_000001405.15_GRCh38_no_alt_analysis_set.fa" "d7_human_hg002/ref.fa"
[ -s d7_human_hg002/ref.fa.fai ] || samtools faidx d7_human_hg002/ref.fa 2>/dev/null || true

# ground truth: Whole-genome bisulfite sequencing of GM24385 (HG002), Bismark CpG coverage (row hg002_cpg).
mkdir -p d7_human_hg002/ground_truth
[ -s d7_human_hg002/ground_truth/CpG.bismark.zero.cov.gz ] || aws s3 cp --no-sign-request "s3://ont-open-data/gm24385_mod_2021.09/bisulphite/cpg/CpG.gz.bismark.zero.cov.gz" "d7_human_hg002/ground_truth/CpG.bismark.zero.cov.gz"

echo "d7_human_hg002: done"
