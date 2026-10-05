#!/bin/bash
# Dataset D3: H. pylori 26695, native DNA and whole-genome-amplified control (R10.4.1, 5 kHz).
# Rows: hp26695_4mC, hp26695_6mA.
# Download size: ~4.0 GB.
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d3_hpylori_26695

# hp26695: Native DNA. Benchmark reads: read_ids/hp26695.txt.gz (the reads of the first 15,000 primary alignments).
mkdir -p d3_hpylori_26695/pod5_files
[ -s d3_hpylori_26695/pod5_files/HP26695_WT_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/HP26695_WT_5kHz/HP26695_WT_5kHz.pod5 d3_hpylori_26695/pod5_files/

# hp26695_wga: Whole-genome-amplified (unmodified) control over the region covered by hp26695.
[ -s d3_hpylori_26695/pod5_files/HP26695_WGA_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/HP26695_WGA_5kHz/HP26695_WGA_5kHz.pod5 d3_hpylori_26695/pod5_files/

# reference: NC_018939.1.
[ -s d3_hpylori_26695/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/hpylori_26695.fa.gz" "d3_hpylori_26695/ref.fa.gz" && gunzip -f d3_hpylori_26695/ref.fa.gz; }
[ -s d3_hpylori_26695/ref.fa.fai ] || samtools faidx d3_hpylori_26695/ref.fa 2>/dev/null || true

echo "d3_hpylori_26695: done"
