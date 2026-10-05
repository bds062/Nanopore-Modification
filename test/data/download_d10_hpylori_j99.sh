#!/bin/bash
# Dataset D10: H. pylori J99, native DNA (R10.4.1, 5 kHz), with PacBio HiFi kinetics for the ground truth.
# Rows: hpj99_4mC_pb, hpj99_6mA_pb.
# Download size: ~2.1 GB (+0.7 GB of PacBio reads with WITH_PACBIO=1).
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Source: PacBio 2021-11 Microbial 96-plex HiFi with kinetics (https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/)
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d10_hpylori_j99

# hpj99: Benchmark reads: read_ids/hpj99.txt.gz (the reads of the first 15,000 primary alignments).
mkdir -p d10_hpylori_j99/pod5_files
[ -s d10_hpylori_j99/pod5_files/HPJ99_WT_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/HPJ99_WT_5kHz/HPJ99_WT_5kHz.pod5 d10_hpylori_j99/pod5_files/

# reference: H. pylori J99 (ATCC 700824).
[ -s d10_hpylori_j99/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/hpylori_J99_ATCC700824.fa.gz" "d10_hpylori_j99/ref.fa.gz" && gunzip -f d10_hpylori_j99/ref.fa.gz; }
[ -s d10_hpylori_j99/ref.fa.fai ] || samtools faidx d10_hpylori_j99/ref.fa 2>/dev/null || true

# PacBio HiFi reads with kinetics (ground truth of the _pb rows). The rows are shipped in rows/; download these
# only to rebuild them: bash pacbio/run_ipdsummary.sh d10_hpylori_j99 && python -m bench.pacbio_rows (from test/benchmark).
if [ "${WITH_PACBIO:-0}" = 1 ]; then
  mkdir -p d10_hpylori_j99/pacbio/reads
  [ -s d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2009.bam ] || wget -q -O d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2009.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2009.bam
  [ -s d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2033.bam ] || wget -q -O d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2033.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2033.bam
  [ -s d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2057.bam ] || wget -q -O d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2057.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2057.bam
  [ -s d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2081.bam ] || wget -q -O d10_hpylori_j99/pacbio/reads/m64004_210929_143746.bc2081.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2081.bam
fi

echo "d10_hpylori_j99: done"
