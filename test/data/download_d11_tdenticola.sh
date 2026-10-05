#!/bin/bash
# Dataset D11: T. denticola ATCC 35405, native DNA (R10.4.1, 5 kHz), with PacBio HiFi kinetics for the ground truth.
# Rows: tden_4mC_pb, tden_6mA_pb.
# Download size: ~3.5 GB (+1.8 GB of PacBio reads with WITH_PACBIO=1).
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Source: PacBio 2021-11 Microbial 96-plex HiFi with kinetics (https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/)
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d11_tdenticola

# tdenticola: Benchmark reads: read_ids/tdenticola.txt.gz (the reads of the first 15,000 primary alignments).
mkdir -p d11_tdenticola/pod5_files
[ -s d11_tdenticola/pod5_files/Tdenticola_WT_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/Tdenticola_WT_5kHz/Tdenticola_WT_5kHz.pod5 d11_tdenticola/pod5_files/

# reference: T. denticola ATCC 35405.
[ -s d11_tdenticola/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/treponema_denticola_ATCC35405.fa.gz" "d11_tdenticola/ref.fa.gz" && gunzip -f d11_tdenticola/ref.fa.gz; }
[ -s d11_tdenticola/ref.fa.fai ] || samtools faidx d11_tdenticola/ref.fa 2>/dev/null || true

# PacBio HiFi reads with kinetics (ground truth of the _pb rows). The rows are shipped in rows/; download these
# only to rebuild them: bash pacbio/run_ipdsummary.sh d11_tdenticola && python -m bench.pacbio_rows (from test/benchmark).
if [ "${WITH_PACBIO:-0}" = 1 ]; then
  mkdir -p d11_tdenticola/pacbio/reads
  [ -s d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2023.bam ] || wget -q -O d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2023.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2023.bam
  [ -s d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2047.bam ] || wget -q -O d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2047.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2047.bam
  [ -s d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2071.bam ] || wget -q -O d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2071.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2071.bam
  [ -s d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2095.bam ] || wget -q -O d11_tdenticola/pacbio/reads/m64004_210929_143746.bc2095.bam https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/demultiplexed-reads/m64004_210929_143746.bc2095.bam
fi

echo "d11_tdenticola: done"
