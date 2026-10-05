#!/bin/bash
# Dataset D9: Bacillus phage SPO1: native DNA (every thymine is 5hmU) and PCR-amplified control (R10.4.1).
# Row: spo1_5hmU.
# Download size: reference only (0.1 MB).
# Source: reference NCBI FJ230960.1; nanopore libraries from this study (not yet publicly available).
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d9_spo1

# spo1_bc07 (pod5): Native SPO1 phage DNA (every thymine is 5hmU), MinION R10.4.1. Not yet publicly available.
echo "spo1_bc07: the pod5 files are not publicly available; place them in d9_spo1/pod5_files/" >&2

# spo1_bc01 (pod5): PCR-amplified SPO1 DNA (unmodified control), same run as spo1_bc07. Not yet publicly available.
echo "spo1_bc01: the pod5 files are not publicly available; place them in d9_spo1/pod5_files/" >&2

# reference: Bacillus phage SPO1, complete genome (132,562 bp).
[ -s d9_spo1/ref.fa ] || wget -q -O "d9_spo1/ref.fa" "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=FJ230960.1&rettype=fasta&retmode=text"
[ -s d9_spo1/ref.fa.fai ] || samtools faidx d9_spo1/ref.fa 2>/dev/null || true

echo "d9_spo1: done"
