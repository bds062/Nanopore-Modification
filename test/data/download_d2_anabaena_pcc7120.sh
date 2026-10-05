#!/bin/bash
# Dataset D2: Anabaena (Nostoc) sp. PCC 7120, native DNA (R10.4.1, 5 kHz).
# Row: anabaena_6mA.
# Download size: ~8.4 GB.
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d2_anabaena_pcc7120

# anabaena: Benchmark reads: read_ids/anabaena.txt.gz (the reads of the first 15,000 primary alignments).
mkdir -p d2_anabaena_pcc7120/pod5_files
[ -s d2_anabaena_pcc7120/pod5_files/Anabaena_WT_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/Anabaena_WT_5kHz/Anabaena_WT_5kHz.pod5 d2_anabaena_pcc7120/pod5_files/

# reference: Nostoc (Anabaena) sp. PCC 7120, GCF_000009705.1 (chromosome and six plasmids).
[ -s d2_anabaena_pcc7120/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/anabaena_sp_PCC7120_ATCC27893.fa.gz" "d2_anabaena_pcc7120/ref.fa.gz" && gunzip -f d2_anabaena_pcc7120/ref.fa.gz; }
[ -s d2_anabaena_pcc7120/ref.fa.fai ] || samtools faidx d2_anabaena_pcc7120/ref.fa 2>/dev/null || true

echo "d2_anabaena_pcc7120: done"
