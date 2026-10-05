#!/bin/bash
# Dataset D1: E. coli K-12 MG1655: wild type, dam- dcm- knockout, and knockout + M.SssI (R10.4.1, 5 kHz).
# Rows: ecoli_dam_6mA_ko, ecoli_dcm_5mC_ko, ecoli_mssi_5mC_ko.
# Download size: ~16.5 GB.
# Source: Kulkarni et al. 2026, Nat. Commun., doi:10.1038/s41467-026-75183-6; AWS Open Data ont-basemod-benchmark-data
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d1_ecoli_k12

# ecoli_wt_w1m: Wild type. Benchmark reads: primary alignments starting in NC_000913.3:0-1,000,000 with mean qscore >= 10, at most 15,000 (read_ids/ecoli_wt_w1m.txt.gz).
mkdir -p d1_ecoli_k12/pod5_files
[ -s d1_ecoli_k12/pod5_files/Ecoli_WT_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/Ecoli_WT_5kHz/Ecoli_WT_5kHz.pod5 d1_ecoli_k12/pod5_files/

# ecoli_dm_w1m: dam- dcm- double knockout; same read rule and window as ecoli_wt_w1m.
[ -s d1_ecoli_k12/pod5_files/Ecoli_DM_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/Ecoli_DM_5kHz/Ecoli_DM_5kHz.pod5 d1_ecoli_k12/pod5_files/

# ecoli_mssi_w1m: dam- dcm- knockout treated with M.SssI (CpG methyltransferase); same read rule and window as ecoli_wt_w1m.
[ -s d1_ecoli_k12/pod5_files/Ecoli_DM_MSssI_5kHz.pod5 ] || aws s3 cp --no-sign-request s3://ont-basemod-benchmark-data/Raw/pod5/bacteria/Ecoli_DM_MSssI_5kHz/Ecoli_DM_MSssI_5kHz.pod5 d1_ecoli_k12/pod5_files/

# reference: NC_000913.3 (E. coli K-12 MG1655).
[ -s d1_ecoli_k12/ref.fa ] || { aws s3 cp --no-sign-request "s3://ont-basemod-benchmark-data/Analysis/Reference/ecoli.fa.gz" "d1_ecoli_k12/ref.fa.gz" && gunzip -f d1_ecoli_k12/ref.fa.gz; }
[ -s d1_ecoli_k12/ref.fa.fai ] || samtools faidx d1_ecoli_k12/ref.fa 2>/dev/null || true

echo "d1_ecoli_k12: done"
