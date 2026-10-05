#!/bin/bash
# Dataset D8: ONT synthetic oligonucleotides covering all 5-mers: 5mC, 5hmC, 6mA and control, replicates 1 and 2 (R10.4.1, 5 kHz).
# Rows: syn_5mC, syn_5hmC, syn_6mA (replicate 1); syn_5mC_rep2, syn_5hmC_rep2, syn_6mA_rep2 (replicate 2).
# Download size: ~3 GB.
# Source: ONT Open Data modbase-validation_2024.10 (https://epi2me.nanoporetech.com/mod-validation-data/)
# Requires: AWS CLI (no account needed, --no-sign-request), wget, gunzip.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p d8_ont_oligos

# syn_5mC: Benchmark reads: read_ids/syn_5mC.txt.gz (a held-out 20% of the reads).
mkdir -p d8_ont_oligos/pod5_files
[ -s d8_ont_oligos/pod5_files/5mC_rep1.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/5mC_rep1.pod5 d8_ont_oligos/pod5_files/

# syn_5mC_rep2: Replicate 2, a library never used in RawMod training. Benchmark reads: 32,000 random reads (seed 0; read_ids/syn_5mC_rep2.txt.gz).
[ -s d8_ont_oligos/pod5_files/5mC_rep2.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/5mC_rep2.pod5 d8_ont_oligos/pod5_files/

# syn_5hmC: Benchmark reads: read_ids/syn_5hmC.txt.gz (a held-out 20% of the reads).
[ -s d8_ont_oligos/pod5_files/5hmC_rep1.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/5hmC_rep1.pod5 d8_ont_oligos/pod5_files/

# syn_5hmC_rep2: Replicate 2, a library never used in RawMod training. Benchmark reads: 32,000 random reads (seed 0; read_ids/syn_5hmC_rep2.txt.gz).
[ -s d8_ont_oligos/pod5_files/5hmC_rep2.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/5hmC_rep2.pod5 d8_ont_oligos/pod5_files/

# syn_6mA: Benchmark reads: read_ids/syn_6mA.txt.gz (a held-out 20% of the reads).
[ -s d8_ont_oligos/pod5_files/6mA_rep1.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/6mA_rep1.pod5 d8_ont_oligos/pod5_files/

# syn_6mA_rep2: Replicate 2, a library never used in RawMod training. Benchmark reads: 32,000 random reads (seed 0; read_ids/syn_6mA_rep2.txt.gz).
[ -s d8_ont_oligos/pod5_files/6mA_rep2.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/6mA_rep2.pod5 d8_ont_oligos/pod5_files/

# syn_control: Benchmark reads: read_ids/syn_control.txt.gz (a held-out 20% of the reads).
[ -s d8_ont_oligos/pod5_files/control_rep1.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/control_rep1.pod5 d8_ont_oligos/pod5_files/

# syn_control_rep2: Replicate 2, a library never used in RawMod training. Benchmark reads: 32,000 random reads (seed 0; read_ids/syn_control_rep2.txt.gz).
[ -s d8_ont_oligos/pod5_files/control_rep2.pod5 ] || aws s3 cp --no-sign-request s3://ont-open-data/modbase-validation_2024.10/subset/control_rep2.pod5 d8_ont_oligos/pod5_files/

# reference: 32 synthetic constructs covering all 5-mers.
[ -s d8_ont_oligos/ref.fa ] || aws s3 cp --no-sign-request "s3://ont-open-data/modbase-validation_2024.10/references/all_5mers.fa" "d8_ont_oligos/ref.fa"
[ -s d8_ont_oligos/ref.fa.fai ] || samtools faidx d8_ont_oligos/ref.fa 2>/dev/null || true

# ground truth: Designed modified positions; negatives are the same positions in the control library.
mkdir -p d8_ont_oligos/ground_truth
[ -s d8_ont_oligos/ground_truth/all_5mers_5mC_sites.bed ] || aws s3 cp --no-sign-request "s3://ont-open-data/modbase-validation_2024.10/references/all_5mers_5mC_sites.bed" "d8_ont_oligos/ground_truth/all_5mers_5mC_sites.bed"

# ground truth: Designed modified positions; negatives are the same positions in the control library.
[ -s d8_ont_oligos/ground_truth/all_5mers_5hmC_sites.bed ] || aws s3 cp --no-sign-request "s3://ont-open-data/modbase-validation_2024.10/references/all_5mers_5hmC_sites.bed" "d8_ont_oligos/ground_truth/all_5mers_5hmC_sites.bed"

# ground truth: Designed modified positions; negatives are the same positions in the control library.
[ -s d8_ont_oligos/ground_truth/all_5mers_6mA_sites.bed ] || aws s3 cp --no-sign-request "s3://ont-open-data/modbase-validation_2024.10/references/all_5mers_6mA_sites.bed" "d8_ont_oligos/ground_truth/all_5mers_6mA_sites.bed"

echo "d8_ont_oligos: done"
