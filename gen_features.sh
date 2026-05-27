#!/bin/bash

#SBATCH --time=4:00:00

#conda activate mod

MAX_PILEUP=$1
OUT_DIR=$2

mkdir -p "$OUT_DIR"

srun python ../../Nanopore-Modification/featurize/per_base_deviation.py \
    --pod5 ../../data/ont-os/subset_control/control_rep1.pod5 \
    --bam ../event_clustering_control/basecalled/reads_refined.bam \
    --peaks ../event_clustering_control/basecalled/peaks_refined.tsv \
    --level-table ../uncalled_r1041_model_only_means.txt \
    --output "$OUT_DIR/control.tsv" \
    --normalize --gt \
    --max-pileup "$MAX_PILEUP" > "$OUT_DIR/control.out" 2>&1

srun python ../../Nanopore-Modification/featurize/per_base_deviation.py \
    --pod5 ../../data/ont-os/subset_5mC/5mC_rep1.pod5 \
    --bam ../event_clustering_5mC/basecalled/reads_refined.bam \
    --peaks ../event_clustering_5mC/basecalled/peaks_refined.tsv \
    --level-table ../uncalled_r1041_model_only_means.txt \
    --output "$OUT_DIR/5mC.tsv" \
    --normalize --gt ../../data/ont-os/references/all_5mers_5mC_sites.bed \
    --max-pileup "$MAX_PILEUP" > "$OUT_DIR/5mC.out" 2>&1

srun python ../../Nanopore-Modification/featurize/per_base_deviation.py \
    --pod5 ../../data/ont-os/subset_5hmC/5hmC_rep1.pod5 \
    --bam ../event_clustering_5hmC/basecalled/reads_refined.bam \
    --peaks ../event_clustering_5hmC/basecalled/peaks_refined.tsv \
    --level-table ../uncalled_r1041_model_only_means.txt \
    --output "$OUT_DIR/5hmC.tsv" \
    --normalize --gt ../../data/ont-os/references/all_5mers_5hmC_sites.bed \
    --max-pileup "$MAX_PILEUP" > "$OUT_DIR/5hmC.out" 2>&1

srun python ../../Nanopore-Modification/featurize/per_base_deviation.py \
    --pod5 ../../data/ont-os/subset_6mA/6mA_rep1.pod5 \
    --bam ../event_clustering_6mA/basecalled/reads_refined.bam \
    --peaks ../event_clustering_6mA/basecalled/peaks_refined.tsv \
    --level-table ../uncalled_r1041_model_only_means.txt \
    --output "$OUT_DIR/6mA.tsv" \
    --normalize --gt ../../data/ont-os/references/all_5mers_6mA_sites.bed \
    --max-pileup "$MAX_PILEUP" > "$OUT_DIR/6mA.out" 2>&1