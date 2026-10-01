#!/bin/bash
# Merge refined shards into reads_refined.bam (+ index) and peaks_refined.tsv.
source "${GM_DIR:?}/steps/common.sh"   # sbatch runs a spooled copy, so not $0
set +u; source "$CONDA_SH"; conda activate "$REFINE_ENV"; set -u
cat "$PRE"/refine/peaks_*.tsv > "$PRE/peaks_refined.tsv"
samtools cat -o "$PRE/reads_refined.unsorted.bam" "$PRE"/refine/reads_refined_*.bam
samtools sort -@ 8 -o "$PRE/reads_refined.bam" "$PRE/reads_refined.unsorted.bam"
samtools index "$PRE/reads_refined.bam"
rm -f "$PRE/reads_refined.unsorted.bam"
echo "peaks_refined.tsv: $(wc -l < "$PRE/peaks_refined.tsv") reads"
