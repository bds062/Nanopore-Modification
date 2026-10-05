#!/bin/bash
# ORCA exact featurization for the ONT synthetic DNA benchmark.
# pod5 -> blow5 (blue-crab) -> f5c eventalign (R10) + samtools mpileup ->
# ORCA signal + basecalling feature extraction -> merged per-site features.
# NOTE: uses cbcb samtools 1.16 for mpileup (the conda one segfaults) and the
# f5c v1.6 static binary (pod5/R10 support).
set -euo pipefail

F5C=/fs/nexus-scratch/vgandhi/f5c-v1.6/f5c_x86_64_linux
SAM=/fs/cbcb-software/RedHat-8-x86_64/local/samtools/1.16/bin/samtools
DATA=/fs/nexus-scratch/bds062/data/ont-os
REF=$DATA/references/all_5mers.fa
OUT=${OUT:-/fs/nexus-scratch/vgandhi/orca_feat}
CONDITIONS=${CONDITIONS:-"control_rep1 5mC_rep1 5hmC_rep1 6mA_rep1"}
T=${T:-6}
mkdir -p $OUT

for COND in $CONDITIONS; do
    W=$OUT/$COND
    mkdir -p $W
    echo "==================== $COND ===================="
    date

    # pod5 -> blow5 (skip if already done)
    [ -f $W/$COND.blow5 ] || blue-crab p2s $DATA/subset/$COND.pod5 -o $W/$COND.blow5

    # reads + sorted/indexed BAM
    $SAM fastq -F 0x900 $DATA/basecalls/$COND.bam > $W/$COND.fastq
    $SAM sort -@ $T -o $W/$COND.sorted.bam $DATA/basecalls/$COND.bam
    $SAM index $W/$COND.sorted.bam

    # f5c index + eventalign (R10 DNA). --min-recalib-events 25: the synthetic
    # constructs are only ~155bp, so the default (200) fails calibration on most
    # short reads; 25 recovers ~93% of reads (vs ~13% at the default).
    $F5C index --slow5 $W/$COND.blow5 $W/$COND.fastq
    $F5C eventalign --pore r10 --min-recalib-events 25 --signal-index --scale-events \
        --collapse-events --secondary=no -t $T \
        --slow5 $W/$COND.blow5 --reads $W/$COND.fastq \
        --bam $W/$COND.sorted.bam --genome $REF \
        --summary $W/$COND.summary > $W/$COND.eventalign 2> $W/eventalign.log

    # pileup with the stable samtools (env one segfaults)
    $SAM mpileup -f $REF $W/$COND.sorted.bam > $W/$COND.pileup 2>/dev/null

    # ORCA feature extraction (same --work_dir + --prefix across all three)
    orca-pred_signal_feature_ext --eventalign $W/$COND.eventalign --work_dir $W --prefix $COND --n_processes $T
    orca-pred_bascal_feature_ext  --pileup     $W/$COND.pileup     --work_dir $W --prefix $COND --n_processes $T
    orca-pred_feature_merge       --work_dir   $W --prefix $COND --n_processes $T

    echo "$COND DONE: merged sites = $(wc -l < $W/$COND.merged.feature.per.site)"
done
echo "=== ALL CONDITIONS FEATURIZED: $(date) ==="
