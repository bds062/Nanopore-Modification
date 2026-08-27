#!/bin/bash
# ORCA exact featurization for the UMBC SPO1 dataset.
#
# Same pipeline as featurize_orca.sh, but the SPO1 layout differs from the ONT
# synthetic one: pod5 and BAMs are split per barcode under run/mode dirs, and
# the BAMs are already coordinate-sorted and indexed, so we read them in place
# from the lab share instead of re-sorting into a workspace.
#
# Labels are sample level, not per site: WGS barcodes are the modified class,
# PCR barcodes the unmodified class. That is much noisier than the synthetic
# GT BEDs, where every positive is a known modified position.
#
# NOTE: unlike the synthetic constructs (~155bp, which needed
# --min-recalib-events 25), SPO1 reads are long, so f5c's default calibration
# threshold is appropriate here.
set -euo pipefail

F5C=${F5C:-/fs/nexus-scratch/vgandhi/f5c-v1.6/f5c_x86_64_linux}
SAM=${SAM:-/fs/cbcb-software/RedHat-8-x86_64/local/samtools/1.16/bin/samtools}
DATA=${DATA:-/fs/cbcb-lab/storm/shared/umbc-ont-data}
REF=${REF:-$DATA/ref/SPO1_FJ230960.1.fasta}
OUT=${OUT:-/fs/nexus-scratch/vgandhi/orca_feat_spo1}

RUN=${RUN:-run1_jan31}          # run1_jan31 | run2_feb02
MODE=${MODE:-single_end}        # single_end matches the single-stranded results
POD5_SUB=${POD5_SUB:-}          # set to "high_quality" for the filtered subset
BARCODES=${BARCODES:-"barcode01 barcode02 barcode03 barcode04 barcode05 barcode06 barcode07"}
MINRECALIB=${MINRECALIB:-200}   # long reads: keep f5c's default
# SPO1 is a ~132kb genome with ~695k reads, so per-site depth runs into the
# thousands and ORCA's per-contig concat OOMs. Its signal features are sorted
# quantiles (depth invariant, min depth 10), so a few percent of reads is
# plenty. Set FRAC= to disable.
FRAC=${FRAC:-0.05}
PORE=${PORE:-r10}
T=${T:-4}

POD5_DIR=$DATA/pod5_by_barcode/$RUN/$MODE${POD5_SUB:+/$POD5_SUB}
BAM_DIR=$DATA/mapping/$RUN/$MODE

for BC in $BARCODES; do
    W=$OUT/$RUN/$MODE/$BC${FRAC:+_sub$FRAC}
    mkdir -p $W
    echo "==================== $RUN $MODE $BC ===================="
    date

    POD5=$POD5_DIR/$BC.pod5
    BAM=$BAM_DIR/$BC.sorted.bam
    for f in $POD5 $BAM $REF; do
        [ -f "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
    done

    # pod5 -> blow5 (skip if already done)
    [ -f $W/$BC.blow5 ] || blue-crab p2s $POD5 -o $W/$BC.blow5

    # subsample reads to keep the per-contig concat in ORCA's feature extraction
    # within memory. the share is read-only, so the subset lands in the workspace
    if [ -n "$FRAC" ]; then
        if [ ! -s $W/$BC.sub.bam ]; then
            $SAM view -s $FRAC -b -@ $T $BAM > $W/$BC.sub.bam
            $SAM index $W/$BC.sub.bam
        fi
        BAM=$W/$BC.sub.bam
        echo "$BC: using $FRAC subsample ($($SAM view -c -@ $T $BAM) reads)"
    fi

    # reads. the BAM is already sorted+indexed, so use it in place
    [ -f $W/$BC.fastq ] || $SAM fastq -F 0x900 $BAM > $W/$BC.fastq

    # eventalign is the expensive step (~3h for a 700MB BAM), so keep any
    # completed output on a rerun
    if [ ! -s $W/$BC.eventalign ]; then
        $F5C index --slow5 $W/$BC.blow5 $W/$BC.fastq
        $F5C eventalign --pore $PORE --min-recalib-events $MINRECALIB \
            --signal-index --scale-events --collapse-events --secondary=no -t $T \
            --slow5 $W/$BC.blow5 --reads $W/$BC.fastq \
            --bam $BAM --genome $REF \
            --summary $W/$BC.summary > $W/$BC.eventalign 2> $W/eventalign.log
    else
        echo "$BC: reusing existing eventalign ($(du -h $W/$BC.eventalign | cut -f1))"
    fi

    # yield check: this is what bit us on the synthetic data
    TOT=$(grep -c "^@" $W/$BC.fastq || true)
    BAD=$(grep -c "could not calibrate" $W/eventalign.log || true)
    echo "$BC reads=$TOT could-not-calibrate=$BAD"

    [ -s $W/$BC.pileup ] || $SAM mpileup -f $REF $BAM > $W/$BC.pileup 2>/dev/null

    orca-pred_signal_feature_ext --eventalign $W/$BC.eventalign --work_dir $W --prefix $BC --n_processes $T
    orca-pred_bascal_feature_ext  --pileup     $W/$BC.pileup     --work_dir $W --prefix $BC --n_processes $T
    orca-pred_feature_merge       --work_dir   $W --prefix $BC --n_processes $T

    echo "$BC DONE: merged sites = $(wc -l < $W/$BC.merged.feature.per.site)"
done
echo "=== SPO1 FEATURIZATION COMPLETE: $(date) ==="
