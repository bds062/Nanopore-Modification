#!/bin/bash
# Merge per-file BAMs and split read names into REFINE_SHARDS buckets for refinement.
source "${GM_DIR:?}/steps/common.sh"   # sbatch runs a spooled copy, so not $0
set +u; source "$CONDA_SH"; conda activate "$REFINE_ENV"; set -u
samtools cat -o "$PRE/reads.bam" "$PRE"/bc/*.bam
samtools quickcheck "$PRE/reads.bam"
mkdir -p "$PRE/shards"; rm -f "$PRE"/shards/names_*.txt
samtools view -F 0x900 "$PRE/reads.bam" | cut -f1 | sort -u |
  awk -v n="$REFINE_SHARDS" -v d="$PRE/shards" '{print > (d "/names_" (NR%n) ".txt")}'
for i in $(seq 0 $((REFINE_SHARDS-1))); do
  samtools view -b -N "$PRE/shards/names_$i.txt" -o "$PRE/shards/shard_$i.bam" "$PRE/reads.bam" &
done
wait
echo "merged $(samtools view -c "$PRE/reads.bam") records into $REFINE_SHARDS shards"
