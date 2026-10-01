#!/bin/bash
# Collect the basecalled reads into one aligned BAM and split it into REFINE_SHARDS
# parts for parallel refinement. Input: $BAM if set (existing basecalls), otherwise the
# per-file BAMs written by 1_basecall.sh. The BAM must carry dorado move tables (mv tag);
# an unaligned BAM is aligned to $REF with `dorado aligner`, which keeps the move tables.
source "${GM_DIR:?}/steps/common.sh"   # sbatch runs a spooled copy, so not $0
set +u; source "$CONDA_SH"; conda activate "$REFINE_ENV"; set -u
mkdir -p "$PRE"
if [[ -n "${BAM:-}" ]]; then src=("$BAM"); else src=("$PRE"/bc/*.bam); fi
samtools cat -o "$PRE/basecalls.bam" "${src[@]}"
samtools quickcheck "$PRE/basecalls.bam"
# count, not `grep -q`: under pipefail, head closing the pipe would read as "no tags"
n_mv=$(samtools view "$PRE/basecalls.bam" 2>/dev/null | head -n 1000 | grep -c $'\tmv:B:' || true)
if [[ ${n_mv:-0} -eq 0 ]]; then
  echo "ERROR: no move tables (mv tag) in the basecalled reads. Re-basecall with" >&2
  echo "       dorado basecaller --emit-moves; refinement needs the move tables." >&2
  exit 1
fi
if [[ $(samtools view -c -F 4 "$PRE/basecalls.bam") -eq 0 ]]; then
  [[ -n "${DORADO:-}" ]] || { echo "ERROR: reads are unaligned and DORADO is not set (needed for dorado aligner)" >&2; exit 1; }
  echo "reads are unaligned; aligning to $REF with dorado aligner"
  "$DORADO" aligner "$REF" "$PRE/basecalls.bam" > "$PRE/reads.bam"
else
  mv "$PRE/basecalls.bam" "$PRE/reads.bam"
fi
rm -f "$PRE/basecalls.bam"
mkdir -p "$PRE/shards"; rm -f "$PRE"/shards/names_*.txt
samtools view -F 0x900 "$PRE/reads.bam" | cut -f1 | sort -u |
  awk -v n="$REFINE_SHARDS" -v d="$PRE/shards" '{print > (d "/names_" (NR%n) ".txt")}'
for i in $(seq 0 $((REFINE_SHARDS-1))); do
  samtools view -b -N "$PRE/shards/names_$i.txt" -o "$PRE/shards/shard_$i.bam" "$PRE/reads.bam" &
done
wait
echo "split $(samtools view -c "$PRE/reads.bam") records into $REFINE_SHARDS shards"
