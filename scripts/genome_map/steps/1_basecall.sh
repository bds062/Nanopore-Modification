#!/bin/bash
# Dorado basecalling with move tables, aligned to REF. One array task per .pod5 file.
source "${GM_DIR:?}/steps/common.sh"   # sbatch runs a spooled copy, so not $0
mapfile -t P < <(ls "$POD5"/*.pod5)
p=${P[$SLURM_ARRAY_TASK_ID]}
name=$(basename "$p" .pod5)
mkdir -p "$PRE/bc"
echo "$(date) basecalling $name on $(hostname)"
"$DORADO" basecaller --emit-moves --reference "$REF" "$DORADO_MODEL" "$p" \
  > "$PRE/bc/$name.bam" 2> "$PRE/bc/$name.err"
echo "$(date) done $name"
