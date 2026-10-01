#!/bin/bash
# Remora reference-anchored move refinement, one array task per shard.
source "${GM_DIR:?}/steps/common.sh"   # sbatch runs a spooled copy, so not $0
set +u; source "$CONDA_SH"; conda activate "$REFINE_ENV"; set -u
i=$SLURM_ARRAY_TASK_ID
mkdir -p "$PRE/refine"
python "$REFINE_PY" --pod5 "$POD5" --bam "$PRE/shards/shard_$i.bam" \
  --level-table "$LEVELS" --ref-mapping --min-mapq "$MIN_MAPQ" \
  --output "$PRE/refine/moves_$i.tsv" --output-peaks "$PRE/refine/peaks_$i.tsv" \
  --output-bam "$PRE/refine/reads_refined_$i.bam"
echo "$(date) done shard $i"
