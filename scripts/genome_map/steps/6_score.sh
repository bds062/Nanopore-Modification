#!/bin/bash
# Score every featurized chunk with each checkpoint and concatenate to
#   $OUT/scores/<NAME>_<model>_strand<+|->_scores.tsv.gz   (contig pos score label n_reads)
source "${GM_DIR:?}/steps/common.sh"   # sbatch runs a spooled copy, so not $0
set +u; source "$CONDA_SH"; conda activate "$PYTHON_ENV"; set -u
export PILEUP_MASK_BASES
mkdir -p "$OUT/scores/tmp"
for strand in $STRANDS; do
  s=$(sdir "$strand")
  for spec in $CHECKPOINTS; do
    m=${spec%%=*}; ck=${spec#*=}
    parts=()
    for h in "$OUT/feat_$s"/chunk_*.h5; do
      c=$(basename "$h" .h5); tag="${NAME}_${m}_${s}_${c}"
      o="$OUT/scores/tmp/${tag}_scores.tsv.gz"
      [[ -s "$o" ]] || python "$REPO/scripts/test/score_genome.py" --h5 "$h" --dataset "$tag" \
                         --checkpoint "$ck" --out-dir "$OUT/scores/tmp" --batch 1024
      parts+=("$o")
    done
    final="$OUT/scores/${NAME}_${m}_strand${strand}_scores.tsv.gz"
    # header from the first part, bodies from all (no `head` under pipefail: SIGPIPE)
    { zcat "${parts[0]}" | sed -n 1p; for f in "${parts[@]}"; do zcat "$f" | tail -n +2; done; } | gzip > "$final"
    echo "wrote $final ($(zcat "$final" | tail -n +2 | wc -l) sites)"
  done
done
