#!/bin/bash
# Submit the genome-map chain as SLURM jobs linked by afterok dependencies:
#   basecall -> shard -> refine -> merge -> candidates+featurize -> score
#
#   bash scripts/genome_map/submit.sh CONFIG.env [--dry-run] [--from STEP] [--until preprocess]
#
# --until preprocess stops after merge: you get $OUT/preprocess/reads_refined.bam and
# peaks_refined.tsv, the inputs scripts/test/test_external_sites.py needs.
# STEP is one of basecall|refine|featurize|score; --from featurize reuses an existing
# $OUT/preprocess/{reads_refined.bam,peaks_refined.tsv} (e.g. from an earlier run),
# --from score reuses $OUT/feat_*/chunk_*.h5 (e.g. to rescore with a new checkpoint).
# After the score job finishes, draw the map with make_genome_map.py.
set -euo pipefail
CFG=$(readlink -f "${1:?usage: submit.sh CONFIG.env [--dry-run] [--from STEP]}"); shift
DRY=0; FROM=basecall; UNTIL=score
while [[ $# -gt 0 ]]; do
  case $1 in --dry-run) DRY=1; shift;; --from) FROM=$2; shift 2;; --until) UNTIL=$2; shift 2;;
             *) echo "unknown arg $1"; exit 1;; esac
done
[[ $UNTIL == score || $UNTIL == preprocess ]] || { echo "--until must be preprocess or score"; exit 1; }
GM_DIR=$(cd "$(dirname "$0")" && pwd)
export CFG GM_DIR
# shellcheck disable=SC1090
source "$CFG"
mkdir -p "$OUT/logs"
LOG=(--output="$OUT/logs/%x_%A_%a.out" --error="$OUT/logs/%x_%A_%a.err")
order=(basecall refine featurize score)
start=-1; for k in "${!order[@]}"; do [[ ${order[$k]} == "$FROM" ]] && start=$k; done
[[ $start -ge 0 ]] || { echo "--from must be one of ${order[*]}"; exit 1; }

dep=""
sub () {  # sub NAME RESOURCES... -- SCRIPT ; echoes job id, chains on $dep
  local name=$1; shift; local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  local d=(); [[ -n $dep ]] && d=(--dependency=afterok:$dep)
  if [[ $DRY == 1 ]]; then echo "sbatch --job-name=${NAME}_$name ${args[*]} ${d[*]} $1" >&2; echo "DRY_$name"; return; fi
  sbatch --parsable --export=ALL --job-name="${NAME}_$name" "${LOG[@]}" "${args[@]}" "${d[@]}" "$1"
}

if [[ $start -le 0 ]]; then
  n=$(ls "$POD5"/*.pod5 | wc -l)
  dep=$(sub basecall $SBATCH_GPU --array=0-$((n-1)) --cpus-per-task=8 --mem=64G --time=12:00:00 -- "$GM_DIR/steps/1_basecall.sh")
  dep=$(sub shard $SBATCH_CPU --cpus-per-task=8 --mem=32G --time=4:00:00 -- "$GM_DIR/steps/2_shard.sh")
fi
if [[ $start -le 1 ]]; then
  dep=$(sub refine $SBATCH_CPU --array=0-$((REFINE_SHARDS-1)) --cpus-per-task=4 --mem=24G --time=8:00:00 -- "$GM_DIR/steps/3_refine.sh")
  dep=$(sub merge $SBATCH_CPU --cpus-per-task=8 --mem=32G --time=2:00:00 -- "$GM_DIR/steps/4_merge.sh")
fi
if [[ $UNTIL == preprocess ]]; then
  echo "submitted through merge; final job $dep -> $OUT/preprocess/{reads_refined.bam,peaks_refined.tsv}"
  exit 0
fi
if [[ $start -le 2 ]]; then
  set +u; source "$CONDA_SH"; conda activate "$PYTHON_ENV"; set -u
  feats=()
  for strand in $STRANDS; do
    s=$([[ $strand == "+" ]] && echo plus || echo minus)
    [[ $DRY == 1 ]] && echo "python make_candidates.py --strand $strand -> $OUT/candidates_$s" >&2 ||
    python "$GM_DIR/make_candidates.py" --ref "$REF" --target-base "$TARGET_BASE" --strand "$strand" \
      --chunks "$CHUNKS" --out-dir "$OUT/candidates_$s" ${CONTIGS:+--contigs "$CONTIGS"}
    export STRAND=$strand
    feats+=("$(sub "feat_$s" $SBATCH_CPU --array=0-$((CHUNKS-1)) --cpus-per-task=6 --mem=32G --time=8:00:00 -- "$GM_DIR/steps/5_featurize.sh")")
  done
  dep=$(IFS=:; echo "${feats[*]}")
fi
dep=$(sub score $SBATCH_GPU --cpus-per-task=8 --mem=48G --time=8:00:00 -- "$GM_DIR/steps/6_score.sh")
echo "submitted; final job $dep. Scores land in $OUT/scores/; logs in $OUT/logs/"
