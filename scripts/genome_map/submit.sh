#!/bin/bash
# Build a RawMod genome map: submit the processing chain as SLURM jobs.
#
#   basecall -> collect/align -> refine -> merge -> featurize -> score
#
# Usage (flags, a config file, or both; flags override the file):
#   bash scripts/genome_map/submit.sh --pod5 DIR --ref FASTA --out DIR [options]
#   bash scripts/genome_map/submit.sh my_genome.env [options]
#
# Inputs
#   --pod5 DIR         raw signal (.pod5 files); always required
#   --ref FASTA        reference genome
#   --out DIR          output directory
#   --bam FILE         existing dorado basecalls made with --emit-moves; skips basecalling
#   --base B           modified base to score: T (5hmU, default), A (6mA), C (5mC/5hmC/4mC)
#   --sites FILE       score only these positions (contig, 0-based pos[, strand]); default: every B
#   --strands "+ -"    strands to score (default "+")
#   --contigs a,b      restrict to these contigs
#   --name NAME        prefix for output files (default: name of --out)
#   --checkpoint PATH  model to score with (default checkpoints/rawmod_final/mixed.pt)
# Control
#   --dry-run          print the jobs without submitting
#   --from STEP        start at basecall | refine | featurize | score (reuse earlier outputs)
#   --until preprocess stop after the refined BAM and peaks (input for test_external_sites.py)
#
# Tool locations and SLURM arguments are read from the config file, the environment or
# ~/.config/rawmod/paths.env (see paths.env.example and config.example.env).
set -euo pipefail
GM_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$GM_DIR/../.." && pwd)
set -a; source "${RAWMOD_PATHS_FILE:-$HOME/.config/rawmod/paths.env}" 2>/dev/null || true; set +a

DRY=0; FROM=basecall; UNTIL=score; CFG_IN=""; declare -A F=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1; shift;;
    --from) FROM=$2; shift 2;;
    --until) UNTIL=$2; shift 2;;
    --pod5) F[POD5]=$2; shift 2;;      --ref) F[REF]=$2; shift 2;;
    --out) F[OUT]=$2; shift 2;;        --bam) F[BAM]=$2; shift 2;;
    --base) F[TARGET_BASE]=$2; shift 2;; --sites) F[SITES]=$2; shift 2;;
    --strands) F[STRANDS]=$2; shift 2;; --contigs) F[CONTIGS]=$2; shift 2;;
    --name) F[NAME]=$2; shift 2;;      --checkpoint) F[CHECKPOINTS]="mixed=$2"; shift 2;;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *.env) CFG_IN=$(readlink -f "$1"); shift;;
    *) echo "unknown argument: $1 (see --help)"; exit 1;;
  esac
done
[[ -n $CFG_IN ]] && source "$CFG_IN"
for k in "${!F[@]}"; do printf -v "$k" '%s' "${F[$k]}"; done

# defaults
: "${OUT:?--out is required}" "${POD5:?--pod5 is required}" "${REF:?--ref is required}"
OUT=$(readlink -m "$OUT")
NAME=${NAME:-$(basename "$OUT")}
TARGET_BASE=${TARGET_BASE:-T}; STRANDS=${STRANDS:-+}; CONTIGS=${CONTIGS:-}; SITES=${SITES:-}
BAM=${BAM:-}; CHUNKS=${CHUNKS:-24}; REFINE_SHARDS=${REFINE_SHARDS:-16}
REPO=${REPO:-$REPO_ROOT}
CHECKPOINTS=${CHECKPOINTS:-mixed=$REPO/checkpoints/rawmod_final/mixed.pt}
PILEUP_MASK_BASES=${PILEUP_MASK_BASES:-0}
MAX_READS=${MAX_READS:-15}; MIN_READS=${MIN_READS:-10}; MIN_MAPQ=${MIN_MAPQ:-20}
PYTHON_ENV=${PYTHON_ENV:-${RAWMOD_ENV:-}}; REFINE_ENV=${REFINE_ENV:-${RAWMOD_REFINE_ENV:-remora}}
CONDA_SH=${CONDA_SH:-}
DORADO=${DORADO:-${RAWMOD_DORADO:-$(command -v dorado || true)}}
DORADO_MODEL=${DORADO_MODEL:-${RAWMOD_DORADO_MODEL:-}}
REFINE_PY=${REFINE_PY:-${RAWHASH2_DIR:-}/test/scripts/refine_moves_remora.py}
LEVELS=${LEVELS:-${RAWMOD_LEVEL_TABLE:-${RAWHASH2_DIR:-}/extern/local_kmer_models/uncalled_r1041_model_only_means.txt}}
SBATCH_CPU=${SBATCH_CPU:-${RAWMOD_SBATCH_CPU:-${RAWMOD_SBATCH_ARGS:-}}}
SBATCH_GPU=${SBATCH_GPU:-${RAWMOD_SBATCH_GPU:-${RAWMOD_SBATCH_ARGS:-} --gres=gpu:1}}

# checks
[[ $UNTIL == score || $UNTIL == preprocess ]] || { echo "--until must be preprocess or score"; exit 1; }
order=(basecall refine featurize score)
start=-1; for k in "${!order[@]}"; do [[ ${order[$k]} == "$FROM" ]] && start=$k; done
[[ $start -ge 0 ]] || { echo "--from must be one of ${order[*]}"; exit 1; }
[[ $TARGET_BASE =~ ^[ACGT]$ ]] || { echo "--base must be one of A C G T"; exit 1; }
need () { for v in "$@"; do [[ -n ${!v} ]] || { echo "missing setting: $v (config file, environment or paths.env)"; exit 1; }; done; }
need CONDA_SH PYTHON_ENV
[[ $start -le 0 && -z $BAM ]] && need DORADO DORADO_MODEL
[[ $start -le 0 && -n $BAM ]] && { [[ -s $BAM ]] || { echo "--bam $BAM not found"; exit 1; }; }
[[ $start -le 1 ]] && need REFINE_PY LEVELS
[[ -n $SITES && ! -s $SITES ]] && { echo "--sites $SITES not found"; exit 1; }

# record the resolved settings; every job reads this file
mkdir -p "$OUT/logs"
CFG=$OUT/genome_map.env
{ echo "# resolved by submit.sh on $(date)"
  for v in NAME POD5 BAM REF OUT TARGET_BASE STRANDS CONTIGS SITES CHUNKS REPO CHECKPOINTS \
           PILEUP_MASK_BASES MAX_READS MIN_READS MIN_MAPQ PYTHON_ENV REFINE_ENV CONDA_SH DORADO \
           DORADO_MODEL REFINE_PY LEVELS REFINE_SHARDS SBATCH_CPU SBATCH_GPU; do
    printf '%s=%q\n' "$v" "${!v}"; done; } > "$CFG"
export CFG GM_DIR
echo "settings: $CFG" >&2
LOG=(--output="$OUT/logs/%x_%A_%a.out" --error="$OUT/logs/%x_%A_%a.err")

dep=""
sub () {  # sub NAME RESOURCES... -- SCRIPT ; echoes the job id, chained on $dep
  local name=$1; shift; local args=(); while [[ $1 != -- ]]; do args+=("$1"); shift; done; shift
  local d=(); [[ -n $dep ]] && d=(--dependency=afterok:$dep)
  if [[ $DRY == 1 ]]; then echo "sbatch --job-name=${NAME}_$name ${args[*]} ${d[*]} $1" >&2; echo "DRY_$name"; return; fi
  sbatch --parsable --export=ALL --job-name="${NAME}_$name" "${LOG[@]}" "${args[@]}" "${d[@]}" "$1"
}

if [[ $start -le 0 ]]; then
  if [[ -n $BAM ]]; then
    echo "using existing basecalls $BAM (basecalling skipped)" >&2
  else
    n=$(ls "$POD5"/*.pod5 | wc -l)
    dep=$(sub basecall $SBATCH_GPU --array=0-$((n-1)) --cpus-per-task=8 --mem=64G --time=12:00:00 -- "$GM_DIR/steps/1_basecall.sh")
  fi
  dep=$(sub collect $SBATCH_CPU --cpus-per-task=8 --mem=32G --time=4:00:00 -- "$GM_DIR/steps/2_shard.sh")
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
    python "$GM_DIR/make_candidates.py" --ref "$REF" --target-base "$TARGET_BASE" --strand "$strand" \
      --chunks "$CHUNKS" --out-dir "$OUT/candidates_$s" ${CONTIGS:+--contigs "$CONTIGS"} ${SITES:+--sites "$SITES"} >&2
    n=$(ls "$OUT/candidates_$s"/chunk_*.tsv 2>/dev/null | wc -l)
    [[ $n -gt 0 ]] || { echo "no $strand-strand sites; skipping that strand" >&2; continue; }
    export STRAND=$strand
    feats+=("$(sub "feat_$s" $SBATCH_CPU --array=0-$((n-1)) --cpus-per-task=6 --mem=32G --time=8:00:00 -- "$GM_DIR/steps/5_featurize.sh")")
  done
  [[ ${#feats[@]} -gt 0 ]] || { echo "nothing to score"; exit 1; }
  dep=$(IFS=:; echo "${feats[*]}")
fi
dep=$(sub score $SBATCH_GPU --cpus-per-task=8 --mem=48G --time=8:00:00 -- "$GM_DIR/steps/6_score.sh")
echo "submitted; final job $dep."
echo "When it finishes, scores are in $OUT/scores/ and the map is drawn with:"
echo "  python $GM_DIR/make_genome_map.py --scores mixed=$OUT/scores/${NAME}_mixed_strand+_scores.tsv.gz --out $OUT/genome_map.png"
