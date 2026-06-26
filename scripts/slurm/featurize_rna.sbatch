#!/bin/bash
#SBATCH --job-name=rna_featurize
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=huge-long
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/nfshomes/vgandhi/slurm_logs/rna_featurize_%A_%a.out
#SBATCH --error=/nfshomes/vgandhi/slurm_logs/rna_featurize_%A_%a.err
set -euo pipefail

# Step 2 of the DeepMod pipeline for RNA: build pileup HDF5 features from the
# refined peaks. One array task per condition (5 types x 2 replicates).
# Mirrors scripts/gen_features.sh conventions (no --target-base; control gets an
# empty --gt) with two RNA adjustments: --rna and --min-mapq 20 (the synthetic
# RNA reference yields low MAPQ, so the DNA default of 60 would drop most reads).

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-/fs/nexus-scratch/bds062/envs/campolina}

REPO=${REPO:-~/Nanopore-Modification}
cd "$(eval echo "$REPO")"
python -c "import pod5, pysam, h5py, numpy, tqdm" || {
    echo "ERROR: featurize deps missing in $(which python)." >&2; exit 1; }

DATA=${DATADIR:-/fs/nexus-scratch/vgandhi/rna_mod_validation}
REF=$DATA/references
OUT=$DATA/features_deepmod
mkdir -p $OUT

CONDITIONS=(control_rep1 m6A_rep1 m5C_rep1 pseU_rep1 Inosine_rep1 \
            control_rep2 m6A_rep2 m5C_rep2 pseU_rep2 Inosine_rep2)
declare -A GT=(
  [control_rep1]=""  [control_rep2]=""
  [m6A_rep1]=$REF/sampled_context_m6A_sites.bed
  [m6A_rep2]=$REF/sampled_context_m6A_sites.bed
  [m5C_rep1]=$REF/sampled_context_m5C_sites.bed
  [m5C_rep2]=$REF/sampled_context_m5C_sites.bed
  [pseU_rep1]=$REF/sampled_context_pseU_sites.bed
  [pseU_rep2]=$REF/sampled_context_pseU_sites.bed
  [Inosine_rep1]=$REF/sampled_context_inosine_sites.bed
  [Inosine_rep2]=$REF/sampled_context_inosine_sites.bed
)

COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
GTBED=${GT[$COND]}

ARGS=(
  -m deepmod.featurization
  --rna
  --pod5        $DATA/pod5/${COND}.pod5
  --bam         $DATA/basecalls_moves/${COND}_md.bam
  --peaks       $DATA/refined/${COND}_peaks_refined.tsv
  --level-table $REF/rna004_9mer_levels.tsv
  --normalize
  --min-mapq    20
  --half-window 10  --L 10  --max-reads 30  --max-images-per-base 5  --min-reads 5
  --output      $OUT/${COND}.h5
)
if [[ -n "$GTBED" ]]; then
    ARGS+=(--gt "$GTBED")
else
    ARGS+=(--gt)   # control: all-zero (unmodified) labels
fi

echo "=== Featurizing $COND (gt=${GTBED:-<empty/control>}) ==="
echo "Start: $(date)"
python "${ARGS[@]}"
echo "Done: $(date)"
