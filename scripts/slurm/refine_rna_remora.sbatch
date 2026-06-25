#!/bin/bash
#SBATCH --job-name=rna_refine
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=huge-long
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --output=/nfshomes/vgandhi/slurm_logs/rna_refine_%A_%a.out
#SBATCH --error=/nfshomes/vgandhi/slurm_logs/rna_refine_%A_%a.err
set -euo pipefail

# Step 1 of the DeepMod pipeline for RNA: Remora signal-mapping refinement.
# Produces refined reference-anchored peaks (+ moves) per condition, which feed
# deepmod.featurization. One SLURM array task per condition.
#
# Paths default to the RNA004 validation layout but can be overridden with
# environment variables, mirroring scripts/gen_features.sh.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-uncalled4-env}
module load samtools/1.16 2>/dev/null || true

# Dependencies for refine_moves_remora.py
python -c "import remora" 2>/dev/null || pip install ont-remora -q
python -c "import pod5"   2>/dev/null || pip install pod5 -q
python -c "import pysam"  2>/dev/null || pip install pysam -q
python -c "import tqdm"   2>/dev/null || pip install tqdm -q

DATADIR=${DATADIR:-~/rna_mod_validation}
DATADIR=$(eval echo "$DATADIR")
REF=${REF:-$DATADIR/references/sampled_context_strands.fa}
BAMDIR=${BAMDIR:-$DATADIR/basecalls_moves}
OUTDIR=${OUTDIR:-$DATADIR/refined}
LEVEL_TSV=${LEVEL_TSV:-$DATADIR/references/rna004_9mer_levels.tsv}
mkdir -p $OUTDIR

# RawHash2 refinement helper (clone if absent)
REFINE=${REFINE:-~/src/RawHash2/test/scripts/refine_moves_remora.py}
REFINE=$(eval echo "$REFINE")
if [[ ! -f $REFINE ]]; then
    mkdir -p ~/src
    git clone --depth 1 https://github.com/STORMgroup/RawHash2.git ~/src/RawHash2
fi

# Remora wants a header-less, whitespace-separated table with exactly 4^k rows.
LEVEL_TABLE=$DATADIR/references/rna004_9mer_levels_remora.txt
if [[ ! -f $LEVEL_TABLE ]]; then
    tail -n +2 $LEVEL_TSV > $LEVEL_TABLE
fi
echo "Level table rows: $(wc -l < $LEVEL_TABLE) (expect 262144)"

CONDITIONS=(${CONDITIONS:-control_rep1 m6A_rep1 m5C_rep1 pseU_rep1 Inosine_rep1 \
            control_rep2 m6A_rep2 m5C_rep2 pseU_rep2 Inosine_rep2})
COND=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "=== Refining $COND ==="
echo "Start: $(date)"

ALIGNED=$BAMDIR/${COND}_aligned.bam
POD5=$DATADIR/pod5/${COND}.pod5

# Reference-anchored refinement needs MD tags; add them if missing.
MDBAM=$BAMDIR/${COND}_md.bam
if [[ ! -f $MDBAM ]]; then
    echo "  adding MD tags (samtools calmd)..."
    samtools calmd -b $ALIGNED $REF > $MDBAM 2>/dev/null
    samtools index $MDBAM
fi

python $REFINE \
    --pod5 $POD5 \
    --bam $MDBAM \
    --level-table $LEVEL_TABLE \
    --output $OUTDIR/${COND}_moves_refined.tsv \
    --output-peaks $OUTDIR/${COND}_peaks_refined.tsv \
    --ref-mapping \
    --min-mapq 20

echo "Done: $(date)"
echo "  peaks: $(wc -l < $OUTDIR/${COND}_peaks_refined.tsv) reads"
echo "  moves: $(wc -l < $OUTDIR/${COND}_moves_refined.tsv) reads"
