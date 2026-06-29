#!/usr/bin/env bash
#SBATCH --job-name=dorado_benchmark
#SBATCH --time=12:00:00
#SBATCH --output=/fs/nexus-scratch/bds062/logs/dorado_benchmark_%j.out
#SBATCH --qos=scavenger
#SBATCH --mem=96G
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

set -euo pipefail

mkdir -p /fs/nexus-scratch/bds062/logs

echo "[$(date '+%F %T')] Starting Dorado benchmark job ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

run_dataset() {
  local name="$1"
  shift
  echo
  echo "[$(date '+%F %T')] ===== ${name}: start ====="
  echo "Command:"
  printf ' %q' "$@"
  echo
  "$@"
  echo "[$(date '+%F %T')] ===== ${name}: done ====="
}

COMMON_ARGS=(
  --device cuda:0
  --min-mapq 60
  --max-reads 30
  --max-images-per-base 1
  --target-bases AC
)

run_dataset barcode6 \
  /fs/nexus-scratch/bds062/Nanopore-Modification/scripts/run_deepmod_vs_dorado.sh \
  --dataset barcode6 \
  --pod5 /fs/cbcb-lab/storm/shared/umbc-ont-data/pod5_by_barcode/run1_jan31/single_end/barcode6_filtered.pod5 \
  --bam /fs/nexus-scratch/bds062/results/event_clustering_barcode6/basecalled/reads_refined.bam \
  --peaks /fs/nexus-scratch/bds062/results/event_clustering_barcode6/basecalled/peaks_refined.tsv \
  --reference /fs/cbcb-lab/storm/shared/umbc-ont-data/ref/SPO1_FJ230960.1.fasta \
  --level-table /fs/nexus-scratch/bds062/results/event_clustering_barcode6/uncalled_r1041_model_only_means.txt \
  --deepmod-model /fs/nexus-scratch/bds062/results/deep_modification/results6/best_model.pt \
  --out-dir /fs/nexus-scratch/bds062/results/deepmod_barcode6/dorado_comparison \
  "${COMMON_ARGS[@]}"


# run_dataset d9 \
#   /fs/nexus-scratch/bds062/Nanopore-Modification/scripts/run_deepmod_vs_dorado.sh \
#   --dataset d9 \
#   --pod5 /fs/nexus-scratch/bds062/results/event_clustering_d9/d9_small.pod5 \
#   --bam /fs/nexus-scratch/bds062/results/event_clustering_d9/reads_refined.bam \
#   --peaks /fs/nexus-scratch/bds062/results/event_clustering_d9/peaks_refined.tsv \
#   --reference /fs/cbcb-lab/storm/shared/rawhash2/data/d9_ecoli_r1041/ref.fa \
#   --level-table /fs/nexus-scratch/bds062/results/event_clustering_d9/uncalled_r1041_model_only_means.txt \
#   --deepmod-model /fs/nexus-scratch/bds062/results/deep_modification/results6/best_model.pt \
#   --out-dir /fs/nexus-scratch/bds062/results/deepmod_d9/dorado_comparison \
#   "${COMMON_ARGS[@]}"

echo "[$(date '+%F %T')] Dorado benchmark job complete"
