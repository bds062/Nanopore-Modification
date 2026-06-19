#!/usr/bin/env bash
#SBATCH --job-name=dorado_benchmark
#SBATCH --time=1-00:00:00
#SBATCH --output=/fs/nexus-scratch/bds062/logs/dorado_benchmark_%j.out
#SBATCH --qos=scavenger
#SBATCH --mem=25G
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1

/fs/nexus-scratch/bds062/Nanopore-Modification/scripts/run_deepmod_vs_dorado.sh \
  --dataset barcode4 \
  --pod5 /fs/nexus-scratch/bds062/data/barcode4_filtered/barcode4_filtered.pod5 \
  --bam /fs/nexus-scratch/bds062/results/event_clustering_barcode4/basecalled/reads_refined_filtered.bam \
  --peaks /fs/nexus-scratch/bds062/results/event_clustering_barcode4/basecalled/peaks_refined.tsv \
  --reference /fs/cbcb-lab/storm/shared/umbc-ont-data/ref/SPO1_FJ230960.1.fasta \
  --level-table /fs/nexus-scratch/bds062/results/event_clustering_barcode4/uncalled_r1041_model_only_means.txt \
  --deepmod-model /fs/nexus-scratch/bds062/results/deep_modification/results6/best_model.pt \
  --out-dir /fs/nexus-scratch/bds062/results/deepmod_barcode4/dorado_comparison \
  --min-mapq 0


/fs/nexus-scratch/bds062/Nanopore-Modification/scripts/run_deepmod_vs_dorado.sh \
  --dataset d9 \
  --pod5 /fs/nexus-scratch/bds062/results/event_clustering_d9/d9_small.pod5 \
  --bam /fs/nexus-scratch/bds062/results/event_clustering_d9/reads_refined.bam \
  --peaks /fs/nexus-scratch/bds062/results/event_clustering_d9/peaks_refined.tsv \
  --reference /fs/cbcb-lab/storm/shared/rawhash2/data/d9_ecoli_r1041/ref.fa \
  --level-table /fs/nexus-scratch/bds062/results/event_clustering_d9/uncalled_r1041_model_only_means.txt \
  --deepmod-model /fs/nexus-scratch/bds062/results/deep_modification/results6/best_model.pt \
  --out-dir /fs/nexus-scratch/bds062/results/deemod_d9/dorado_comparison \
  --min-mapq 0