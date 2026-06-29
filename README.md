# DeepMod for Nanopore Modified-Base Detection

DeepMod converts aligned nanopore reads into DeepVariant-style pileup images and
trains an Inception-style convolutional neural network to classify modified
versus unmodified reference positions.

The current method uses:

- a reference row plus read rows in each pileup image
- raw signal, dwell time, base identity, strand, mapping quality, and
  reference-match channels
- binary modified-base labels from BED files
- precision-recall/AUPRC as the primary model-selection metric
- optional leave-one-dataset-out evaluation and permutation channel importance

Large data files, HDF5 features, checkpoints, and generated figures are not
tracked in this repository.

## Repository Layout

```text
deepmod/
  featurization.py      Build HDF5 pileup tensors from POD5 + BAM + moves/peaks
  model.py             Train/evaluate the PileupInceptionV3 model
  visualization.py     Plot PR curves, confusion matrices, LODO summaries
  lodo.py              Leave-one-dataset-out training/evaluation helpers

scripts/
  gen_features.sh      ONT benchmark featurization workflow
  extract_moves_from_bam.py
  visualize_deepmod.py
  slurm/train_deepmod.sbatch
  benchmarks/          Remora/Rockfish comparison workflows
  evaluation/          Reference-level prediction and plotting utilities

docs/
  r10_4_modified_base_datasets.md
```

## Installation

From the repository root:

```bash
python -m pip install -e .
```

The core dependencies are listed in `pyproject.toml`. GPU training requires a
PyTorch build compatible with the local CUDA environment.

## Featurization

Build HDF5 pileup tensors from POD5, an aligned BAM, move/peak boundaries, and
optional BED labels:

```bash
deepmod-featurize \
  --pod5 reads.pod5 \
  --bam reads.bam \
  --moves moves.tsv \
  --level-table kmer_levels.tsv \
  --gt modified_sites.bed \
  --output sample.h5 \
  --normalize \
  --min-reads 5 \
  --max-reads 30 \
  --max-images-per-base 5
```

For the local ONT benchmark layout, use:

```bash
scripts/gen_features.sh
```

Override paths and parameters with environment variables such as `OUT_DIR`,
`DATA_ROOT`, `LEVEL_TABLE`, `MAX_READS`, and `MAX_IMAGES_PER_BASE`.

## Training

Train the binary modified-base classifier:

```bash
deepmod-train \
  --input results/features/*.h5 \
  --out-dir results/training \
  --batch 128 \
  --num-workers 6
```

The training command writes:

- `best_model.pt`
- `test_predictions.npz`
- `precision_recall.png`
- `confusion_matrix.png`
- `training_curves.png`
- optional LODO and channel-importance artifacts

On the UMD cluster, submit:

```bash
sbatch scripts/slurm/train_deepmod.sbatch
```

To train on the local `results9/*.h5` tensors with a contig-level holdout and
write the run to `results10/`, submit:

```bash
sbatch scripts/slurm/train_deepmod_results10_contig_split.sbatch
```

That script runs `deepmod.model` with `--split-mode contig`, `--test-frac 0.30`,
and `--val-frac 0.10`. All images from the same reference contig are assigned to
one split only, so the final test metrics are measured on held-out contigs.

## Replot Existing Results

```bash
deepmod-visualize --out-dir results/training
```

This regenerates plots from saved `.npz`, `.pt`, and metric files without
retraining.

## Benchmark Utilities

The `scripts/benchmarks/` and `scripts/evaluation/` folders contain the current
Remora and Rockfish comparison utilities used during method development. They
are intentionally kept separate from the core package because they depend on
local dataset paths and external tool outputs.

Examples:

```bash
scripts/benchmarks/run_deepmod_remora_only.sh
scripts/benchmarks/run_rockfish_remora_only.sh
```

Both scripts default to the current `/fs/nexus-scratch/bds062` workspace layout,
but all important paths can be overridden with environment variables.

## Testing a Trained Model on a New Dataset

Use the generic pipeline runner when you have a POD5 file/directory, aligned
BAM, and either `peaks_refined.tsv` or a Dorado moves TSV:

```bash
scripts/run_deepmod_pipeline.sh \
  --dataset my_sample \
  --pod5 /path/to/sample.pod5 \
  --bam /path/to/reads_refined.bam \
  --peaks /path/to/peaks_refined.tsv \
  --level-table /path/to/uncalled_r1041_model_only_means.txt \
  --model /path/to/best_model.pt \
  --out-dir results/deepmod_my_sample
```

The runner writes HDF5 features, reference-level predictions, a score summary,
a score histogram, thresholded BED-style calls, and logs. Example launchers for
the current workspace live in:

```text
/fs/nexus-scratch/bds062/results/deepmod_barcode4/run_deepmod_test.sh
/fs/nexus-scratch/bds062/results/deemod_d9/run_deepmod_test.sh
```

## Comparing DeepMod to Dorado Mod Calls

Use the Dorado comparison runner when you want to treat Dorado MM/ML modified
base calls as reference labels for a quick sanity check:

```bash
scripts/run_deepmod_vs_dorado.sh \
  --dataset my_sample \
  --pod5 /path/to/sample.pod5 \
  --bam /path/to/reads_refined.bam \
  --peaks /path/to/peaks_refined.tsv \
  --reference /path/to/ref.fa \
  --level-table /path/to/uncalled_r1041_model_only_means.txt \
  --deepmod-model /path/to/best_model.pt \
  --out-dir results/deepmod_vs_dorado_my_sample
```

By default this uses Dorado 1.4.0 R10.4.1 SUP `v5.2.0` from the local
RawHash2 installation and runs the newest compatible DNA modification model in
each detected family, including 5mC/5hmC, CpG 5mC/5hmC, 6mA, and 4mC/5mC when
available. The runner asks Dorado to emit candidate modified-base calls with
`--modified-bases-threshold 0`, then applies its own `--dorado-threshold 0.5`
when making binary reference labels. The output includes Dorado-aligned BAMs,
Dorado reference-level labels, DeepMod-vs-Dorado joined tables, per-type PR/F1
and confusion plots, and a summary accuracy/F1/AUPRC figure.

Current workspace examples:

```bash
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
```

```bash
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
```
