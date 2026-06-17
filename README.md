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
