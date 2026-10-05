# RawMod

RawMod detects DNA base modifications directly from raw nanopore signal. For each
candidate position it assembles the signal of up to 15 overlapping reads into a
pileup image and scores it with a neural network trained on several modification
chemistries (5mC, 5hmC, 6mA, 4mC and 5hmU). A single model scores any of these
marks, and models trained without a given chemistry are provided to evaluate
transfer to modifications not seen during training.

## Contents

1. [Installation](#installation)
2. [Pretrained models](#pretrained-models)
3. [Choosing a workflow](#choosing-a-workflow)
4. [Genome maps](#genome-maps)
5. [Evaluation on sites with known modification status](#evaluation-on-sites-with-known-modification-status)
6. [Reproducing the benchmark](#reproducing-the-benchmark)
7. [Training](#training)
8. [Site configuration](#site-configuration)

## Installation

```bash
git clone https://github.com/bds062/Nanopore-Modification.git
cd Nanopore-Modification
pip install -e .
```

All commands below are run from the repository root (`Nanopore-Modification/`).

The preprocessing steps require the following software, which is not distributed
with this repository:

| Software | Purpose | Source |
|---|---|---|
| [Dorado](https://github.com/nanoporetech/dorado) with model `dna_r10.4.1_e8.2_400bps_sup@v5.0.0` | Basecalling with move tables | Oxford Nanopore Technologies |
| `refine_moves_remora.py`, with [Remora](https://github.com/nanoporetech/remora) and samtools | Alignment of the raw signal to reference bases | RawHash2, `test/scripts/` |
| `uncalled_r1041_model_only_means.txt` | Expected current level for each 9-mer (R10.4.1) | RawHash2, `extern/local_kmer_models/` |

Their locations are given once, in a site configuration file (see
[Site configuration](#site-configuration)).

## Pretrained models

The models in `checkpoints/rawmod_final/` were trained with a binary
cross-entropy objective combined with supervised contrastive learning (weight 1.0,
temperature 0.20) and Deep SAD (hinge loss, 16 dimensions).

| Model | Use | AUROC [95% CI] | AUPRC | Positives / test sites |
|---|---|---:|---:|---:|
| `mixed.pt` | Scoring new data | 0.991 [0.991, 0.992] | 0.989 | 12,769 / 27,715 |
| `loco_5mC.pt` | Zero-shot evaluation (5mC withheld) | 0.992 [0.990, 0.995] | 0.955 | 487 / 3,441 |
| `loco_5hmC.pt` | Zero-shot evaluation (5hmC withheld) | 0.993 [0.991, 0.995] | 0.922 | 256 / 3,210 |
| `loco_6mA.pt` | Zero-shot evaluation (6mA withheld) | 0.780 [0.778, 0.783] | 0.711 | 41,175 / 87,405 |
| `loco_4mC.pt` | Zero-shot evaluation (4mC withheld) | 0.905 [0.896, 0.913] | 0.936 | 3,432 / 5,652 |
| `loco_5hmU.pt` | Zero-shot evaluation (5hmU withheld) | 0.513 [0.507, 0.520] | 0.716 | 23,942 / 34,104 |

Test sets contain one pileup image per site and strand. Per-model metrics are
provided in `checkpoints/rawmod_final/metrics.tsv`. Use `mixed.pt` to score new
data; the `loco_<mark>.pt` models are intended only to measure how well RawMod
generalizes to a modification it has not seen.

`checkpoints/rawmod_leaveout/` contains the leave-out models of the benchmark: each was
trained with the configuration of `mixed.pt` but without one organism or library
(see [Reproducing the benchmark](#reproducing-the-benchmark)).

`checkpoints/results20_sad_dim16/` contains an earlier model, retained for
reproducibility. Its reported metrics were computed on different test sets and
are not comparable with the table above.

**Note on motif-derived ground truth.** RawMod uses the reference sequence as part
of its input. When the ground truth is defined by sequence motifs (for example,
every GATC site), the sequence alone predicts the label, and AUROC on such data
does not measure modification detection. In that setting, compare RawMod with a
sequence-only baseline, or compare treated and control samples at identical
positions.

## Choosing a workflow

| Available data | Goal | Workflow |
|---|---|---|
| Raw signal (`.pod5`) and a reference genome; optionally existing Dorado basecalls | Modification scores along a genome | [Genome maps](#genome-maps) |
| As above, plus a list of positions with known modification status | Accuracy (AUROC, AUPRC, F1) on those positions | [Evaluation on sites with known modification status](#evaluation-on-sites-with-known-modification-status) |

### File formats

| File | Content |
|---|---|
| `.pod5` | Raw nanopore signal (R10.4.1 flow cells, 5 kHz sampling). Required by every workflow. |
| Reference `.fasta` | Genome to which the reads are aligned. Contig names must match the BAM. |
| Basecalled BAM | Dorado output produced with `--emit-moves`. Aligned or unaligned. |
| `reads_refined.bam`, `peaks_refined.tsv` | Aligned reads and per-read signal boundaries after refinement. Produced by the preprocessing steps. |
| Site list (`.tsv`) | Positions to score: tab-separated, first column contig, second column position. A header line is optional. |
| Ground-truth BED | Positions known to be modified: two tab-separated columns (contig, position), no header. |
| `features.h5` | Pileup images in HDF5 format, one per scored position. Produced by `rawmod/featurization.py`. |

All positions are **0-based**: the first base of a contig is position 0.

## Genome maps

A genome map assigns a RawMod score to every occurrence of one base (for example,
every thymine for 5hmU) along a genome, or to a list of positions supplied by the
user. No ground truth is needed. The procedure runs as a chain of SLURM jobs that
are submitted together by `scripts/genome_map/submit.sh`.

### Step 1. Configure the site (once)

Copy `paths.env.example` to `~/.config/rawmod/paths.env` and set the locations of
the conda environments, Dorado, RawHash2 and the SLURM partitions. Every later
run reads this file.

### Step 2. Submit the jobs

To score every thymine of a genome (5hmU), starting from raw signal:

```bash
bash scripts/genome_map/submit.sh \
  --pod5 /path/to/pod5_dir \
  --ref  /path/to/genome.fasta \
  --out  /path/to/my_map
```

If the reads have already been basecalled with Dorado using `--emit-moves`, add
`--bam` to skip basecalling. An unaligned BAM is aligned to the reference
automatically.

```bash
bash scripts/genome_map/submit.sh \
  --pod5 /path/to/pod5_dir \
  --bam  /path/to/basecalls.bam \
  --ref  /path/to/genome.fasta \
  --out  /path/to/my_map
```

Basecalls produced without `--emit-moves` cannot be used, because the move table
links the raw signal to individual bases. In that case, omit `--bam` and RawMod
basecalls the reads itself. The `.pod5` files are required in either case.

Run the same command with `--dry-run` first to list the jobs without submitting
them.

### Step 3. Select the positions to score (optional)

| Option | Effect |
|---|---|
| `--base T` | Score every position of this base. T (5hmU) is the default; use A for 6mA and C for 5mC, 5hmC or 4mC. |
| `--sites my_sites.tsv` | Score only the listed positions (contig and 0-based position; see below). |
| `--contigs chr1,chr2` | Restrict scoring to these contigs. |
| `--strands "+ -"` | Score both strands. The default is the forward strand only. |
| `--checkpoint PATH` | Score with a different model. The default is `checkpoints/rawmod_final/mixed.pt`. |
| `--name NAME` | Prefix for output files. The default is the name of the output directory. |

A site list is a tab-separated file with the contig in the first column and the
0-based position in the second. An optional column named `strand` (`+` or `-`)
assigns positions to strands; without it, all positions are treated as forward
strand. For example:

```
contig	pos	strand
chr1	1042	+
chr1	1187	+
chr1	2210	-
```

Each listed position must carry the selected base on its strand: on the forward
strand the reference base must be the selected base (T for 5hmU), and on the
reverse strand it must be the complementary base (A for 5hmU). Positions that do
not satisfy this are reported and skipped. Reverse-strand positions are scored
only when `--strands "+ -"` is given.

### Step 4. Collect the results

The jobs write to the output directory:

| Path | Content |
|---|---|
| `scores/<name>_mixed_strand+_scores.tsv.gz` | One row per position: `contig`, `pos`, `score` (0–1), `label` (always 0 here), `n_reads` |
| `genome_map.env` | The settings used for the run |
| `logs/` | Job logs |
| `preprocess/` | Aligned and refined reads (reusable, see below) |

### Step 5. Draw the map

```bash
python scripts/genome_map/make_genome_map.py \
  --scores mixed=/path/to/my_map/scores/my_map_mixed_strand+_scores.tsv.gz \
  --gff annotations.gff3 \
  --window 10000 \
  --out /path/to/my_map/genome_map.png
```

The figure shows the mean score in consecutive windows along the contig
(`--window`, default 10 kb) against the contig median, a gene track if a GFF3 file
is given, and read coverage. The window values are also written to
`genome_map.windows.tsv`.

### Interpreting a genome map

- **Compare regions, not single positions.** At positions covered by more than 15
  reads, RawMod uses a random subset of 15 reads, so the score of an individual
  position varies between runs; window averages are stable.
- **Compare within a strand.** The two strands are scored on different scales, so
  each strand should be compared with its own median.
- **Treat scores as relative.** A score ranks positions by the strength of
  modification evidence. It is not the fraction of modified reads, and scores from
  different models are not on the same scale.

### Reusing earlier results

| Option | Effect |
|---|---|
| `--from featurize` | Reuse the reads in `preprocess/` from an earlier run, e.g. to score other positions. |
| `--from score` | Reuse the pileup images from an earlier run, e.g. to score with another model. |
| `--until preprocess` | Stop once the reads are aligned and refined. |

The settings may also be kept in a configuration file
(`scripts/genome_map/config.example.env`) and passed as
`bash scripts/genome_map/submit.sh my_genome.env`; command-line options take
precedence over the file.

## Evaluation on sites with known modification status

This workflow measures RawMod's accuracy on positions whose modification status
is known, for example from bisulfite or EM-seq data, or from a mutant lacking a
methyltransferase.

**1. Prepare the reads.** If `reads_refined.bam` and `peaks_refined.tsv` are not
available, produce them with the genome-map pipeline:

```bash
bash scripts/genome_map/submit.sh --pod5 /path/to/pod5_dir --ref /path/to/genome.fasta \
  --out /path/to/my_sample --until preprocess
```

**2. Prepare two lists.** `sites.tsv` contains all positions to evaluate, modified
and unmodified (header line required):

```
contig	pos
chr1	1042
chr1	1187
chr1	1530
```

`gt.bed` lists the modified positions among them (no header):

```
chr1	1042
chr1	1530
```

Positions in `sites.tsv` that are absent from `gt.bed` are treated as unmodified.
Only the first two columns of `sites.tsv` are used.

**3. Score the sites.**

```bash
python scripts/test/test_external_sites.py \
  --sites sites.tsv \
  --gt gt.bed \
  --pod5 /path/to/pod5_dir \
  --bam /path/to/my_sample/preprocess/reads_refined.bam \
  --peaks /path/to/my_sample/preprocess/peaks_refined.tsv \
  --level-table /path/to/uncalled_r1041_model_only_means.txt \
  --checkpoint checkpoints/rawmod_final/mixed.pt \
  --strand + --normalize \
  --out-dir results/my_sample
```

| Output | Content |
|---|---|
| `scores.tsv` | One row per position: `contig`, `pos`, `score`, `label` |
| `metrics.tsv` | AUROC, AUPRC, F1, precision and recall over all positions |
| `features.h5` | The scored pileup images; pass `--skip-featurize` to rescore them |

Requirements:

- **Normalization.** `--normalize` must be given; the models were trained on
  MAD-normalized signal.
- **One strand per run.** Each run covers one strand. For `--strand +`, list
  positions whose reference base is the modified base (C for 5mC, A for 6mA, T for
  5hmU). For `--strand -`, list positions whose reference base is its complement
  (G, T and A respectively).
- **Coverage.** Positions with fewer than `--min-reads` reads (default 12) are
  excluded. Lower this value for low-coverage data.
- **Both classes.** The list must contain both modified and unmodified positions
  for AUROC to be defined.

An existing `features.h5` can also be scored directly:

```bash
python scripts/test/score_genome.py --h5 features.h5 --dataset my_sample \
  --checkpoint checkpoints/rawmod_final/mixed.pt --out-dir results/my_sample
```

## Reproducing the benchmark

[`test/`](test/README.md) reproduces the paper's comparison of RawMod with Dorado,
UniMeth, DeepMod2, Rockfish and MethyNano on 15 ground-truth rows across five
modifications: download scripts with the links of every dataset
([`test/data`](test/data/README.md)), the fixed read sets and ground-truth
positions, and the scripts that run every tool and write Table 1, Figure 3 and the
supplementary benchmark tables ([`test/evaluation`](test/evaluation/)). Any RawMod
checkpoint can be added to the benchmark as its own column.

## Training

The released models are reproduced by `scripts/train/run_matched_loco.sh`, which
trains the mixed model and the five withheld-chemistry models as separate SLURM
jobs:

```bash
cd scripts/train
RAWMOD_STRANDRES_ROOT=/path/to/features \
RAWMOD_DATA_GEN=strandres EXTRA_ORGANISMS=1 INCLUDE_HUMAN=1 \
TF_LAYERS=2 EXCLUDE_UNTYPED_POS=1 CURRICULUM=1 \
POS_BASE_CAP_RATIO=4 POS_DROP_BASES=T \
CTRL_TARGET_BASE_TO_TEST=orphan NEG_CAP_SPO1=200000 \
PILEUP_MASK_BASES=0 BCE_WEIGHT=1.0 \
SUPCON_WEIGHT=1.0 SUPCON_DIM=128 SUPCON_TEMP=0.20 \
SAD_DIM=16 SAD_LOSS=hinge SAD_MARGIN=2.0 SAD_WEIGHT=1.0 SAD_ETA=1.0 \
OUTDIR=/path/to/output \
bash run_matched_loco.sh --seed 42
```

`--dry-run` prints the job submissions without running them, and `FOLDS` selects
a subset of models (for example `FOLDS="mixed loco_6mA"`). Each model is written
to `models/<fold>/<fold>/best_model.pt`, with test metrics in
`metrics/<fold>.tsv` and per-site test scores in `scores/`.

Training features are produced from the ground-truth data by
`scripts/featurize/refeaturize_strand_resolved.py`, which builds one pileup image
per position and strand. Ground-truth BED files are split by strand with
`scripts/ground_truth/split_gt_by_strand.py`.

## Site configuration

No machine-specific path is stored in this repository. Scripts read locations
from environment variables, or from `~/.config/rawmod/paths.env` (override with
`RAWMOD_PATHS_FILE`). `paths.env.example` lists every variable; only those used by
a given workflow need to be set.
