# RawMod

Detects DNA base modifications from raw nanopore signal. Per-read pileup
images classified by `ConvFormerV2`.

## Pretrained checkpoint

`checkpoints/rawmod_final/mixed.pt` is the general-purpose model: use it to
score new data. `loco_<CHEM>.pt` hold out `CHEM` from training. Use them only to
reproduce zero-shot results, or to ask what a model that never saw `CHEM` would
say. Don't use them to score `CHEM` in new data.

The model is trained with BCE + supervised contrastive (weight 1.0, temperature
0.20) + DeepSAD (hinge, dimension 16), with **base identity visible**. Score
with `PILEUP_MASK_BASES=0` (the default).

| Fold | AUROC [95% CI] | AUPRC | n_pos / n_test |
|---|---:|---:|---:|
| `mixed` | 0.991 [0.991, 0.992] | 0.989 | 12,769 / 27,715 |
| `loco_5mC` | 0.992 [0.990, 0.995] | 0.955 | 487 / 3,441 |
| `loco_5hmC` | 0.993 [0.991, 0.995] | 0.922 | 256 / 3,210 |
| `loco_6mA` | 0.780 [0.778, 0.783] | 0.711 | 41,175 / 87,405 |
| `loco_4mC` | 0.905 [0.896, 0.913] | 0.936 | 3,432 / 5,652 |
| `loco_5hmU` | 0.513 [0.507, 0.520] | 0.716 | 23,942 / 34,104 |

Test sets are strand-resolved (one image per site and strand). Per-fold numbers
are in `checkpoints/rawmod_final/metrics.tsv`. Training run:
`/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results81_vis_supsad_s42`.

`checkpoints/results20_sad_dim16/` is the earlier (v0 paper) model, kept for
reproducing old results. Its numbers came from forward-strand-only test sets
and are not comparable with the table above.

**Caveat with bases visible:** on ground truth built from motif lists (REBASE
presets), the reference sequence alone predicts the label (11-mer AUROC
0.88–1.00). On such data, report RawMod next to that sequence-only baseline, or
compare treated and control samples at identical coordinates.

## Paths

```
/fs/cbcb-lab/storm/bds062/data/benchmark/                                 POD5 and references, benchmark organisms
/fs/cbcb-scratch/bds062/data/human/{hg001,hg002}/pod5/                     POD5, human
/fs/cbcb-scratch/bds062/data/gt_strand/                                    strand-resolved ground-truth BEDs
/fs/cbcb-scratch/bds062/results/benchmark_results/                         reads_refined.bam / peaks_refined.tsv
/fs/cbcb-lab/storm/bds062/rawmod_strandres_merged_fast/features/           training features.h5 (<dataset>_{plus,minus}.h5)
/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/                       run_matched_loco.py output: models/, metrics/, scores/
```

## Install

```bash
git clone https://github.com/bds062/Nanopore-Modification.git
cd Nanopore-Modification          # the repository folder (the package inside it is called rawmod)
pip install -e .
```

Run every command below from inside `Nanopore-Modification/`, in the Python
environment you installed into.

You also need three things that are not in this repository:

| What | Used for | Where to get it |
|---|---|---|
| [dorado](https://github.com/nanoporetech/dorado) + the `dna_r10.4.1_e8.2_400bps_sup@v5.0.0` model | basecalling with move tables | Oxford Nanopore |
| `refine_moves_remora.py` (needs [Remora](https://github.com/nanoporetech/remora) and samtools) | reference-anchored signal refinement | RawHash2, `test/scripts/` |
| `uncalled_r1041_model_only_means.txt` | expected current per k-mer ("level table") | RawHash2, `extern/local_kmer_models/` |

The CBCB-cluster paths for all three are in `scripts/genome_map/config.example.env`.

## Which workflow do I need?

| You have | You want | Use |
|---|---|---|
| pod5 + reference, no ground truth | modification scores along a genome | **Scoring a new genome** (below) |
| pod5 + reference + a list of sites whose modification status you know | an AUROC / F1 for RawMod on those sites | **Scoring a site list** (further below) |
| a `features.h5` someone already made | scores for it | `score_genome.py` (last step of either workflow) |

### Input files, in plain terms

| File | What it is | Where it comes from |
|---|---|---|
| `pod5/` | raw nanopore signal (R10.4.1, 5 kHz) | the sequencer |
| reference `.fasta` | the genome the reads are aligned to | you |
| `reads_refined.bam` | reads aligned to the reference, with refined move tables | preprocessing (`submit.sh --until preprocess`) |
| `peaks_refined.tsv` | per-read signal-to-base boundaries from refinement | same step as the BAM |
| **sites file** (`--sites`) | **which positions to score**: a tab-separated table with a `contig` and a `pos` column, one site per row | you, or `make_candidates.py` |
| **ground-truth BED** (`--gt`) | **which of those positions are modified**: two columns, contig and position, no header, one modified site per row. Every site in `--sites` that is *not* listed here counts as unmodified | you (e.g. from bisulfite / EM-seq, or a known mutant) |
| `features.h5` | the pileup images RawMod reads (HDF5 = a binary file format for large arrays); one image per site, built from up to 15 reads | `rawmod/featurization.py` (run for you by both workflows) |

Positions everywhere are **0-based** (first base of a contig is `0`), and
`contig` must match the names in the reference FASTA and BAM.

## Scoring a new genome (genome map)

This scores every site of one base across a genome, with no ground truth. It is
how the *A. carterae* contig_3762 5hmU map was made. Requirements: R10.4.1
reads at 5 kHz (the chemistry RawMod was trained on) and a reference FASTA.

```bash
cp scripts/genome_map/config.example.env my_genome.env   # set POD5, REF, OUT, TARGET_BASE, STRANDS
bash scripts/genome_map/submit.sh my_genome.env --dry-run  # print the sbatch chain
bash scripts/genome_map/submit.sh my_genome.env
```

`submit.sh` chains these SLURM jobs with `afterok` dependencies:

| Step | Script | Output (under `$OUT`) |
|---|---|---|
| basecall | `steps/1_basecall.sh`: dorado `--emit-moves --reference`, one task per pod5 | `preprocess/bc/*.bam` |
| shard | `steps/2_shard.sh`: merge, split read names into `REFINE_SHARDS` | `preprocess/shards/` |
| refine | `steps/3_refine.sh`: Remora reference-anchored move refinement | `preprocess/refine/` |
| merge | `steps/4_merge.sh` | `preprocess/reads_refined.bam`, `peaks_refined.tsv` |
| candidates | `make_candidates.py` (run at submit time): every target-base site | `candidates_{plus,minus}/chunk_NN.tsv` |
| featurize | `steps/5_featurize.sh`: `rawmod/featurization.py`, one task per chunk | `feat_{plus,minus}/chunk_NN.h5` |
| score | `steps/6_score.sh`: `scripts/test/score_genome.py` for each checkpoint | `scores/<NAME>_<model>_strand<±>_scores.tsv.gz` |

Then draw the map:

```bash
python scripts/genome_map/make_genome_map.py \
  --scores mixed=$OUT/scores/${NAME}_mixed_strand+_scores.tsv.gz \
  --contig contig_3762 --gff annotations.gff3 --window 10000 \
  --out $OUT/genome_map.png          # also writes genome_map.windows.tsv
```

Notes:

- **Reusing earlier steps.** `--until preprocess` stops after the refined BAM
  and peaks are written (all you need for **Scoring a site list**). `--from featurize` reuses an existing
  `$OUT/preprocess/{reads_refined.bam,peaks_refined.tsv}`. `--from score`
  rescores the existing `feat_*/chunk_*.h5` files, e.g. with a new checkpoint.
  Feature files don't depend on the model, so rescoring never needs
  re-featurization.
- **`TARGET_BASE`** is the base that carries the mark on the read: `T` for
  5hmU, `A` for 6mA, `C` for 5mC/5hmC/4mC.
  - On the `+` strand, candidates are reference positions with that base.
  - On the `-` strand, they are reference positions with its complement,
    featurized with `--strand - --orient read`, exactly as in training.
  - Use `STRANDS="+ -"` to map both strands; each strand gets its own scores
    file. The two strands are calibrated differently (on contig_3762 the mean
    score is 0.56 on `+` and 0.87 on `-`), so compare windows within a strand,
    not across strands.
- **Re-featurizing changes scores slightly.** The 15 reads kept per site are
  sampled, so re-featurizing the same data gives slightly different scores
  (site-level Spearman 0.92 between two runs on contig_3762). Window means are
  far more stable than single sites.
- **Keep the featurization flags fixed** (`--normalize`, 15 reads, 21-base
  window, `L=10`, `MIN_READS=10`, `MIN_MAPQ=20`). They reproduce the training
  features, and changing them gives the model images it was not trained on.
- **Single-site scores are noisy; windows are not.** When a site has more than
  15 reads, the featurizer samples 15 with a seeded random generator that runs
  through the whole job, so the draw for a site depends on how the candidates
  were chunked. On contig_3762, re-chunking left half the sites with identical
  images and scores, but moved the other half by a median of 0.02 (90th
  percentile 0.50). Scoring itself is deterministic. Interpret windows or
  regions, not individual sites, and keep `CHUNKS` fixed when comparing runs.
- **Compare within a strand.** On contig_3762 the minus strand scores much
  higher on average than the plus strand (0.87 vs 0.56 with `mixed.pt`). Rank
  windows against their own strand's median, and never mix strands in one
  ranking.
- **Read the map as relative.** Scores are model probabilities, not methylation
  fractions, and each checkpoint is calibrated differently. Compare windows
  against the contig median, and compare models by rank.
- **`mixed.pt` vs `loco_5hmU.pt`.** For a mark the model was trained on, use
  `mixed.pt`. Adding `loco5hmU=checkpoints/rawmod_final/loco_5hmU.pt` to
  `CHECKPOINTS` gives a zero-shot comparison track, but the 5hmU zero-shot
  AUROC is near chance (0.51), so treat that track as a sensitivity check only.

## Training data (featurization)

```bash
python scripts/ground_truth/split_gt_by_strand.py --ref REF.fa --bed gt.bed --outdir data/gt_strand/<name>   # -> gt_plus.bed / gt_minus.bed
python scripts/featurize/refeaturize_strand_resolved.py --dry-run         # one image per (site, strand)
python scripts/featurize/refeaturize_strand_resolved.py
```

Ground-truth builders (`scripts/ground_truth/`) and basecalling and refinement
(`scripts/ground_truth/submit_all.sh`) are unchanged:

```bash
python scripts/ground_truth/motif_gt.py --ref REF.fa.gz --preset ecoli_dam --outdir data/gt/Ecoli_DM
python scripts/ground_truth/extract_gt_bismark.py ...     # EM-seq / WGBS
python scripts/ground_truth/extract_gt_from_pileup.py ... # pre-computed bedMethyl
bash scripts/ground_truth/submit_all.sh                   # per-dataset table; invokes pipeline.sh
```

## Training and evaluation

Reproduces `checkpoints/rawmod_final` (mixed + the five LOCO folds):

```bash
cd scripts/train
RAWMOD_STRANDRES_ROOT=/fs/cbcb-lab/storm/bds062/rawmod_strandres_merged_fast/features \
RAWMOD_DATA_GEN=strandres EXTRA_ORGANISMS=1 INCLUDE_HUMAN=1 \
TF_LAYERS=2 EXCLUDE_UNTYPED_POS=1 CURRICULUM=1 \
POS_BASE_CAP_RATIO=4 POS_DROP_BASES=T \
CTRL_TARGET_BASE_TO_TEST=orphan NEG_CAP_SPO1=200000 \
PILEUP_MASK_BASES=0 BCE_WEIGHT=1.0 \
SUPCON_WEIGHT=1.0 SUPCON_DIM=128 SUPCON_TEMP=0.20 \
SAD_DIM=16 SAD_LOSS=hinge SAD_MARGIN=2.0 SAD_WEIGHT=1.0 SAD_ETA=1.0 \
OUTDIR=/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/<results_dir> \
bash run_matched_loco.sh --seed 42
```

- **Outputs.** Each fold trains and evaluates in one job, writing
  `metrics/<fold>.tsv` and per-site scores to `scores/<fold>__*.npz`.
  `--dry-run` prints the generated `sbatch` commands without submitting.
- **Choosing folds.** `FOLDS="mixed loco_6mA"` runs a subset.
  `FOLDS="logo_ecoli"` or `"logo_plant"` holds out a whole genome.
- **SLURM settings.** `TIME_LIMIT`, `PARTITION`, `GPU_TYPE` and `BEGIN`
  override the defaults (see the script header). Stagger large batches with
  `BEGIN=now+20minutes`: many jobs reading the same h5 files over NFS at once
  can fail with I/O errors.
- **Which checkpoint to use.** Every fold saves `models/<fold>/<fold>/best_model.pt`,
  the final model, and `models/<fold>/<fold>_cur1/best_model.pt`, the
  curriculum stage-1 checkpoint. Always use the first one.

## Scoring a site list (with ground truth)

Use this to measure how well RawMod does on sites whose answer you already
know. It needs two lists: the sites to score and the subset that is modified.
If you have no ground truth, use **Scoring a new genome** instead.

**1. Preprocess the reads** (skip if you already have `reads_refined.bam` and
`peaks_refined.tsv`):

```bash
cp scripts/genome_map/config.example.env my_sample.env      # set POD5, REF, OUT
bash scripts/genome_map/submit.sh my_sample.env --until preprocess
# -> $OUT/preprocess/reads_refined.bam, $OUT/preprocess/peaks_refined.tsv
```

**2. Write the two site files.** Example for 5mC at three cytosines on
`chr1`, of which positions 1042 and 1530 are methylated:

`sites.tsv`, the positions to score (header row required):
```
contig	pos
chr1	1042
chr1	1187
chr1	1530
```

`gt.bed`, the modified ones (no header):
```
chr1	1042
chr1	1530
```

Only the `contig` and `pos` columns of `sites.tsv` are read; any other column,
including a label, is ignored. Labels come only from `--gt`.

**3. Featurize and score in one command:**

```bash
python scripts/test/test_external_sites.py \
  --sites sites.tsv \
  --gt gt.bed \
  --pod5 /path/to/pod5_dir \
  --bam  $OUT/preprocess/reads_refined.bam \
  --peaks $OUT/preprocess/peaks_refined.tsv \
  --level-table /path/to/uncalled_r1041_model_only_means.txt \
  --checkpoint checkpoints/rawmod_final/mixed.pt \
  --strand + --normalize \
  --out-dir results/my_sample
```

It writes three files to `--out-dir`. All three are outputs; `--gt` is an input.

| File | Contents |
|---|---|
| `scores.tsv` | one row per site: `contig`, `pos`, RawMod `score` (0–1), ground-truth `label` |
| `metrics.tsv` | AUROC, AUPRC, F1 and so on for all sites together |
| `features.h5` | the images that were scored; reuse them with `--skip-featurize` |

Rules:

- **Always pass `--normalize`.** The model was trained on MAD-normalized
  signal; without the flag the images are z-scored, which cost 0.02–0.03
  AUROC on rice.
- **One strand per run.** With `--strand +`, list only positions whose
  reference base is the modified base (C for 5mC, A for 6mA, T for 5hmU). For
  the other strand, run again with `--strand -` and the complementary
  reference positions (G for 5mC, T for 6mA, A for 5hmU).
- **Coverage.** Sites with fewer than `--min-reads` reads (default 12) are
  dropped; lower it (e.g. `--min-reads 1`) for low-coverage data.
- **You need both classes.** If every site in `sites.tsv` is in `gt.bed` (or
  none is), the AUROC is undefined.

**`score_genome.py` is optional.** You don't need it after
`test_external_sites.py`, which already scores. Use it only when you already
have a `features.h5` (for example from the genome-map workflow, or to try
another checkpoint):

```bash
python scripts/test/score_genome.py --h5 results/my_sample/features.h5 \
  --dataset my_sample --checkpoint checkpoints/rawmod_final/mixed.pt \
  --out-dir results/my_sample
# -> results/my_sample/my_sample_scores.tsv.gz  (contig, pos, score, label, n_reads)
```

`--dataset` is just a name used for the output file.

## Analysis

```bash
python analysis/visualize_h5_pileup.py --h5 features.h5 --cartoon
bash analysis/chem_diversity_sweep/run_chem_diversity_sweep.sh
python analysis/orca_remake/loco_embedding_cluster.py --help
```
