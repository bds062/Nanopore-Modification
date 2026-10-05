# Reproducing the benchmark

This directory reproduces the site-level comparison of RawMod with Dorado, UniMeth, DeepMod2, Rockfish and MethyNano:
Table 1, Figure 3 and the supplementary benchmark tables and figures of the paper. It covers 15 ground-truth rows across
five modifications (5mC, 5hmC, 6mA, 4mC, 5hmU) from 11 datasets, plus seven additional rows (oligonucleotide
replicate-2 libraries and two bacteria with PacBio ground truth).

All results in Table 1 and Figure 3 are in distribution for RawMod: except mouse brain and *O. sativa*, every row comes
from a library that is in RawMod's training data, and every model is scored only at positions that RawMod never trained
on. The leave-out evaluation (step 3) repeats the RawMod column with each row's organism or library removed from training.

## Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Datasets](#datasets)
4. [Evaluation](#evaluation)
5. [Outputs](#outputs)
6. [Benchmarking another RawMod model](#benchmarking-another-rawmod-model)
7. [What is measured](#what-is-measured)
8. [Directory layout](#directory-layout)

## Prerequisites

We compare RawMod with the following tools. The versions are the ones used for the paper.

| Tool | Version | Used for |
|---|---|---|
| [Dorado](https://github.com/nanoporetech/dorado) | [1.4.0](https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.4.0-linux-x64.tar.gz) | basecalling of every sample (shared by all tools); Dorado modification models |
| Dorado models | [`dna_r10.4.1_e8.2_400bps_sup@v5.0.0`](https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r10.4.1_e8.2_400bps_sup@v5.0.0.zip) with [`_6mA@v3`](https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v3.zip), [`_4mC_5mC@v3`](https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_4mC_5mC@v3.zip), [`_5mC_5hmC@v3`](https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v3.zip), [`_5mCG_5hmCG@v3`](https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mCG_5hmCG@v3.zip) | basecalling and modification calls |
| [RawHash2](https://github.com/STORMgroup/RawHash2) | commit [`2ff150d`](https://github.com/STORMgroup/RawHash2/tree/2ff150d) | `test/scripts/refine_moves_remora.py` and the R10.4.1 9-mer level table (RawMod input) |
| [Remora](https://github.com/nanoporetech/remora) | 3.3.0 | signal refinement (environment of `refine_moves_remora.py`) |
| [UniMeth](https://github.com/sekeyWang/Unimeth) | 0.3.3, commit [`d79e2d7`](https://github.com/sekeyWang/Unimeth/tree/d79e2d7) | released 5 kHz 5mC and 6mA models ([checkpoints](https://drive.google.com/drive/folders/1f8bWVFmbPxL6WqukOUi_BufCEvpOHaxR)) |
| UniMeth (fine-tuning) | commit [`11215d4`](https://github.com/sekeyWang/Unimeth/tree/11215d4) + [`benchmark/patches/unimeth-11215d4-finetune.patch`](benchmark/patches/unimeth-11215d4-finetune.patch) | UniMeth-FT columns (5hmC, 4mC and 5hmU labels) |
| [DeepMod2](https://github.com/WGLab/DeepMod2) | commit [`ff910f6`](https://github.com/WGLab/DeepMod2/tree/ff910f6), model `bilstm_r10.4.1_5khz_v5.0` (in the repository) | 5mCG |
| [Rockfish](https://github.com/lbcb-sci/rockfish/tree/r10.4.1) | branch `r10.4.1`, commit [`a9fb2ce`](https://github.com/lbcb-sci/rockfish/tree/a9fb2ce), model [`rf_5kHz.ckpt`](https://drive.google.com/uc?id=1yD7zAq58uj2_lRb53Wu6TT4tt1hAw4sj) | 5mCG (needs an Ampere or newer GPU) |
| [MethyNano](https://github.com/baigeHUI/MethyNano) | commit [`4f66a60`](https://github.com/baigeHUI/MethyNano/tree/4f66a60) + [`benchmark/patches/methynano-4f66a60-predict_aln.patch`](benchmark/patches/methynano-4f66a60-predict_aln.patch) | 5mC (released checkpoints for three species) |
| Dorado (MethyNano input) | [0.9.2](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.9.2-linux-x64.tar.gz) with [`dna_r10.4.1_e8.2_400bps_hac@v4.2.0`](https://cdn.oxfordnanoportal.com/software/analysis/dorado/dna_r10.4.1_e8.2_400bps_hac@v4.2.0.zip) | the move tables MethyNano's features expect |
| [SAMtools](https://github.com/samtools/samtools) | 1.16.1 | sorting and indexing |
| [pod5](https://github.com/nanoporetech/pod5-file-format) | 0.3.39 | read subsets |
| [AWS CLI](https://aws.amazon.com/cli/) | any | downloads from the public ONT buckets (no account needed) |
| [SMRT Link](https://www.pacb.com/support/software-downloads/) (optional) | 13.1, Sequel II SMRT Tools | rebuilding the PacBio ground truth of the `_pb` rows |

The patches change the tools as follows:

* `unimeth-11215d4-finetune.patch` adds 5hmU, 5hmC and 4mC labels to UniMeth's fine-tuning (vocabulary token `[5hmU]`
  and the options `--hmU`, `--hmC`, `--m4C`), so that UniMeth can be fine-tuned on the same positions as RawMod.
* `methynano-4f66a60-predict_aln.patch` adds `predict_aln.py`, a copy of MethyNano's `predict.py` that maps each call to
  its reference position through the alignment CIGAR (`predict.py` adds the read offset to the alignment start, which
  ignores soft clips and indels), with options to restrict calls to given sites. The model and features are unchanged.

RawMod itself is installed from the repository root (`pip install -e .`, Python 3.11); the benchmark scripts also need
`pysam`, `pandas`, `scikit-learn` and `matplotlib`, which are RawMod dependencies.

### Example installation

Each tool keeps its own Python environment; follow each tool's README for its dependencies. The commands below fetch
the exact versions and apply the patches.

```bash
mkdir -p rawmod-bench-env && cd rawmod-bench-env

# Dorado 1.4.0 and its models; Dorado 0.9.2 and the hac@v4.2.0 model for MethyNano
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/dorado-1.4.0-linux-x64.tar.gz | tar xz
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.9.2-linux-x64.tar.gz | tar xz
M=https://cdn.oxfordnanoportal.com/software/analysis/dorado
mkdir -p dorado_models && for m in sup@v5.0.0 sup@v5.0.0_6mA@v3 sup@v5.0.0_4mC_5mC@v3 sup@v5.0.0_5mC_5hmC@v3 sup@v5.0.0_5mCG_5hmCG@v3; do
  wget -q "$M/dna_r10.4.1_e8.2_400bps_$m.zip" && unzip -q -d dorado_models "dna_r10.4.1_e8.2_400bps_$m.zip"; done
wget -q "$M/dna_r10.4.1_e8.2_400bps_hac@v4.2.0.zip" && unzip -q -d dorado-0.9.2-linux-x64/bin "dna_r10.4.1_e8.2_400bps_hac@v4.2.0.zip"

# RawHash2 (signal refinement script and level table)
git clone --recursive https://github.com/STORMgroup/RawHash2.git rawhash2 && (cd rawhash2 && git checkout 2ff150d)

# UniMeth: released version, and the fine-tuning version with the patch
git clone https://github.com/sekeyWang/Unimeth.git unimeth && (cd unimeth && git checkout d79e2d7)
git clone https://github.com/sekeyWang/Unimeth.git unimeth_ft && (cd unimeth_ft && git checkout 11215d4 && \
  git apply ../../Nanopore-Modification/test/benchmark/patches/unimeth-11215d4-finetune.patch)
# released checkpoints unimeth_r10.4.1_5kHz_{5mC,6mA}.pt: https://drive.google.com/drive/folders/1f8bWVFmbPxL6WqukOUi_BufCEvpOHaxR

# DeepMod2, Rockfish, MethyNano
git clone https://github.com/WGLab/DeepMod2.git && (cd DeepMod2 && git checkout ff910f6)
git clone -b r10.4.1 https://github.com/lbcb-sci/rockfish.git && (cd rockfish && git checkout a9fb2ce)
git clone https://github.com/baigeHUI/MethyNano.git && (cd MethyNano && git checkout 4f66a60 && \
  git apply ../../Nanopore-Modification/test/benchmark/patches/methynano-4f66a60-predict_aln.patch)
```

## Configuration

Every tool location is given once, in a site file. Copy the template and fill in the keys of the tools you run:

```bash
cp test/benchmark/config/site.example.env test/benchmark/config/site.env
python test/benchmark/rawmod_bench.py config          # every key, and whether it is set
```

| Needed for | Keys |
|---|---|
| all steps | `BENCH_WORK` (outputs, default `test/work`), `BENCH_DATA` (downloads, default `test/data`), `RAWMOD_PYTHON`, `SAMTOOLS`, `POD5` |
| basecalling (every tool uses it) | `DORADO`, `DORADO_MODELS` |
| RawMod | `RAWHASH2_DIR`, `REFINE_PYTHON` (environment with Remora 3.3) |
| UniMeth | `UNIMETH`, `UNIMETH_CHECKPOINTS` |
| UniMeth-FT | `UNIMETH_FT_PYTHONPATH` (patched source), `UNIMETH_FT_CHECKPOINTS` (written by `evaluation/train_unimeth_ft.sh`) |
| DeepMod2 / Rockfish / MethyNano | `DEEPMOD2`; `ROCKFISH`, `ROCKFISH_MODEL`; `METHYNANO_DIR`, `DORADO_092_DIR` |
| SLURM | `BENCH_SLURM_CPU`, `BENCH_SLURM_GPU` (sbatch arguments) |

The same keys are read from the environment and from `~/.config/rawmod/paths.env`. A tool whose keys are not set is
skipped with a message.

## Datasets

The scripts and a [README](data/README.md) in the [data directory](data/) download every dataset with the links we
used. Each script places its files where the benchmark expects them (`test/data/<dataset>/`).

```bash
cd test/data
bash download_d1_ecoli_k12.sh          # E. coli K-12: wild type, dam- dcm-, dam- dcm- + M.SssI
bash download_d2_anabaena_pcc7120.sh   # Anabaena PCC 7120
bash download_d3_hpylori_26695.sh      # H. pylori 26695: native and WGA
bash download_d4_athaliana.sh          # A. thaliana + EM-seq
bash download_d5_osativa.sh            # O. sativa + EM-seq
bash download_d6_mouse_brain.sh        # mouse brain + EM-seq
bash download_d7_human_hg002.sh        # HG002 + bisulfite
bash download_d8_ont_oligos.sh         # ONT all-5-mer oligonucleotides (5mC, 5hmC, 6mA, control; two replicates)
bash download_d9_spo1.sh               # SPO1 reference (the SPO1 nanopore libraries are not yet public)
bash download_d10_hpylori_j99.sh       # H. pylori J99 (additional rows; WITH_PACBIO=1 also fetches the PacBio reads)
bash download_d11_tdenticola.sh        # T. denticola (additional rows; WITH_PACBIO=1 also fetches the PacBio reads)
cd ../..
python test/benchmark/rawmod_bench.py datasets       # every input, its link, and whether it is on disk
```

The benchmark does not use whole runs: each sample is a fixed set of reads listed in `data/read_ids/`, and the
ground-truth positions of every row are fixed in `data/rows/`, so all tools are scored on the same reads and sites.

## Evaluation

The scripts in [evaluation](evaluation/) run the benchmark in order. Every step skips work whose output already
exists, so any script can be rerun after a failure and continues where it stopped.

```bash
# 0. (optional) UniMeth-FT: fine-tune UniMeth on RawMod's training positions (one GPU, ~3,000 steps per model)
bash test/evaluation/train_unimeth_ft.sh

# 1. run every tool on every sample (add --slurm to submit one SLURM job per task; --tools all includes UniMeth-FT)
bash test/evaluation/1_run_benchmark.sh --tools all

# 2. tables and figures at the positions RawMod never trained on, and at all sites
bash test/evaluation/2_score.sh

# 3. supplementary leave-out table (RawMod trained without each row's organism or library)
bash test/evaluation/3_leaveout.sh
```

| Step | What it does | Resource |
|---|---|---|
| `subset` | cut the sample's listed reads out of the downloaded pod5 files | CPU |
| `basecall` | Dorado 1.4.0 `sup@v5.0.0` with move tables, aligned to the reference; every tool reads this basecall | GPU |
| `readsel` | for each site, the K = 10 reads it is scored from (lowest seeded hash among primary reads with mean qscore >= 10); identical for every tool | CPU |
| `rows` | ground-truth positions: motif rows rebuilt from the reference, all others installed from `data/rows/` | CPU |
| `refine` | Remora reference-anchored signal refinement (RawMod input) | CPU |
| `tool` | each tool or model on each sample | GPU |
| `aggregate` | per-read calls to one score per site: the mean probability over the site's K reads (a selected read with no call counts as 0) | CPU |
| `score` | sequence-only nulls, tables, figures, `summary.md` | CPU |

`python test/benchmark/rawmod_bench.py status` shows what has run, per sample and tool.

## Outputs

`2_score.sh` writes to `$BENCH_WORK/results_own1_k10_seed0_q10_holdout-r81/rawmod/` (held-out positions, the paper's
numbers) and `$BENCH_WORK/results_own1_k10_seed0_q10/rawmod/` (all sites):

| File | In the paper |
|---|---|
| `tables/benchmark_models.tex` | Table 1: every model on every row |
| `fig_benchmark_main.pdf` | Figure 3: (a) mean AUROC and rows with AUROC >= 0.9 per model, (b) one threshold across all rows |
| `tables/benchmark_agnostic.tex` | Supplementary table: modification-agnostic AUROC and pooled AUROC |
| `tables/benchmark.tex` | Supplementary table: matched evaluation (each tool with the models it ships for the row's modification) |
| `fig_benchmark_roc_all.pdf` | Supplementary figure: ROC curves of every row |
| `tables/benchmark_leaveout.tex`, `leaveout.tsv` | Supplementary table: RawMod in distribution vs. leave-out (written by `3_leaveout.sh`) |
| `paper_models.tsv`, `paper_tables.tsv` | every number of the tables |
| `summary.md` | the main tables in plain text |

The sequence-only null of every row (AUROC of a logistic regression on the reference 11-mer) is in
`$BENCH_WORK/nulls_<tag>.tsv`. Per-sample intermediates live in `$BENCH_WORK/<sample>/`.

## Benchmarking another RawMod model

```bash
python test/benchmark/rawmod_bench.py rawmod --checkpoint path/to/best_model.pt --name mymodel [--slurm]
```

This registers the checkpoint in `benchmark/config/rawmod_models.tsv` as the column `rawmod_mymodel`, runs every step
it needs (finished steps such as the basecalls are skipped), and writes the tables and figures with the released model
and the new one side by side. The new column is scored exactly like the released one: same reads, same sites, same image
settings; its reads per image are read from the checkpoint.

* `--mask-bases`: the checkpoint was trained with base identity blanked.
* `--legacy-ch9`: the checkpoint was trained before the channel-9 fix.
* `--marks 5mC,6mA`: the modifications it was trained on (bold cells and "own rows" averages).
* `score --rawmod rawmod,mymodel`: which RawMod columns appear, in that order.

`--holdout r81` scores only the positions the released model never trained on
(`data/holdout/rawmod_seen_results81_mixed.tsv.gz`). For a model trained on a different split, these are not its
held-out positions.

## What is measured

* **Per model** (Table 1): every model on every row, on all of the row's sites. A model with no call at the row's base
  scores 0.5. Bold marks the rows of the modifications a model is trained for (`benchmark/config/claims.tsv`, fixed
  from each tool's documentation before scoring).
* **Matched**: each tool scored only with the models it provides for the row's modification and context, taking the
  per-site maximum when there are several; CpG-only models are scored on the CpG sites of mixed-context rows.
* **Modification-agnostic**: one score per site per tool (the maximum over its models), with no knowledge of the row's
  modification, and a **pooled AUROC with one threshold across all rows** (each row subsampled to at most 1,000
  positives and 1,000 negatives, seed 0).
* **Held-out positions**: a position counts as trained if any training image of the same organism lies on it, on either
  strand and under either label; trained positions are removed from every row (on two-sample rows from both libraries,
  so both classes stay at identical coordinates).
* **Nulls**: rows where the reference sequence alone predicts the label (null >= 0.9) are marked; two-sample rows
  (treated vs. control at the same positions) are about 0.5 by construction.

## Directory layout

| Path | Contents |
|---|---|
| `data/download_*.sh` | one download script per dataset ([README](data/README.md)) |
| `data/read_ids/` | the fixed read set of every sample, with counts and checksums (`MANIFEST.tsv`) |
| `data/rows/` | ground-truth positions of the rows that are not rebuilt from the reference, with checksums |
| `data/holdout/` | every position the released RawMod model trained on |
| `data/pod5_urls/` | file lists of the multi-file runs |
| `data/pacbio/run_ipdsummary.sh` | PacBio ipdSummary calls for the `_pb` rows |
| `data/unimeth_ft/` | 4mC label positions for UniMeth-FT |
| `evaluation/` | the scripts above |
| `benchmark/rawmod_bench.py` | the command-line entry point (`--help` lists every command) |
| `benchmark/bench/` | the pipeline: steps, runner, read selection, aggregation, scoring, tables, figures; `adapters/` parses each tool's output |
| `benchmark/config/` | datasets and links, rows, motifs, models and their documented marks, RawMod checkpoints, site template |
| `benchmark/patches/` | the UniMeth and MethyNano patches |
