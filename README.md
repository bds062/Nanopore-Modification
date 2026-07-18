# RawMod — modification-agnostic nanopore base-modification detection

Detects DNA base modifications from raw nanopore signal **without being told which
modification to look for**. Reads are stacked into a DeepVariant-style *pileup
image* per candidate reference position and classified by a per-read conv trunk +
cross-read transformer (`ConvFormerV2`, ~209K params).

This repo holds the scripts that produce the current results, in the order they
run. Heavy data (POD5, BAM, `features.h5`, checkpoints) lives on scratch, not here.

---

## Layout

| Path | What it is |
|---|---|
| `deepmod/` | Core library. `featurization.py` builds pileup tensors; `model.py` holds `PileupDataset`, the architectures, splits and eval. `lodo.py`/`visualization.py` are imported by `model.py` and must stay. |
| `pipeline/` | Raw signal → ground truth → `features.h5`. |
| `experiments/pipeline1/` | ONT + UMCES cross-dataset + leave-one-modification-out study. |
| `experiments/pipeline2/` | All-genome mixed / LODO / LOMO study (the current headline experiment). |
| `analysis/` | Figure generation and diagnostics. |

## Paths

Scripts reference absolute scratch paths, since that is where the data lives:

```
/fs/cbcb-scratch/bds062/data/benchmark/       # POD5 + references
/fs/cbcb-scratch/bds062/data/gt/              # ground-truth BEDs
/fs/cbcb-scratch/bds062/results/benchmark_results/     # features.h5 (v1)
/fs/cbcb-scratch/bds062/results/benchmark_results_v2/  # features.h5 (balanced labels)
/fs/cbcb-scratch/bds062/results/deepmod_genomes/manifest.tsv     # dataset -> features/role
/fs/cbcb-scratch/bds062/results/deepmod_genomes/manifest_v2.tsv  # same, v2 features
```

The manifest is the single source of truth for which dataset is `train` vs `test`
and which `features.h5` backs it. `MANIFEST=<path>` overrides it.

---

## Reproducing, end to end

### 1. Ground truth

Three independent sources, none derived from the nanopore signal being modelled:

```bash
python pipeline/motif_gt.py --ref REF.fa.gz --preset ecoli_dam --outdir data/gt/Ecoli_DM
python pipeline/extract_gt_bismark.py ...     # EM-seq / WGBS (arabidopsis)
python pipeline/extract_gt_from_pileup.py ... # pre-computed bedMethyl
```

`motif_gt.py` writes `gt_modified.bed` (positives). **It also writes
`candidate.bed` identical to it** — using that as the candidate set makes a
dataset 100% positive, which is what step 3 fixes.

### 2. Basecall → refine → featurize

```bash
bash pipeline/submit_all.sh          # per-dataset table; calls pipeline.sh
```

`pipeline.sh` runs dorado basecalling, Remora move refinement, then
`deepmod/featurization.py`, emitting `(31, 210, 9)` tensors — 1 reference row +
30 read rows, 21 window positions × 10 samples, 9 channels
(`raw_signal, dwell_log1p, is_A, is_C, is_G, is_T, strand, mapq_norm, matches_ref`).

### 3. Balanced labels (Dorado-screened negatives)

Motif ground truth alone yields *only* positives. Negatives are non-motif A/C
positions that Dorado **confidently calls unmodified** — a non-motif base Dorado
flags as modified (possible uncharacterised MTase) is excluded rather than
mislabelled 0.

```bash
sbatch pipeline/dorado_modbasecall.sh    # DATASET, POD5_DIR, REF, OUTDIR
                                         # -> modbam + modkit pileup
sbatch pipeline/refeaturize_screened.sh  # + GT_BED, PILEUP, EXCLUDE_MOTIFS
                                         # -> build_screened_candidates.py
                                         #    -> featurize -> rechunk
```

Result: E. coli datasets go from 100% positive to ~50/50.

Memory: the output tensor is `n × 31 × 210 × 9` float32, so a 500K-site dataset
needs ~117 GB just for that array — use `--qos=highmem --mem=400G` for the large
ones (`qos=high` caps at 128 G).

### 4. Storage layout matters

`rechunk_features.py` re-chunks `tensors` from 64 images/chunk to **1**. With
64-image chunks a random read decompresses all 64 (**45 img/s**); at 1 it is
**1,536 img/s** (34×). That is what makes streaming (`PILEUP_PRELOAD=0`) viable
and drops training from `mem=460G` (whole split in RAM) to `mem=48G`.

### 5. Train + evaluate

```bash
# pipeline2 — mixed + leave-one-dataset-out + leave-one-modification-out
MANIFEST=.../manifest_v2.tsv OUTDIR=.../results2 \
  bash experiments/pipeline2/run_pipeline2.sh

python experiments/pipeline2/dorado_baseline2.py --out-dir results2   # baseline bars
python experiments/pipeline2/collect2.py        --out-dir results2   # merge + figures
```

Folds: `mixed`, `lodo_<dataset>` (×8), `lomo_{5mC,5hmC,6mA,5hmU}`.
Each fold job streams from disk: `PILEUP_PRELOAD=0 PILEUP_WORKERS=8`,
`--mem=48G --qos=high`.

**LOMO uses pipeline1's test definition** so the numbers are directly comparable:
`ONT_heldout_<mod>` (that modification's whole ONT file) and
`UMCES_heldout_<mod>` (`umces_lomo_split`, modkit-dominant-code typing, T ⇒ 5hmU).

### 6. Baseline

`dorado_baseline{,2}.py` runs Dorado specialist models as an **OR-ensemble**
(5mC_5hmC + 6mA; any active code > 50% ⇒ modified) to simulate a
modification-agnostic caller. For LOMO the held-out modification's code is
dropped. 5hmU has no Dorado model at all — that is the "cannot detect 5hmU" bar,
not a bug. Scope is ONT + UMCES: the bacteria/arabidopsis BAMs carry no MM/ML
tags, so Dorado has no calls there.

---

## Analysis

```bash
python analysis/make_reverse_complement_plots.py --out-dir .../reverse_complement
python analysis/visualize_h5_pileup.py --h5 features.h5 --cartoon
```

---

## Known issues

**Strand handling.** `get_ref_info_from_bam` (`deepmod/featurization.py:203-205`)
reverse-complements the reference span for a reverse read and reverses its
positions, but sites are keyed by `(contig, pos)` with **no strand**. Measured
consequences on Ecoli_DM (6mA at palindromic GATC):

- Candidate images are **strand-pure** — each site is 100% forward or 100%
  reverse reads, never mixed.
- The reference row is **50% A / 50% T** at 6mA sites, though every 6mA is on an A.
- Where the reference row reads T, **0% of reads match it** (`matches_ref = 0`)
  while every read still reads A.

Downstream code must therefore not trust the reference row's base identity.
`experiments/pipeline2/run_pipeline2.py` reads the reference base from the tensor
one-hot at `half_window * L` (**not** `center_idx * L`, a different quantity) and
types 6mA as `{A,T}` / 5mC as `{C,G}` to absorb the strand collapse.

Per-strand calling is not yet implemented; other callers (modkit, nanopolish)
call per strand by default.
