# RawMod — modification-agnostic nanopore base-modification detection

Detects DNA base modifications from raw nanopore signal **without being told which
modification to look for**. Reads are stacked into a DeepVariant-style *pileup
image* per candidate reference position and classified by a per-read conv trunk +
cross-read transformer (`ConvFormerV2`, ~209K params), trained with a paired
supervised-contrastive (SupCon) objective on top of BCE.

This repo holds the scripts that produce the current, paper-reported results, in
the order they run. Heavy data (POD5, BAM, `features.h5`, checkpoints) lives on
scratch, not here.

---

## Layout

| Path | What it is |
|---|---|
| `deepmod/` | Core library. `featurization.py` builds pileup tensors; `model.py` holds `PileupDataset`, position-grouped splits, and per-position evaluation. `lodo.py`/`visualization.py` are imported by `model.py` and must stay. |
| `pipeline/` | Raw signal → ground truth → `features.h5` (ground-truth extraction, motif-based GT, background/negative-site generation). |
| `experiments/pipeline1/` | Shared training/eval library (`run_pipeline.py`: `train_one_model`, `evaluate`, `compute_metrics`), the model architecture (`run_convformer_v2.py`), and chemistry typing (`mod_types.py`). Imported by every later pipeline — not run standalone. |
| `experiments/pipeline2/` | Earlier all-genome mixed / LODO / LOMO study. Superseded by pipeline4 (see "Known issues" — pipeline2's both-strand pooling had a reference-row framing bug that pipeline4 sidesteps structurally). Kept for history. |
| `experiments/pipeline3/` | De novo motif re-discovery / type clustering from a scored genome; `score_genome.py` loads any saved checkpoint and scores an arbitrary `features.h5` — the tool for scoring genuinely new data (not one of the built-in test folds). |
| `experiments/pipeline4/` | **Current, paper-reported pipeline.** Matched-pool, causal-label, leave-one-chemistry-out (LOCO) + leave-one-organism-group-out (LOGO) training. `run_matched_loco.py`/`.sh` is the training+testing entry point; `refeaturize_strand15.py`/`refeaturize_benchmark.py`/`featurize_background.py` are the featurization entry points. See below. |
| `analysis/` | Figure generation and diagnostics, including `analysis/orca_remake/recompute_bench_types.py` (recovers real modification-chemistry labels for the benchmark organisms, for embedding/analysis figures — not required for training or testing). |

## Paths

Scripts reference absolute scratch paths, since that is where the data lives:

```
/fs/cbcb-lab/storm/bds062/data/benchmark/                       # POD5 + refs, 7 benchmark organisms
/fs/cbcb-scratch/bds062/data/human/{hg001,hg002}/pod5/           # POD5, human
/fs/cbcb-scratch/bds062/data/gt/                                 # ground-truth BEDs (all organisms)
/fs/cbcb-scratch/bds062/results/benchmark_results/                     # reads_refined.bam/peaks_refined.tsv (HP26695, benchmark orgs)
/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4/features/        # strand15 features.h5 (ONT/SPO1/HP + benchmark + background)
/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/                   # run_matched_loco.py output: models/, metrics/
```

---

## Reproducing the paper models, end to end

### 1. Ground truth

Independent of the nanopore signal being modelled — motif-derived (Dam/Dcm/etc.,
REBASE-characterized), bisulfite/EM-seq (arabidopsis, hg001, hg002), or
pre-computed bedMethyl:

```bash
python pipeline/motif_gt.py --ref REF.fa.gz --preset ecoli_dam --outdir data/gt/Ecoli_DM
python pipeline/extract_gt_bismark.py ...     # EM-seq / WGBS (arabidopsis, hg001, hg002)
python pipeline/extract_gt_from_pileup.py ... # pre-computed bedMethyl
```

`motif_gt.py` writes `gt_modified.bed` (positives) and a `candidate.bed`
identical to it, which makes a motif-derived dataset 100% positive on its own —
see step 3 for how the matched-pool design and the background-site step both
address this.

For each of the 6 motif-saturated bacterial benchmark organisms, also generate
genuine non-motif negative sites (used only as test negatives, for
`logo_bacteria` — see step 4):

```bash
python pipeline/generate_background_sites.py
```

### 2. Basecall → refine (once, reused by every featurization pass below)

```bash
bash pipeline/submit_all.sh          # per-dataset table; calls pipeline.sh
```

Runs Dorado basecalling + Remora move refinement, producing
`reads_refined.bam` / `peaks_refined.tsv` per dataset. This is a one-time cost;
every featurization script below reuses these outputs and only varies
featurization flags (window, strand, read cap).

### 3. Featurize — the matched pool, the curriculum organisms, and background negatives

Three scripts, one per data source, all producing `(16, 210, 9)` tensors — 1
reference row + 15 read rows (forward-strand-only pileups; single-strand by
design, see "Known issues"), 21 window positions × 10 samples, 9 channels
(`raw_signal, dwell_log1p, is_A, is_C, is_G, is_T, strand, mapq_norm, matches_ref`):

```bash
# ONT + SPO1/UMCES + HP26695 -- the matched pool (has a genuine unmodified
# twin at the same coordinate for every chemistry: synthetic control /
# amplicon-stripped / whole-genome-amplified). 13 dataset files.
python experiments/pipeline4/refeaturize_strand15.py --dry-run   # inspect
python experiments/pipeline4/refeaturize_strand15.py

# 7 benchmark organisms (Anabaena, 2x Ecoli, Tdenticola, HPJ99, arabidopsis)
# + hg001/hg002 -- single-sample WT/native, no matched twin, used only as
# always-on curriculum data (never in the core LOCO test sets).
python experiments/pipeline4/refeaturize_benchmark.py --dry-run
python experiments/pipeline4/refeaturize_benchmark.py

# Non-motif background negatives for the 6 bacterial organisms (from step 1),
# test-only, for logo_bacteria.
python experiments/pipeline4/featurize_background.py --dry-run
python experiments/pipeline4/featurize_background.py
```

hg001/hg002 need `--sample-n-sites 300000` (already set in
`refeaturize_benchmark.py`'s `DATASETS` table) and `--mem=192G` — their
candidate pools are large (hg001: 1.73M candidates) and 96G OOMs.

Memory in general: the output tensor is `n × 16 × 210 × 9` float32; large
uncapped datasets (HP26695 WGA, arabidopsis) need `--mem=128-192G`.

`rechunk_features.py` (used by the older pipeline/pipeline2 path) re-chunks
`tensors` to 1 image/chunk so streaming (`PILEUP_PRELOAD=0`) is fast — 1,536
img/s vs 45 img/s at the default 64-image chunking. `refeaturize_strand15.py`
et al. write single-chunked output directly.

### 4. Train + test

**One script does both** — each fold job trains a model, then immediately
evaluates it on that fold's held-out test set and writes `metrics/<fold>.tsv`.
There is no separate "test-only" mode for the built-in folds; to score a saved
checkpoint against genuinely new/external data, use
`experiments/pipeline3/score_genome.py --checkpoint <path> ...` instead.

```bash
FOLDS="mixed loco_5hmU loco_4mC loco_6mA loco_5mC loco_5hmC logo_bacteria logo_plant logo_mammal" \
OUTDIR=/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/<your_results_dir> \
RAWMOD_DATA_GEN=strand15 \
EXTRA_ORGANISMS=1 \
INCLUDE_HUMAN=1 \
SUPCON_DIM=128 \
SUPCON_WEIGHT=1.0 \
SUPCON_TEMP=0.20 \
CURRICULUM=1 \
CURRICULUM_EPOCHS=15 \
SAD_DIM=32 \
SAD_WEIGHT=1.0 \
SAD_ETA=1.0 \
BCE_WEIGHT=1.0 \
bash experiments/pipeline4/run_matched_loco.sh
```

This is the **exact final recipe** (`results19/`, the paper models). Each fold
runs as its own SLURM job (`--gres=gpu:1 --mem=48G`, streaming from disk;
`~7h` per fold on an RTX A5000), submitted independently so they run in
parallel subject to cluster capacity. `TIME_LIMIT=HH:MM:SS` overrides the
default 12h wall-clock; `PARTITION=scavenger` and `GPU_TYPE=<model>` are also
available — see the script header. Add `--dry-run` to inspect the `sbatch`
commands without submitting.

**Folds:**
- `mixed` — position-grouped 85/15 split over the whole matched pool (in-distribution reference point).
- `loco_<CHEM>` (CHEM ∈ `5hmU,4mC,6mA,5mC,5hmC`) — leave-one-chemistry-out: trains on every chemistry except CHEM, tests zero-shot on CHEM. **Also excludes any curriculum organism that biologically carries CHEM under a different name** (see `BENCH_ORG_CHEMS` in `run_matched_loco.py`) — without this, 6mA/5mC/4mC leak back into training via the always-on benchmark-organism curriculum data even when nominally "held out." 5hmC/5hmU were never affected (no curriculum organism carries either).
- `logo_<group>` (group ∈ `bacteria,plant,mammal`) — leave-one-organism-group-out: holds out entire curriculum organism(s), never in training, scored zero-shot as their own test set.
- `subset_<c1>+<c2>[+c3]` — exploratory training-diversity sweep (2-4 of the 5 chemistries in training); see the module docstring in `run_matched_loco.py`. Not part of the paper models; the chemistry-leak fix above is **not** applied to these (a single subset model is evaluated against multiple held-out targets at once, so one clean exclusion isn't well-defined). Treat its 6mA/5mC/4mC numbers with that caveat.

Metrics columns (`metrics/<fold>.tsv`, one row per fold): `micro_f1, mod_f1,
unmod_f1, macro_f1, mod_prec, mod_rec, auprc, auroc, auroc_sad, threshold,
n_pos, n_test`. `auroc` is threshold-free and the fairest cross-fold
comparison; `mod_f1`/`macro_f1` are computed at a per-fold F1-optimal
threshold fit on that fold's own test labels (`run_pipeline.optimal_threshold`)
— informative as a best-case operating point, but not directly comparable
across folds with different test-set class balance the way `auroc` is.
`n_pos`/`n_test` are **position-level** counts (multiple pileup images at the
same `(file, contig, position)` are averaged into one row before scoring,
`deepmod/model.py:aggregate_by_position`), not raw image counts.

The final, paper-reported metrics are in `paper_results/results19/metrics/`
(see below).

---

## Analysis

```bash
python analysis/make_reverse_complement_plots.py --out-dir .../reverse_complement
python analysis/visualize_h5_pileup.py --h5 features.h5 --cartoon
python analysis/orca_remake/recompute_bench_types.py --npz <embeddings_allorg.npz>  # real chemistry labels for embedding figures
```

---

## Known issues

**Strand handling.** `pipeline2`'s both-strand pooling had a real defect: the
reference row's base identity could be in the opposite frame from its own
reads (built from `pos_ref_context`, which every read writes in its own
orientation — last write wins), so at 6mA sites the reference row read 50% A /
50% T though every 6mA is genuinely on an A, and `matches_ref` was
meaningless wherever the reference row happened to land on the "wrong" base.
**`pipeline4` (the current pipeline) sidesteps this structurally** by
featurizing forward-strand-only (`--strand +`, `refeaturize_strand15.py` /
`refeaturize_benchmark.py`) — a pileup is single-strand by construction, so
the reference-row framing ambiguity above cannot occur. See
`analysis/make_reverse_complement_plots.py` for the original measurement (on
`pipeline2` data) if you need the historical detail.

**BENCH:: curriculum data can leak the "held-out" chemistry (fixed for
`loco_<CHEM>`, see `run_matched_loco.py`).** The 7 benchmark organisms +
hg001/hg002 are always unioned into stage-2 training (`EXTRA_ORGANISMS=1` /
`INCLUDE_HUMAN=1`), and `chem_array()` never assigns them a chemistry label —
so nothing in the original split logic prevented, say, `loco_6mA` from still
training on real 6mA-modified sites via Anabaena/Ecoli_DM/Ecoli_DM_MSssI/
Ecoli_WT/Tdenticola/HPJ99 (6 of 7 curriculum organisms are Dam-like 6mA under
a name the code never recognized as "6mA"). Quantified once: unfixed, BENCH::
leaked 131,660 6mA / 100,138 5mC / 4,807 4mC images into every fold's
training — 3×-17× more than the entire core-pool census of the chemistry
supposedly being held out. `BENCH_ORG_CHEMS` now records each organism's real
chemistry content and `loco_<CHEM>` excludes any organism carrying CHEM from
`extra_idx`. **`subset_<...>` (the exploratory diversity sweep) is not
fixed** — read its 6mA/5mC/4mC numbers with this caveat in mind.
