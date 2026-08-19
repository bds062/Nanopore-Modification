#!/usr/bin/env python3
"""
rawmod_matched_loco — matched-only, causal-label, leave-one-chemistry-out (LOCO)
training with a paired-contrastive (SupCon) objective.

WHY THIS EXPERIMENT
-------------------
Every earlier pipeline mixed motif-derived labels into training, where
P(mod | context) approx 0.94 (e.g. Dam methylates ~100% of GATC), so sequence
predicts the label and the model memorises the recognition motif — firing on
UNMODIFIED DNA (deepmod_full_pipeline2/datasets.md sec 5a). Post-hoc fixes
(flank masking, ch9 fix, DANN, SupCon-on-mixed) all failed to remove it.

This experiment removes the shortcut *at the source* by training ONLY on samples
that have a matched UNMODIFIED counterpart, so every label is causal:

  ONT    : {5mC,5hmC,6mA}.h5 positives  vs  control.h5 negatives   (synthetic control)
  SPO1   : bc06/07 native positives     vs  bc01-05 amplicon negs  (amplification strips mods)
  HP26695: WT native positives          vs  WGA negatives          (whole-genome amplified)

Within a matched sample the SAME genomic context appears as a modified positive
AND an unmodified negative, so P(mod | context) = 0.5 by construction and the
motif shortcut is structurally unavailable — the amplification analogue of a
Dam-knockout (datasets.md sec 6).

CHEMISTRY TYPING (per modified image)
  ONT   -> the file's modification name (5mC / 5hmC / 6mA)
  SPO1  -> modkit dominant-code map (mod_types.build_umces_mod_map):
           forward-T -> 5hmU (precedence), else m/h/a -> 5mC/5hmC/6mA
  HP WT -> centre reference base: A/T -> 6mA, C/G -> 4mC (both strands)

Chemistries present in the core matched pool: 5hmU, 5mC, 5hmC, 6mA, 4mC.

CURRICULUM DATA (EXTRA_ORGANISMS=1 / INCLUDE_HUMAN=1, BENCH:: members)
7 single-sample WT/native benchmark organisms (Kulkarni et al. 2024) + hg001/
hg002 -- no matched unmodified twin, so they never enter curriculum stage 1
(paired anchors), but ARE unioned into every fold's stage-2 training by
default (see fit()'s `extra` param). chem_array() never assigns these images a
chemistry label (they sit at chem=''), and several BENCH:: organisms carry the
SAME chemistry as a core-pool target under a DIFFERENT name -- unaddressed,
this leaks that "held-out" chemistry straight into training. Measured: with
both flags on, BENCH:: leaks 131,660 6mA / 100,138 5mC / 4,807 4mC images into
every fold regardless of which core chemistry is held out, versus core-pool
censuses of only 40,452 / 5,870 / 4,780 for those three -- the leak outweighs
the intended holdout by 3x-17x. 5hmC and 5hmU are never present in BENCH::
and are unaffected.
FIX: BENCH_ORG_CHEMS records each organism's real (REBASE/motif-characterized)
chemistry content; loco_<CHEM> strips any BENCH:: organism whose set contains
the held-out CHEM out of `extra_idx` before training, so "held out" means
never-seen-anywhere, not just absent from the core pool. This fix is applied
for loco_<CHEM> only -- subset_<...> (below) evaluates one trained model
against MULTIPLE held-out targets at once, so a single clean exclusion isn't
well-defined there; read subset_ chemistry-leak numbers for 6mA/5mC/4mC with
that caveat.

FOLDS (one SLURM job each)
  loco_<CHEM>  leave-one-chemistry-out (CHEM in 5hmU/4mC/6mA/5mC/5hmC):
     train = {positives typed != CHEM}  U  {85% of controls, position-grouped}
              U  {BENCH:: curriculum data, minus organisms carrying CHEM}
     test  = {positives typed == CHEM}  U  {15% controls, from the organism(s)
              carrying CHEM AND whose centre ref base matches CHEM's target
              base(s)} -- a pure signal contrast (e.g. modified-T vs unmodified-T
              for 5hmU), not a trivial base-composition split.
     This is the zero-shot "modification-agnostic" test: the model is scored on a
     chemistry it never saw ANYWHERE in training, using causal negatives from
     the same sample. These 5 folds, together with mixed and the 3
     logo_<group> folds, form the primary evaluation for this pipeline.
  logo_<group>  leave-one-organism-group-out (group in bacteria/plant/mammal,
     see LOGO_GROUPS): holds out entire BENCH:: organism(s) -- never in stage-2
     training, scored zero-shot as their own test set. logo_bacteria adds
     BGCTRL:: (non-motif background negatives) as test-only negatives, since
     the 6 bacterial BENCH:: datasets are 100% positive on their own.
  subset_<c1>+<c2>[+c3]  training-diversity sweep (2-4 of the 5 CHEMS in
     training; see parse_subset_fold): a single model trained on exactly this
     chemistry subset is zero-shot-evaluated against every chemistry NOT in
     it, so C(5,2)+C(5,3)+C(5,4)=25 unique subsets cover the full "AUROC vs
     #training-chemistries" sweep -- the 5 size-4 subsets are exactly the
     loco_<CHEM> folds above, reused rather than retrained. Exploratory, and
     NOT covered by the leak fix -- see caveat above.
  mixed        position-grouped 85/15 split over the whole matched pool
               (in-distribution reference point).

MODEL: ConvFormerV2(supcon_dim=SUPCON_DIM, default 128) trained by
run_pipeline.train_one_model with total loss BCE + SUPCON_WEIGHT*SupCon on the
causal labels. All model/train/eval code is imported from the repo; this file
only assembles the matched pool and defines the LOCO splits.

CURRENT RECIPE: RAWMOD_DATA_GEN=strand15 EXTRA_ORGANISMS=1 INCLUDE_HUMAN=1
SUPCON_DIM=128 SUPCON_WEIGHT=1.0 SUPCON_TEMP=0.20 CURRICULUM=1
CURRICULUM_EPOCHS=15 SAD_DIM=32 SAD_WEIGHT=1.0 SAD_ETA=1.0 BCE_WEIGHT=1.0 --
see run_matched_loco.sh and the repo README for the full launch command.

Usage:
  python run_matched_loco.py \
      --fold {mixed|loco_5hmU|loco_4mC|loco_6mA|loco_5mC|loco_5hmC|
              logo_bacteria|logo_plant|logo_mammal|subset_<c1>+<c2>[+c3]} \
      --out-dir <dir> [--epochs N]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Resolve imports against THIS repo (never the scratch working copies): the model
# and training code must be the committed versions. Only data lives on scratch.
REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / 'scripts' / 'train', REPO / 'rawmod'):
    sys.path.insert(0, str(_p))

import run_pipeline as R                          # noqa: E402
from run_convformer_v2 import ConvFormerV2         # noqa: E402
from model import split_position_groups            # noqa: E402
from mod_types import build_umces_mod_map          # noqa: E402

# ── W&B (mirrors pipeline2) ────────────────────────────────────────────────────
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', 'bds062-university-of-maryland')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'rawmod')

CHEMS = ('5hmU', '4mC', '6mA', '5mC', '5hmC')

# Reference base(s) at the candidate centre that carry each chemistry, forward
# strand. 5hmU replaces T (mod_map marks forward-T only). 6mA is A on either
# strand -> forward A(+)/T(-). 4mC/5mC/5hmC are C on either strand -> C(+)/G(-).
CHEM_BASES = {
    '5hmU': (b'T',),
    '6mA':  (b'A', b'T'),
    '4mC':  (b'C', b'G'),
    '5mC':  (b'C', b'G'),
    '5hmC': (b'C', b'G'),
}
# Organisms (member-name prefixes) that carry each chemistry — used to draw
# organism-matched test negatives.
CHEM_ORGS = {
    '5hmU': ('SPO1::',),
    '4mC':  ('HP::',),
    '6mA':  ('ONT::', 'HP::', 'SPO1::'),
    '5mC':  ('ONT::', 'SPO1::'),
    '5hmC': ('ONT::', 'SPO1::'),
}

# Cap negatives per organism so HP WGA (500k) cannot dominate; pos_weight in
# train_one_model handles the residual imbalance. Positives are never capped.
NEG_CAP = {'ONT::': 30000, 'SPO1::': 40000, 'HP::': 40000}

# Real (biological, REBASE/motif-characterized) chemistry content of each
# BENCH:: organism -- NOT the same as chem_array()'s per-image typing, which
# only labels ONT/SPO1/HP images and leaves every BENCH:: image at chem=''.
# That blind spot let BENCH:: positives leak the "held-out" chemistry straight
# into loco_<CHEM> training via fit()'s always-included bench_idx: e.g.
# loco_6mA nominally excludes all 6mA from the core pool, but 6 of 7 BENCH::
# organisms are Dam-like 6mA and were still unioned into stage-2 training
# regardless. Measured: with EXTRA_ORGANISMS=1+INCLUDE_HUMAN=1, BENCH:: leaks
# 131,660 6mA / 100,138 5mC / 4,807 4mC images into every fold's training --
# vs. core-pool censuses of only 40,452 / 5,870 / 4,780 for those chemistries
# respectively (i.e. the
# leaked signal outweighs what was supposedly held out, 3x-17x over). 5hmC and
# 5hmU are never present in BENCH:: -- those two folds were never affected.
# Used by loco_<CHEM> to strip chemistry-matching BENCH:: organisms out of
# extra_idx, mirroring how logo_<group> already excludes the held-out group.
BENCH_ORG_CHEMS = {
    'Anabaena_WT_5kHz':    {'6mA'},
    'Ecoli_DM_5kHz':       {'6mA'},
    'Ecoli_DM_MSssI_5kHz': {'6mA', '5mC'},
    'Ecoli_WT_5kHz':       {'6mA', '5mC'},
    'Tdenticola_WT_5kHz':  {'6mA'},
    'HPJ99_WT_5kHz':       {'6mA', '4mC'},
    'arabidopsis':         {'5mC'},
    'hg001':               {'5mC'},
    'hg002':               {'5mC'},
}

HP_WT  = '/fs/cbcb-scratch/bds062/results/benchmark_results/HP26695_WT_5kHz/features.h5'
HP_WGA = '/fs/cbcb-scratch/bds062/results/benchmark_results/HP26695_WGA_5kHz/features.h5'

# Strand-split, 15-read revamp (rawmod_full_pipeline4/refeaturize_strand15.py):
# forward-strand-only pileups, height 16 (15 reads + ref row) instead of 31,
# also unbiased site/base sampling. See memory: organism-identifiability-root
# -cause (strand-pooling was found to be a major dataset/organism batch-effect
# fingerprint) and results7-8-dann-backfire. Toggle with RAWMOD_DATA_GEN=strand15.
_P4 = '/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4/features'
HP_WT_V2  = f'{_P4}/HP26695_WT_5kHz/features.h5'
HP_WGA_V2 = f'{_P4}/HP26695_WGA_5kHz/features.h5'
ONT_FILES_V2 = {
    '5mC':     f'{_P4}/ONT/5mC.h5',
    '5hmC':    f'{_P4}/ONT/5hmC.h5',
    '6mA':     f'{_P4}/ONT/6mA.h5',
    'control': f'{_P4}/ONT/control.h5',
}
UMCES_FILES_V2 = {
    'bc06': f'{_P4}/deepmod_ont+umces/barcode06.h5',
    'bc07': f'{_P4}/deepmod_ont+umces/barcode07.h5',
    'bc02': f'{_P4}/deepmod_umces/train/barcode02.h5',
    'bc03': f'{_P4}/deepmod_umces/train/barcode03.h5',
    'bc04': f'{_P4}/deepmod_umces/train/barcode04.h5',
    'bc05': f'{_P4}/deepmod_umces/train/barcode05.h5',
    'bc01': f'{_P4}/deepmod_umces/test/barcode01_test.h5',
}
USE_STRAND15 = os.environ.get('RAWMOD_DATA_GEN', '') == 'strand15'
HEIGHT = 16 if USE_STRAND15 else 31   # 1 ref row + (15 or 30) reads

# Extra-organism curriculum (EXTRA_ORGANISMS=1): 7 ONT-basemod-benchmark
# (Kulkarni et al. 2024) datasets genuinely novel relative to the matched pool
# above (HP26695 WT/WGA is EXCLUDED here -- it's already the source of the
# HP:: data above, would be pure duplication). Single-strand featurization
# (--strand + --min-mapq 0), same height=16 convention as strand15 -- requires
# USE_STRAND15. These are single-sample WT/native strains (no matched
# unmodified twin at the same coordinate the way ONT/SPO1/HP are designed), so
# they contribute ZERO curriculum stage-1 anchors by construction and are only
# added to stage-2 (full-fold) training -- see `fit()`'s `bench_idx` union.
USE_EXTRA_ORGS = os.environ.get('EXTRA_ORGANISMS', '0') == '1'
_BENCH_ROOT = f'{_P4}/benchmark'
BENCH_FILES = {
    'Anabaena_WT_5kHz':    f'{_BENCH_ROOT}/Anabaena_WT_5kHz/features.h5',
    'Ecoli_DM_5kHz':       f'{_BENCH_ROOT}/Ecoli_DM_5kHz/features.h5',
    'Ecoli_DM_MSssI_5kHz': f'{_BENCH_ROOT}/Ecoli_DM_MSssI_5kHz/features.h5',
    'Ecoli_WT_5kHz':       f'{_BENCH_ROOT}/Ecoli_WT_5kHz/features.h5',
    'Tdenticola_WT_5kHz':  f'{_BENCH_ROOT}/Tdenticola_WT_5kHz/features.h5',
    'HPJ99_WT_5kHz':       f'{_BENCH_ROOT}/HPJ99_WT_5kHz/features.h5',
    'arabidopsis':         f'{_BENCH_ROOT}/arabidopsis/features.h5',
}

# Human data (hg001/hg002) is gated by its OWN flag, separate from
# EXTRA_ORGANISMS -- so EXTRA_ORGANISMS=1 keeps its documented meaning (the 7
# bacterial/plant benchmark organisms) for results14/results15/temp-sweep
# reproducibility, and human data can be toggled independently. Very
# different scale from the bacterial sets: hg001 has 2,858 images (79
# pos/2,779 neg), hg002 only 98 (71 pos/27 neg) -- both real bisulfite/EM-seq
# GT (not motif), unlike 6 of the 7 bacterial sets.
USE_HUMAN = os.environ.get('INCLUDE_HUMAN', '0') == '1'
HUMAN_FILES = {
    'hg001': f'{_BENCH_ROOT}/hg001/features.h5',
    'hg002': f'{_BENCH_ROOT}/hg002/features.h5',
}

# LOGO (leave-one-group-out, organism/dataset-level holdout): groups of BENCH::
# organisms held out ENTIRELY from training (never in stage-2, unlike the
# always-included bench_idx used by loco_<CHEM>/mixed) and scored zero-shot as
# their own test set. See logo_<group> branch in main().
LOGO_GROUPS = {
    'bacteria': ['Anabaena_WT_5kHz', 'Ecoli_DM_5kHz', 'Ecoli_DM_MSssI_5kHz',
                'Ecoli_WT_5kHz', 'Tdenticola_WT_5kHz', 'HPJ99_WT_5kHz'],
    'plant':    ['arabidopsis'],
    'mammal':   ['hg001', 'hg002'],
}

# The 6 bacterial BENCH:: datasets are 100% positive (no negatives -- see
# insights.md), so logo_bacteria's test set would have an undefined AUROC
# (single class) without help. BGCTRL:: members are non-motif background
# positions (pipeline/generate_background_sites.py + featurize_background.py)
# -- genuine unmodified-context negatives, same base chemistry, just outside
# the recognition motif. Used ONLY as extra test negatives for logo_bacteria;
# never added to any training set (see is_bgctrl handling in main()).
_BGCTRL_ROOT = f'{_P4}/benchmark'
BGCTRL_FILES = {
    'Anabaena_WT_5kHz':    f'{_BGCTRL_ROOT}/Anabaena_WT_5kHz_background/features.h5',
    'Ecoli_DM_5kHz':       f'{_BGCTRL_ROOT}/Ecoli_DM_5kHz_background/features.h5',
    'Ecoli_DM_MSssI_5kHz': f'{_BGCTRL_ROOT}/Ecoli_DM_MSssI_5kHz_background/features.h5',
    'Ecoli_WT_5kHz':       f'{_BGCTRL_ROOT}/Ecoli_WT_5kHz_background/features.h5',
    'Tdenticola_WT_5kHz':  f'{_BGCTRL_ROOT}/Tdenticola_WT_5kHz_background/features.h5',
    'HPJ99_WT_5kHz':       f'{_BGCTRL_ROOT}/HPJ99_WT_5kHz_background/features.h5',
}


def build_members():
    """name -> h5 path for the matched-only pool (prefixes encode the organism)."""
    ont_files = ONT_FILES_V2 if USE_STRAND15 else R.ONT_FILES
    umces_files = UMCES_FILES_V2 if USE_STRAND15 else R.UMCES_FILES
    m = {}
    for mod in R.ONT_ORDER:                 # 5mC,5hmC,6mA,control
        m[f'ONT::{mod}'] = ont_files[mod]
    for bc in R.UMCES_ORDER:                # bc06,bc07 (pos) + bc01-05 (neg)
        m[f'SPO1::{bc}'] = umces_files[bc]
    m['HP::WT'] = HP_WT_V2 if USE_STRAND15 else HP_WT
    m['HP::WGA'] = HP_WGA_V2 if USE_STRAND15 else HP_WGA
    if USE_EXTRA_ORGS:
        assert USE_STRAND15, "EXTRA_ORGANISMS=1 requires RAWMOD_DATA_GEN=strand15 (height must match)"
        for name, path in BENCH_FILES.items():
            m[f'BENCH::{name}'] = path
    if USE_HUMAN:
        assert USE_STRAND15, "INCLUDE_HUMAN=1 requires RAWMOD_DATA_GEN=strand15 (height must match)"
        for name, path in HUMAN_FILES.items():
            m[f'BENCH::{name}'] = path
    if USE_EXTRA_ORGS:
        for name, path in BGCTRL_FILES.items():
            m[f'BGCTRL::{name}'] = path
    return m


def org_of(name):
    return name.split('::')[0] + '::'


def ref_base_center(group):
    """Centre reference base (bytes 'A'/'C'/'G'/'T') for every image, read from
    each file's reference row at the true window centre (half_window*L)."""
    out = np.empty(group.N, dtype='S1')
    bases = np.array([b'A', b'C', b'G', b'T'])
    offsets = np.concatenate([[0], np.cumsum(group.file_sizes)])
    for fi, path in enumerate(group.paths):
        lo, hi = int(offsets[fi]), int(offsets[fi + 1])
        with h5py.File(path, 'r') as hf:
            L = int(hf.attrs['L']); cs = int(hf.attrs['half_window']) * L
            oh = hf['tensors'][:, 0, cs, 2:6]        # (n,4)
        out[lo:hi] = bases[np.argmax(oh, axis=1)]
    return out


def chem_array(group, mod_map, refbase):
    """Per-image chemistry string ('' for unmodified/untyped)."""
    chem = np.array([''] * group.N, dtype=object)
    modified = group.labels > 0
    for i in np.nonzero(modified)[0]:
        nm = group.names[int(group.file_of[i])]
        if nm.startswith('ONT::'):
            chem[i] = nm.split('::')[1]              # control has no positives
        elif nm.startswith('SPO1::'):
            key = (group.contig[i], int(group.ref_pos[i]))
            t = mod_map.get(key)
            if t is None:                            # fallback by ref base
                t = '5hmU' if refbase[i] == b'T' else 'untyped'
            chem[i] = t
        elif nm == 'HP::WT':
            b = refbase[i]
            chem[i] = ('6mA' if b in (b'A', b'T')
                       else '4mC' if b in (b'C', b'G') else 'untyped')
    return chem


def pos_hash_split(group, idx, test_frac=0.15, seed=0):
    """Deterministic position-grouped train/test split of image indices `idx`
    by hashing (contig, ref_pos): all images at one coordinate stay together.

    Uses zlib.crc32 (NOT Python's builtin hash(), which is per-process salted by
    PYTHONHASHSEED) so the split is identical across the separate SLURM job
    processes and reproducible run-to-run."""
    import zlib
    idx = np.asarray(idx, dtype=np.int64)
    tr, te = [], []
    for i in idx:
        key = f"{group.contig[i]}:{int(group.ref_pos[i])}:{seed}".encode()
        h = zlib.crc32(key) & 0xffffffff
        (te if (h / 0xffffffff) < test_frac else tr).append(int(i))
    return np.array(tr, dtype=np.int64), np.array(te, dtype=np.int64)


def subsample_negatives(group, seed=0):
    """Cap control images per organism (NEG_CAP). Returns the kept control idx."""
    rng = np.random.default_rng(seed)
    ctrl = np.nonzero(group.labels <= 0)[0]
    keep = []
    for pref, cap in NEG_CAP.items():
        sel = ctrl[np.array([group.names[int(group.file_of[i])].startswith(pref)
                             for i in ctrl])]
        if len(sel) > cap:
            sel = rng.choice(sel, cap, replace=False)
        keep.append(sel)
    return np.sort(np.concatenate(keep))


def parse_subset_fold(fold):
    """'subset_<chem>+<chem>[+...]' -> sorted list of 2-4 distinct CHEMS, or None
    if `fold` isn't a valid subset fold name. Powers the training-diversity sweep:
    train on exactly this set of chemistries (+ the always-included BENCH::/human
    curriculum data), then zero-shot-evaluate on every chemistry NOT in the set --
    one trained model answers the sweep for all of its held-out chemistries at
    once, so the 2-4 sweep only needs C(5,2)+C(5,3)+C(5,4) = 25 unique trainings
    (5 of which are exactly the existing loco_<CHEM> models, size-4 subsets)
    rather than retraining per (held-out chem, other-chem-subset) pair."""
    if not fold.startswith('subset_'):
        return None
    chems = fold[len('subset_'):].split('+')
    if not (2 <= len(chems) <= 4) or len(set(chems)) != len(chems) or not all(c in CHEMS for c in chems):
        return None
    return sorted(chems)


def mixed_split(pool, is_pos, neg_mask, hp):
    """Deterministic 85/15 position-grouped split of the whole matched pool (all 5
    chemistries + capped controls) -- the 'mixed' in-distribution fold. Factored out
    so a downstream analysis (e.g. a post-hoc embedding probe) can recompute the
    EXACT same train/test image indices used to train the mixed checkpoint, without
    re-deriving the split logic. Deterministic given (pool, hp.seed)."""
    keep = np.nonzero(is_pos | neg_mask)[0]
    tr, _, te, stats = split_position_groups(
        pool.labels[keep], [pool.position_keys[i] for i in keep],
        val_frac=0.0, test_frac=0.15, seed=hp.seed)
    train_idx, test_idx = keep[tr], keep[te]
    R.assert_disjoint(train_idx, test_idx, pool, 'mixed')
    return train_idx, test_idx, stats


def anchor_idx_within(pool, train_idx, is_pos):
    """Image indices at POSITION-PAIRED anchors within train_idx: coordinates
    (contig,pos) that carry BOTH a modified and an unmodified image — the same
    genomic context under both labels, the purest causal contrast. These are the
    curriculum stage-1 examples ("learn modified vs its own unmodified twin")."""
    pos_positions, neg_positions = set(), set()
    for i in train_idx:
        key = (pool.contig[i], int(pool.ref_pos[i]))
        (pos_positions if is_pos[i] else neg_positions).add(key)
    anchors = pos_positions & neg_positions
    if not anchors:
        return np.zeros(0, dtype=np.int64)
    sel = [int(i) for i in train_idx
           if (pool.contig[i], int(pool.ref_pos[i])) in anchors]
    return np.array(sel, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', required=True,
                    help="'mixed' or 'loco_<CHEM>' with CHEM in " + '/'.join(CHEMS))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--epochs', type=int, default=None)
    a = ap.parse_args()

    valid = ['mixed', 'all'] + [f'loco_{c}' for c in CHEMS] + [f'logo_{g}' for g in LOGO_GROUPS]
    subset_chems = parse_subset_fold(a.fold)
    if a.fold not in valid and subset_chems is None:
        raise SystemExit(f"--fold must be one of {valid}, or subset_<chem>+<chem>[+...] "
                         f"(2-4 distinct chems from {CHEMS}), got {a.fold!r}")

    hp = R.HP()
    if a.epochs:
        hp.epochs = a.epochs
    R.set_seed(hp.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out_dir)
    (out / 'models').mkdir(parents=True, exist_ok=True)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    print(f"Device {device}  fold={a.fold}  out={out}", flush=True)

    members = build_members()
    names = list(members)
    pool = R.Group(names, members)
    print(f"Matched pool: {pool.N:,} images across {len(names)} files", flush=True)

    mod_map = build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = ref_base_center(pool)
    chem = chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    kept_neg = subsample_negatives(pool, seed=hp.seed)
    neg_mask = np.zeros(pool.N, dtype=bool); neg_mask[kept_neg] = True

    # Extra-organism curriculum data (BENCH:: prefixed members): stage-2-only,
    # every fold, never in any test set. See build_members()/USE_EXTRA_ORGS.
    is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::')
                         for i in range(pool.N)])
    bench_idx = np.nonzero(is_bench)[0].astype(np.int64)
    if len(bench_idx):
        n_bench_sets = sum(1 for nm in names if nm.startswith('BENCH::'))
        print(f"  extra organisms (BENCH::, stage-2 training only): "
              f"{len(bench_idx):,} images across {n_bench_sets} datasets "
              f"(pos={int(is_pos[bench_idx].sum()):,} "
              f"neg={int((~is_pos[bench_idx]).sum()):,})", flush=True)

    # Per-image BENCH:: organism name (for logo_<group> filtering), '' elsewhere.
    bench_org_of = np.array([
        pool.names[int(pool.file_of[i])].split('::')[1] if is_bench[i] else ''
        for i in range(pool.N)])

    # BGCTRL:: background-control images (non-motif negatives for the 6
    # bacterial datasets) -- ONLY used as extra test negatives for
    # logo_bacteria, NEVER for training. See BGCTRL_FILES docstring.
    is_bgctrl = np.array([pool.names[int(pool.file_of[i])].startswith('BGCTRL::')
                          for i in range(pool.N)])
    if is_bgctrl.any():
        print(f"  background-control (BGCTRL::, logo_bacteria test-only): "
              f"{int(is_bgctrl.sum()):,} images, all label=0", flush=True)

    # census
    from collections import Counter
    print("  positives per chemistry:",
          {k: int(v) for k, v in Counter(chem[is_pos]).items()}, flush=True)
    print(f"  controls kept (capped): {int(neg_mask.sum()):,} "
          f"of {int((~is_pos).sum()):,}", flush=True)

    # ── W&B ────────────────────────────────────────────────────────────────────
    wandb_run = None
    if os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'):
        os.environ.setdefault('WANDB_MODE', 'online')
        os.environ.setdefault('WANDB_DIR', str(Path(__file__).resolve().parent))
        try:
            import wandb, secrets
            rid = secrets.token_hex(4)
            wandb_run = wandb.init(
                entity=WANDB_ENTITY, project=WANDB_PROJECT,
                name=f"matched_loco-{rid}-{a.fold}", group='matched_loco',
                job_type=a.fold.split('_')[0],
                config={'fold': a.fold, 'architecture': 'ConvFormerV2',
                        'supcon_dim': int(os.environ.get('SUPCON_DIM', '128')),
                        'supcon_weight': float(os.environ.get('SUPCON_WEIGHT', '0.1')),
                        **{k: getattr(hp, k) for k in dir(hp) if not k.startswith('_')}},
                settings=wandb.Settings(init_timeout=180, start_method='thread'))
            print(f"  [wandb] {WANDB_ENTITY}/{WANDB_PROJECT}/matched_loco-{rid}-{a.fold}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [wandb] disabled ({e})", file=sys.stderr)

    supcon_dim = int(os.environ.get('SUPCON_DIM', '128'))
    sad_dim = int(os.environ.get('SAD_DIM', '0'))
    if sad_dim > 0:
        print(f"  [DeepSAD] sad_dim={sad_dim} weight={os.environ.get('SAD_WEIGHT','1.0')} "
              f"eta={os.environ.get('SAD_ETA','1.0')}", flush=True)
    model_factory = lambda: ConvFormerV2(dropout=hp.dropout, supcon_dim=supcon_dim,
                                         sad_dim=sad_dim, h=HEIGHT)
    rows = []

    def sad_auroc(model, idx):
        """AUROC of the Deep-SAD anomaly score (||sad_head(rep) - centre||) vs the
        true labels on image indices idx. Higher distance = more anomalous."""
        from model import PileupDataset, make_loader_kwargs, _worker_init_fn
        from torch.utils.data import DataLoader
        from sklearn.metrics import roc_auc_score
        ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                           augment=False, seed=0, signal_noise_std=0.0,
                           delta_channels=True, preload=False,
                           mask_all_bases=os.environ.get('PILEUP_MASK_BASES', '0') == '1')
        loader = DataLoader(ds, shuffle=False,
                            **make_loader_kwargs(512, 6, device, _worker_init_fn))
        c = model.sad_center
        model.eval(); dists = []
        with torch.no_grad():
            for xb, _ in loader:
                model(xb.to(device, non_blocking=True))
                dists.append(((model._sad - c) ** 2).sum(1).sqrt().cpu().numpy())
        s = np.concatenate(dists)
        y = (pool.labels[np.asarray(idx, np.int64)] > 0).astype(int)
        return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')

    # Curriculum (CURRICULUM=1): stage 1 trains only on position-paired anchors
    # (same coordinate, modified vs its unmodified twin) to build a chemistry-
    # agnostic deviation-from-control representation; stage 2 warm-starts from it
    # and trains on the full fold set. Off -> single-stage (identical to results1).
    curriculum = os.environ.get('CURRICULUM', '0') == '1'
    cur_epochs = int(os.environ.get('CURRICULUM_EPOCHS', '15'))

    def fit(train_idx, fold_dir, runtag, extra_idx=None):
        mdir = out / 'models' / fold_dir
        # Stage-2 training set = this fold's train_idx UNION the extra-organism
        # (BENCH::) pool, if any. Stage-1 (anchors, below) deliberately uses the
        # UNEXPANDED train_idx: BENCH:: organisms have no matched unmodified twin
        # at the same coordinate (single WT/native samples, not a synthetic-pair
        # design), so they'd contribute zero anchors anyway -- this just makes
        # that explicit rather than relying on coordinate non-overlap.
        # extra_idx overrides the default (all bench_idx) -- used by logo_<group>
        # to union in only the NON-held-out extra-organism groups.
        extra = bench_idx if extra_idx is None else extra_idx
        stage2_idx = np.sort(np.concatenate([train_idx, extra])) if len(extra) else train_idx
        if curriculum:
            anc = anchor_idx_within(pool, train_idx, is_pos)
            npos = int(is_pos[anc].sum()) if len(anc) else 0
            print(f"  [curriculum] stage1 anchors={len(anc):,} (pos={npos:,} "
                  f"neg={len(anc)-npos:,})  epochs={cur_epochs}", flush=True)
            if len(anc) >= 500 and npos > 0 and (len(anc) - npos) > 0:
                hp1 = R.HP(); hp1.epochs = cur_epochs; hp1.patience = cur_epochs
                m1 = R.train_one_model(pool, anc, hp1, device, mdir, runtag + '_cur1',
                                       model_factory=model_factory)
                state = {k: v.detach().cpu().clone() for k, v in m1.state_dict().items()}
                del m1
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                return R.train_one_model(pool, stage2_idx, hp, device, mdir, runtag,
                                         model_factory=model_factory, init_state=state)
            print("  [curriculum] too few paired anchors; single-stage fallback",
                  flush=True)
        return R.train_one_model(pool, stage2_idx, hp, device, mdir, runtag,
                                 model_factory=model_factory)

    def record(model, test_name, idx, held=''):
        m = R.evaluate(model, pool, idx, device, hp)
        if getattr(model, 'sad_head', None) is not None:
            m['auroc_sad'] = sad_auroc(model, idx)     # anomaly-score AUROC
        rows.append({'fold': a.fold, 'test_set': test_name, 'held_out': held, **m})
        sad_msg = f" auroc_sad={m['auroc_sad']:.3f}" if 'auroc_sad' in m else ""
        print(f"  EVAL {test_name}: mod_f1={m['mod_f1']:.3f} mod_rec={m['mod_rec']:.3f} "
              f"mod_prec={m['mod_prec']:.3f} auprc={m['auprc']:.3f} "
              f"auroc={m['auroc']:.3f}{sad_msg} n_pos={m['n_pos']} n_test={m['n_test']}",
              flush=True)
        if wandb_run is not None:
            wandb_run.summary.update({f'eval/{test_name}/{k}': v for k, v in m.items()})

    if a.fold == 'all':
        # Single DEPLOYABLE model: train on the ENTIRE matched pool (all 5
        # chemistries + controls), no test holdout — train_one_model still carves
        # an internal val split for early stopping. This checkpoint is meant to be
        # scored against a DIFFERENT organism later (never in the matched pool),
        # e.g. experiments/pipeline3/score_genome.py --checkpoint .../all/all/best_model.pt
        keep = np.nonzero(is_pos | neg_mask)[0].astype(np.int64)
        print(f"  ALL matched data -> deployable model: {len(keep):,} images "
              f"(pos={int(is_pos[keep].sum()):,} neg={int((~is_pos[keep]).sum()):,})",
              flush=True)
        model = fit(keep, 'all', 'all')
        print("  no held-out test (deployment model); score externally with a "
              "different organism.", flush=True)

    elif a.fold == 'mixed':
        train_idx, test_idx, stats = mixed_split(pool, is_pos, neg_mask, hp)
        print(f"  {stats}", flush=True)
        model = fit(train_idx, a.fold, 'mixed')
        record(model, 'held_out_test', test_idx)

    elif a.fold.startswith('loco_'):
        chem_x = a.fold[len('loco_'):]
        # controls: position-grouped 85/15 over the capped control pool
        ctrl_idx = np.nonzero(neg_mask)[0]
        tr_ctrl, te_ctrl = pos_hash_split(pool, ctrl_idx, test_frac=0.15, seed=hp.seed)
        # test negatives: from CHEM's organism(s) AND ref-base-matched
        bases = CHEM_BASES[chem_x]; orgs = CHEM_ORGS[chem_x]
        te_ctrl = np.array([i for i in te_ctrl
                            if org_of(pool.names[int(pool.file_of[i])]) in orgs
                            and refbase[i] in bases], dtype=np.int64)
        pos_x = np.nonzero(is_pos & (chem == chem_x))[0].astype(np.int64)
        pos_other = np.nonzero(is_pos & (chem != chem_x) & (chem != '')
                               & (chem != 'untyped'))[0].astype(np.int64)

        train_idx = np.sort(np.concatenate([pos_other, tr_ctrl]))
        test_idx = np.sort(np.concatenate([pos_x, te_ctrl]))
        R.assert_disjoint(train_idx, test_idx, pool, a.fold)

        # BENCH:: leak fix (see BENCH_ORG_CHEMS): strip out any BENCH:: organism
        # that biologically carries chem_x before unioning the curriculum data
        # into stage-2 training, so a "held-out" chemistry is actually never
        # seen anywhere in training, not just absent from the core pool.
        leaky_orgs = {o for o, cs in BENCH_ORG_CHEMS.items() if chem_x in cs}
        clean_extra = np.nonzero(is_bench & ~np.isin(bench_org_of, list(leaky_orgs)))[0].astype(np.int64)
        n_excluded = int(bench_idx.size - clean_extra.size)
        print(f"  train={len(train_idx):,} (pos_other={len(pos_other):,} "
              f"neg={len(tr_ctrl):,})  test={len(test_idx):,} "
              f"(pos_{chem_x}={len(pos_x):,} neg={len(te_ctrl):,})", flush=True)
        if leaky_orgs:
            print(f"  BENCH:: leak fix: excluding {sorted(leaky_orgs)} "
                  f"({n_excluded:,} images that biologically carry {chem_x}) "
                  f"from stage-2 curriculum -- clean_extra={len(clean_extra):,} "
                  f"(of {len(bench_idx):,})", flush=True)
        if len(pos_x) == 0 or len(te_ctrl) == 0:
            raise SystemExit(f"empty test for {a.fold}: pos={len(pos_x)} neg={len(te_ctrl)}")
        model = fit(train_idx, a.fold, 'loco', extra_idx=clean_extra)
        record(model, f'zeroshot_{chem_x}', test_idx, held=chem_x)

    elif a.fold.startswith('subset_'):
        include_chems = subset_chems  # validated at top of main()
        held_chems = [c for c in CHEMS if c not in include_chems]
        print(f"  subset training chemistries: {include_chems}  "
              f"(zero-shot held out: {held_chems})", flush=True)

        ctrl_idx = np.nonzero(neg_mask)[0]
        tr_ctrl, te_ctrl_all = pos_hash_split(pool, ctrl_idx, test_frac=0.15, seed=hp.seed)
        pos_incl = np.nonzero(is_pos & np.isin(chem, include_chems))[0].astype(np.int64)
        train_idx = np.sort(np.concatenate([pos_incl, tr_ctrl]))
        print(f"  train={len(train_idx):,} (pos_incl={len(pos_incl):,} neg={len(tr_ctrl):,})",
              flush=True)
        model = fit(train_idx, a.fold, 'subset')

        for chem_x in held_chems:
            bases = CHEM_BASES[chem_x]; orgs = CHEM_ORGS[chem_x]
            te_ctrl = np.array([i for i in te_ctrl_all
                                if org_of(pool.names[int(pool.file_of[i])]) in orgs
                                and refbase[i] in bases], dtype=np.int64)
            pos_x = np.nonzero(is_pos & (chem == chem_x))[0].astype(np.int64)
            test_idx = np.sort(np.concatenate([pos_x, te_ctrl]))
            if len(pos_x) == 0 or len(te_ctrl) == 0:
                print(f"  WARNING: empty test for held-out {chem_x}: "
                      f"pos={len(pos_x)} neg={len(te_ctrl)} -- skipping", flush=True)
                continue
            R.assert_disjoint(train_idx, test_idx, pool, f'{a.fold}:{chem_x}')
            record(model, f'zeroshot_{chem_x}', test_idx, held=chem_x)

    elif a.fold.startswith('logo_'):  # leave-one-organism-group-out
        group_x = a.fold[len('logo_'):]
        held_orgs = LOGO_GROUPS[group_x]
        held_mask = is_bench & np.isin(bench_org_of, held_orgs)
        test_idx = np.nonzero(held_mask)[0].astype(np.int64)
        if group_x == 'bacteria':
            # 100% positive without help (see BGCTRL_FILES docstring) -- add the
            # non-motif background negatives, test-only, never trained on.
            bg_idx = np.nonzero(is_bgctrl)[0].astype(np.int64)
            test_idx = np.sort(np.concatenate([test_idx, bg_idx]))
            print(f"  logo_bacteria: +{len(bg_idx):,} BGCTRL:: background negatives "
                  f"added to test only", flush=True)
        extra_idx = np.nonzero(is_bench & ~held_mask)[0].astype(np.int64)
        core_idx = np.nonzero((is_pos | neg_mask) & ~is_bench & ~is_bgctrl)[0].astype(np.int64)
        R.assert_disjoint(core_idx, test_idx, pool, a.fold)
        R.assert_disjoint(extra_idx, test_idx, pool, a.fold)
        print(f"  train core={len(core_idx):,}  extra(other groups)={len(extra_idx):,}  "
              f"test({group_x})={len(test_idx):,} (pos={int(is_pos[test_idx].sum()):,} "
              f"neg={int((~is_pos[test_idx]).sum()):,})", flush=True)
        if len(test_idx) == 0:
            raise SystemExit(f"empty test for {a.fold}: no BENCH:: images for group {group_x}")
        if int(is_pos[test_idx].sum()) == 0 or int((~is_pos[test_idx]).sum()) == 0:
            print(f"  WARNING: {a.fold} test set is single-class "
                  f"(pos={int(is_pos[test_idx].sum())} neg={int((~is_pos[test_idx]).sum())}) "
                  f"-- AUROC will be NaN.", flush=True)
        model = fit(core_idx, a.fold, 'logo', extra_idx=extra_idx)
        record(model, f'zeroshot_{group_x}', test_idx, held=group_x)

    else:
        raise SystemExit(f"unhandled fold {a.fold!r}")  # unreachable: validated above

    cols = ['fold', 'test_set', 'held_out', 'micro_f1', 'mod_f1', 'unmod_f1',
            'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc', 'auroc_sad',
            'threshold', 'n_pos', 'n_test']
    tsv = out / 'metrics' / f'{a.fold}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(f"{r[c]:.6f}" if isinstance(r.get(c), float)
                               else str(r.get(c, '')) for c in cols) + '\n')
    print(f"\nWrote {tsv}\nDONE [{a.fold}]", flush=True)
    if wandb_run is not None:
        try:
            wandb_run.save(str(tsv)); wandb_run.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
