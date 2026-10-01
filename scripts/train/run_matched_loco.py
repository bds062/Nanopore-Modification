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
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from rawmod.paths import RAWMOD_DATA, RAWMOD_RESULTS, RAWMOD_STORE  # noqa: E402  (site paths; see paths.env.example)

import collections
import argparse
import json
import os
import time
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
WANDB_ENTITY = os.environ.get('WANDB_ENTITY') or None   # None = the logged-in wandb account
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'rawmod')

CHEMS = ('5hmU', '4mC', '6mA', '5mC', '5hmC')

# Fixed independent of --seed: every train/test split (mixed_split, pos_hash_split,
# subsample_negatives) must stay byte-identical across a --seed replicate run, so a
# different result reflects training stochasticity, not a different test set. Only
# R.set_seed(hp.seed) (weight init, dropout, data-loader shuffling) responds to --seed.
SPLIT_SEED = 42

# Reference base(s) at the candidate centre that carry each chemistry, forward
# strand. 5hmU replaces T (mod_map marks forward-T only). 6mA is A on either
# strand -> forward A(+)/T(-). 4mC/5mC/5hmC are C on either strand -> C(+)/G(-).
# With strand-resolved data the centre reference base of an image IS the
# modified base (minus-strand images are mirrored, so their reference row is
# complemented), so a chemistry matches exactly one base. The legacy
# forward-only data pooled both strands at + coordinates and therefore needed
# the complement as well.
CHEM_BASES = ({
    '5hmU': (b'T',), '6mA': (b'A',), '4mC': (b'C',),
    '5mC':  (b'C',), '5hmC': (b'C',),
} if os.environ.get('RAWMOD_DATA_GEN', '') == 'strandres' else {
    '5hmU': (b'T',),
    '6mA':  (b'A', b'T'),
    '4mC':  (b'C', b'G'),
    '5mC':  (b'C', b'G'),
    '5hmC': (b'C', b'G'),
})
# Organisms (member-name prefixes) that carry each chemistry — used to draw
# organism-matched test negatives.
CHEM_ORGS = {
    '5hmU': ('SPO1::',),
    '4mC':  ('HP::',),
    # HP:: is back in: it was dropped for strandres on the assumption that
    # H. pylori 26695 is 4mC-only, which the modkit pileups disprove (41,004
    # confident 6mA in WT vs 1 in WGA). Its 6mA positives are now typed, so the
    # fold needs HP's own negatives to keep the contrast organism-matched.
    '6mA':  ('ONT::', 'SPO1::', 'HP::'),
    '5mC':  ('ONT::', 'SPO1::'),
    '5hmC': ('ONT::', 'SPO1::'),
}

# Cap negatives per organism so HP WGA (500k) cannot dominate; pos_weight in
# train_one_model handles the residual imbalance. Positives are never capped.
NEG_CAP = {'ONT::': 30000, 'SPO1::': 40000, 'HP::': 40000}
# NEG_CAP_<ORG>=N raises/lowers one organism's control-image cap. The SPO1 cap
# bounds how many bc01-05 control positions exist to split at all, and so caps
# loco_5hmU's achievable test-negative count.
for _o in list(NEG_CAP):
    _v = os.environ.get('NEG_CAP_' + _o.rstrip(':'))
    if _v:
        NEG_CAP[_o] = int(_v)

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
# Measured content, from the per-modification modkit pileups (confident calls
# at >=80% modified, cov>=10; see build_modkit_chem_map.py). This drives which
# BENCH organisms loco_<CHEM> strips from training, so an organism listed here
# with CHEM is removed entirely for that fold -- "held out" means never seen
# anywhere. The previous values were REBASE preset guesses and understated
# several organisms: Anabaena and T. denticola both carry 4mC (16,625 and 7,261
# confident sites) yet were listed 6mA-only, so loco_4mC trained on them.
BENCH_ORG_CHEMS = {
    'Anabaena_WT_5kHz':    {'6mA', '4mC', '5mC'},   # 21,245 / 16,625 / 397
    'Ecoli_DM_5kHz':       set(),                   # dam-/dcm-: no confident marks
    'Ecoli_DM_MSssI_5kHz': {'5mC'},                 # M.SssI; dam- so no 6mA
    'Ecoli_WT_5kHz':       {'6mA', '5mC'},          # 38,041 / 23,285; no 4mC MTase
    'Tdenticola_WT_5kHz':  {'6mA', '4mC'},          # 25,476 / 7,261
    'HPJ99_WT_5kHz':       {'6mA', '4mC', '5mC'},   # 49,951 / 7,132 / 130
    'arabidopsis':         {'5mC'},                 # no per-mod pileup
    'hg001':               {'5mC'},
    'hg002':               {'5mC'},
}

HP_WT  = f'{RAWMOD_RESULTS}/benchmark_results/HP26695_WT_5kHz/features.h5'
HP_WGA = f'{RAWMOD_RESULTS}/benchmark_results/HP26695_WGA_5kHz/features.h5'

# Strand-split, 15-read revamp (rawmod_full_pipeline4/refeaturize_strand15.py):
# forward-strand-only pileups, height 16 (15 reads + ref row) instead of 31,
# also unbiased site/base sampling. See memory: organism-identifiability-root
# -cause (strand-pooling was found to be a major dataset/organism batch-effect
# fingerprint) and results7-8-dann-backfire. Toggle with RAWMOD_DATA_GEN=strand15.
# RAWMOD_FEATURES_ROOT overrides where every features.h5 below is read from,
# without touching this file -- e.g. to point a training run at a parallel
# tree of genomic-holdout-excluded features (see
# scripts/ground_truth/select_holdout_regions.py) instead of the default,
# unrestricted pipeline4 features.
_P4 = os.environ.get('RAWMOD_FEATURES_ROOT',
                     f'{RAWMOD_RESULTS}/rawmod_full_pipeline4/features')
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
USE_STRAND15 = os.environ.get('RAWMOD_DATA_GEN', '') in ('strand15', 'strandres')
# strandres: every dataset featurized twice, once per strand, with the minus
# file written in read orientation (--orient read). A site therefore yields two
# independent images with independent labels, because nanopore reads one strand
# at a time and a modification on the complement does not change the current of
# the sequenced strand. See scripts/featurize/refeaturize_strand_resolved.py.
USE_STRANDRES = os.environ.get('RAWMOD_DATA_GEN', '') == 'strandres'
_SR = os.environ.get('RAWMOD_STRANDRES_ROOT',
                     f'{RAWMOD_STORE}/rawmod_strand_resolved/features')
STRANDS = ('+', '-')
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
    # single-organism holdouts (finer than 'bacteria'): the other bacteria stay in stage 2
    'ecoli':      ['Ecoli_DM_5kHz', 'Ecoli_DM_MSssI_5kHz', 'Ecoli_WT_5kHz'],
    'anabaena':   ['Anabaena_WT_5kHz'],
    'tdenticola': ['Tdenticola_WT_5kHz'],
    'hpj99':      ['HPJ99_WT_5kHz'],
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


def _sr(name, strand):
    return f"{_SR}/{name}_{'plus' if strand == '+' else 'minus'}.h5"


def build_members_strandres():
    """name -> h5 path, one member per (dataset, strand).

    Member names carry the strand after a '|' so org_of() (which splits on '::')
    is unchanged, e.g. 'SPO1::bc06|-'.
    """
    sr_ont = {'5mC': 'ONT_5mC', '5hmC': 'ONT_5hmC', '6mA': 'ONT_6mA',
              'control': 'ONT_control'}
    sr_bc = {'bc06': 'barcode06', 'bc07': 'barcode07', 'bc02': 'barcode02_train',
             'bc03': 'barcode03_train', 'bc04': 'barcode04_train',
             'bc05': 'barcode05_train', 'bc01': 'barcode01_test'}
    m = {}
    for strand in STRANDS:
        if strand == '+':
            # The ONT constructs were sequenced almost entirely in one
            # orientation (e.g. 5mC: 21,923 forward images vs 490 reverse), and
            # their modification sits at 256 designed + strand positions, so the
            # minus strand offers no positives and negligible coverage.
            for mod, f in sr_ont.items():
                m[f'ONT::{mod}|{strand}'] = _sr(f, strand)
        for bc, f in sr_bc.items():
            m[f'SPO1::{bc}|{strand}'] = _sr(f, strand)
        m[f'HP::WT|{strand}'] = _sr('HP26695_WT_5kHz', strand)
        m[f'HP::WGA|{strand}'] = _sr('HP26695_WGA_5kHz', strand)
        if USE_EXTRA_ORGS:
            for name in BENCH_FILES:
                m[f'BENCH::{name}|{strand}'] = _sr(name, strand)
        if USE_HUMAN:
            for name in HUMAN_FILES:
                m[f'BENCH::{name}|{strand}'] = _sr(name, strand)
    return {k: v for k, v in m.items() if os.path.exists(v)}


def strand_of(name):
    """'+' or '-' for a member name; '+' for legacy (forward-only) members."""
    return '-' if name.endswith('|-') else '+'


def dataset_of(name):
    """Member name without the organism prefix and strand suffix."""
    return name.split('::')[1].split('|')[0]


def build_members():
    """name -> h5 path for the matched-only pool (prefixes encode the organism)."""
    if USE_STRANDRES:
        return build_members_strandres()
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
        # Retry on transient NFS read failures, same as model._h5_read(). This
        # path had no retry and it is the one that killed loco_5hmU (7643900)
        # with [Errno 5] on Ecoli_WT_5kHz_minus.h5 when 18 jobs streamed the
        # same mount: the DataLoader was protected, the setup pass was not.
        for attempt in range(5):
            try:
                with h5py.File(path, 'r') as hf:
                    L = int(hf.attrs['L']); cs = int(hf.attrs['half_window']) * L
                    oh = hf['tensors'][:, 0, cs, 2:6]        # (n,4)
                break
            except OSError as e:
                if attempt == 4:
                    raise
                print(f"  [ref_base_center] read retry {attempt + 1}/4 on "
                      f"{os.path.basename(path)}: {e}", file=sys.stderr, flush=True)
                time.sleep(0.5 * 2 ** attempt)
        out[lo:hi] = bases[np.argmax(oh, axis=1)]
    return out


CHEM_MAP_DIR = os.environ.get(
    'RAWMOD_CHEM_MAP', f'{RAWMOD_DATA}/gt_modkit/chem_map')
_chem_map_cache = {}


def modkit_chem_map(dataset):
    """{(contig, pos, strand): chem} from build_modkit_chem_map.py, or {} if
    that dataset has no per-modification pileups (ONT, human, arabidopsis,
    SPO1 -- those keep the reference-base rules below).

    Why this exists: typing a positive by its reference base assigns every
    C-site positive the one C mark BENCH_ORG_CHEMS lists for the organism.
    H. pylori 26695 carries 4mC AND 5mC AND 6mA, so that rule mislabels its
    C positives and throws away its 41,004 6mA positives as 'untyped'.
    """
    if dataset in _chem_map_cache:
        return _chem_map_cache[dataset]
    path = os.path.join(CHEM_MAP_DIR, f'{dataset}.tsv')
    m = {}
    if os.path.exists(path):
        with open(path) as fh:
            next(fh, None)
            for line in fh:
                c, pos, st, chem = line.rstrip('\n').split('\t')
                m[(c, int(pos), st)] = chem
    _chem_map_cache[dataset] = m
    return m


def chem_array(group, mod_map, refbase):
    """Per-image chemistry string ('' for unmodified/untyped)."""
    chem = np.array([''] * group.N, dtype=object)
    modified = group.labels > 0
    for i in np.nonzero(modified)[0]:
        nm = group.names[int(group.file_of[i])]
        if nm.startswith('ONT::'):
            chem[i] = dataset_of(nm)                 # control has no positives
        elif nm.startswith('SPO1::'):
            if strand_of(nm) == '-':
                # SPO1 minus positives are USUALLY the reference-A positions,
                # i.e. 5hmU on the complementary strand, and the + strand modkit
                # typing (mod_map) describes the other strand so it must not be
                # used here. But this branch used to assert '5hmU'
                # unconditionally, from the file's strand alone, without
                # checking the base -- and modkit_gt.py has a SECOND source
                # besides the all-T rule: dorado's own calls, keyed on the
                # pileup's strand column. Measured in gt_minus.bed for bc06:
                # 43,581 entries at reference-A (correct, minus base T = 5hmU)
                # but 95 at reference-T, whose minus base is A. A modified
                # adenine is 6mA, not 5hmU. Those 150 images (bc06+bc07) sat in
                # loco_6mA's TRAINING set -- the held-out chemistry, wearing a
                # 5hmU label -- and they were also what disabled the
                # orphan-base guard for that fold. refbase is already
                # complemented for a mirrored minus image, so it IS the
                # modified base; type from it, exactly as the plus branch does.
                if refbase[i] == b'T':
                    chem[i] = '5hmU'
                elif refbase[i] == b'A':
                    chem[i] = '6mA'
                else:
                    chem[i] = 'untyped'
            else:
                key = (group.contig[i], int(group.ref_pos[i]))
                t = mod_map.get(key)
                if t is None:                        # fallback by ref base
                    t = '5hmU' if refbase[i] == b'T' else 'untyped'
                chem[i] = t
        elif nm.startswith('BENCH::'):
            ds = dataset_of(nm)
            cm = modkit_chem_map(ds)
            if cm:
                chem[i] = cm.get((group.contig[i], int(group.ref_pos[i]),
                                  strand_of(nm)), 'untyped')
            else:
                # No per-modification pileup for this dataset (arabidopsis,
                # hg001/hg002): fall back to the organism's single mark.
                cs = BENCH_ORG_CHEMS.get(ds, set())
                b = refbase[i]
                if b == b'A' and '6mA' in cs:
                    chem[i] = '6mA'
                elif b == b'C':
                    chem[i] = ('5mC' if '5mC' in cs else '4mC' if '4mC' in cs else 'untyped')
                else:
                    chem[i] = 'untyped'
        elif nm.split('|')[0] == 'HP::WT':
            # H. pylori 26695 is NOT 4mC-only. Its modkit pileups give 41,004
            # confident 6mA and 16,209 confident 5mC calls alongside 19,442 4mC,
            # and the WGA control has 1, 7 and 449 respectively -- so the 6mA and
            # 5mC are real, and the old hardcoded '4mC if C' rule was scoring a
            # 4mC/5mC mixture as 4mC. The earlier hp_offset_scan.py result showed
            # only that the preset's CTTCAAG motif is not where the 6mA sits.
            chem[i] = modkit_chem_map('HP26695_WT_5kHz').get(
                (group.contig[i], int(group.ref_pos[i]), strand_of(nm)), 'untyped')
    # GLOBAL INVARIANT: a chemistry must sit on the base it modifies.
    # refbase is the image's own centre reference base, already complemented for
    # a mirrored minus image, so it IS the modified base. Anything inconsistent
    # is a mistyped image, not a discovery -- audit_loco.py found 7 HP positives
    # typed 6mA on a C and 2 bacterial positives typed 4mC on an A, all from
    # modkit reporting a mod code at a base that cannot carry it. This catches
    # every such case regardless of which source assigned the type, which is the
    # same class of bug as the SPO1 minus-strand one above.
    _BASE_OF = {'5mC': b'C', '5hmC': b'C', '4mC': b'C', '6mA': b'A', '5hmU': b'T'}
    n_bad = 0
    for i in np.nonzero(modified)[0]:
        want = _BASE_OF.get(chem[i])
        if want is not None and refbase[i] != want:
            chem[i] = 'untyped'; n_bad += 1
    if n_bad:
        print(f"  [invariant] {n_bad:,} positives retyped 'untyped': assigned "
              f"chemistry did not match centre base", flush=True)
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


def mixed_split(pool, is_pos, neg_mask, hp, core_mask=None):
    """Deterministic 85/15 position-grouped split of the whole matched pool (all 5
    chemistries + capped controls) -- the 'mixed' in-distribution fold. Factored out
    so a downstream analysis (e.g. a post-hoc embedding probe) can recompute the
    EXACT same train/test image indices used to train the mixed checkpoint, without
    re-deriving the split logic. Deterministic given (pool, SPLIT_SEED) -- NOT hp.seed,
    so this stays identical across a --seed replicate run.

    core_mask (optional) restricts the split to the core matched pool. main()
    passes ~is_bench: BENCH:: extra organisms are unioned into stage-2 training
    for EVERY fold by fit(), so leaving them in this split would put images in
    the held-out test set that the model is trained on regardless. They now go
    entirely into training, matching how every other fold treats them."""
    keep = np.nonzero((is_pos | neg_mask) if core_mask is None
                      else ((is_pos | neg_mask) & core_mask))[0]
    tr, _, te, stats = split_position_groups(
        pool.labels[keep], [pool.position_keys[i] for i in keep],
        val_frac=0.0, test_frac=0.15, seed=SPLIT_SEED)
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
    ap.add_argument('--seed', type=int, default=None,
                    help='Overrides ONLY training stochasticity (weight init, dropout, '
                         'data-loader shuffling, curriculum stage-1 -> stage-2 handoff) -- '
                         'for a clean replicate-seed run. Deliberately does NOT affect '
                         'SPLIT_SEED (train/test composition stays identical across seeds), '
                         'so a different result reflects training noise, not a different test set.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Build the pool and the fold split, print the census and the '
                         'disjointness checks, then exit without training or scoring. '
                         'Used to sanity-check a new data generation before burning GPU hours.')
    a = ap.parse_args()

    valid = ['mixed', 'all'] + [f'loco_{c}' for c in CHEMS] + [f'logo_{g}' for g in LOGO_GROUPS]
    subset_chems = parse_subset_fold(a.fold)
    if a.fold not in valid and subset_chems is None:
        raise SystemExit(f"--fold must be one of {valid}, or subset_<chem>+<chem>[+...] "
                         f"(2-4 distinct chems from {CHEMS}), got {a.fold!r}")

    hp = R.HP()
    if a.epochs:
        hp.epochs = a.epochs
    if a.seed is not None:
        hp.seed = a.seed
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
    kept_neg = subsample_negatives(pool, seed=SPLIT_SEED)
    neg_mask = np.zeros(pool.N, dtype=bool); neg_mask[kept_neg] = True

    # Extra-organism curriculum data (BENCH:: prefixed members): stage-2-only,
    # every fold, never in any test set. See build_members()/USE_EXTRA_ORGS.
    is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::')
                         for i in range(pool.N)])
    bench_idx = np.nonzero(is_bench)[0].astype(np.int64)
    # AMBIGUOUS POSITIVES (EXCLUDE_UNTYPED_POS=1, default on)
    # ------------------------------------------------------------------
    # A C-site positive that BOTH the 4mC and the 5mC single-mod models call
    # at >=80% cannot be assigned a chemistry from this data (see
    # build_modkit_chem_map.py: 81% of H. pylori's C positives are claimed by
    # both). chem_array types those 'untyped', and the CORE pool already drops
    # them -- pos_other requires chem not in ('', 'untyped'). The BENCH
    # curriculum did NOT: fit() unions every BENCH image regardless of
    # chemistry, so they entered stage-2 training as positives anyway.
    #
    # Measured by audit_loco.py, untyped positives on the held-out base that
    # were reaching training:  loco_5hmC 16,378   loco_4mC 6,733
    # loco_5mC 803   loco_6mA 146   loco_5hmU 0.
    # The 6,733 are the real exposure: some genuine fraction of them IS 4mC,
    # which made loco_4mC impossible to certify. An ambiguous label helps no
    # fold enough to be worth that, so they are dropped from training
    # everywhere. Their NEGATIVES are unaffected.
    if os.environ.get('EXCLUDE_UNTYPED_POS', '1') == '1' and len(bench_idx):
        amb = is_pos[bench_idx] & np.isin(chem[bench_idx], ['', 'untyped'])
        if amb.any():
            print(f"  [ambiguity] dropping {int(amb.sum()):,} BENCH positives with "
                  f"no assignable chemistry from stage-2 training "
                  f"({len(bench_idx):,} -> {int((~amb).sum()):,})", flush=True)
            bench_idx = bench_idx[~amb]
    if len(bench_idx):
        n_bench_sets = sum(1 for nm in names if nm.startswith('BENCH::'))
        print(f"  extra organisms (BENCH::, stage-2 training only): "
              f"{len(bench_idx):,} images across {n_bench_sets} datasets "
              f"(pos={int(is_pos[bench_idx].sum()):,} "
              f"neg={int((~is_pos[bench_idx]).sum()):,})", flush=True)

    # Per-image BENCH:: organism name (for logo_<group> filtering), '' elsewhere.
    # dataset_of() strips the '|+'/'|-' strand suffix so these still match
    # BENCH_ORG_CHEMS / LOGO_GROUPS keys under strand-resolved data
    bench_org_of = np.array([
        dataset_of(pool.names[int(pool.file_of[i])]) if is_bench[i] else ''
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
    if a.dry_run:
        os.environ['WANDB_DISABLED'] = '1'
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
    # RAWMOD_DROP_CH9=1 drops the center-base delta channel (see model.py /
    # PileupDataset.drop_ch9): 11 -> 10 input channels.
    _in_ch = 10 if os.environ.get('RAWMOD_DROP_CH9', '0') == '1' else 11
    # Architecture ablations: TF_LAYERS (cross-read Transformer depth, 0 = mean-pool read
    # tokens directly) and ROW_EMB=0 (no learned row embedding). Defaults = published model.
    tf_layers = int(os.environ.get('TF_LAYERS', '2'))
    row_emb = os.environ.get('ROW_EMB', '1') == '1'
    if tf_layers != 2 or not row_emb:
        print(f"  [arch] tf_layers={tf_layers} row_emb={row_emb}", flush=True)
    model_factory = lambda: ConvFormerV2(in_ch=_in_ch, dropout=hp.dropout,
                                         supcon_dim=supcon_dim,
                                         sad_dim=sad_dim, h=HEIGHT,
                                         layers=tf_layers, row_emb=row_emb)
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

    # RAWMOD_SCORE_CKPT=<path> : skip training and score this checkpoint on the
    # fold's test set instead. Reuses main()'s exact test_idx construction, so the
    # ceiling ("what does a model that HAS seen this chemistry get on these very
    # positions?") is measured on the identical population the LOCO run was scored
    # on -- not an approximation of it. Checked FIRST in fit(): an earlier version
    # sat on the non-curriculum return path, so with CURRICULUM=1 it silently
    # trained a fresh LOCO model instead.
    _score_ckpt = os.environ.get('RAWMOD_SCORE_CKPT', '')

    def fit(train_idx, fold_dir, runtag, extra_idx=None):
        if a.dry_run:
            extra = bench_idx if extra_idx is None else extra_idx
            n2 = len(np.union1d(train_idx, extra)) if len(extra) else len(train_idx)
            print(f"  [dry-run] would train {runtag}: core={len(train_idx):,} "
                  f"+extra={len(extra):,} -> stage2={n2:,} distinct images "
                  f"(pos={int(is_pos[np.union1d(train_idx, extra)].sum()):,})", flush=True)
            return None
        if _score_ckpt:
            m = model_factory() if model_factory else None
            if m is None:
                raise SystemExit('RAWMOD_SCORE_CKPT needs a model_factory')
            st = torch.load(_score_ckpt, map_location=device)
            sd = st.get('model_state', st.get('model', st))
            m.load_state_dict(sd)
            if isinstance(sd, dict) and 'sad_center' in sd:
                m.sad_center = sd['sad_center'].to(device)
            print(f"  [score-ckpt] loaded {_score_ckpt} (no training)", flush=True)
            return m.to(device).eval()
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

    def _fewshot(chem_x, train_idx, test_idx):
        """FEWSHOT_K=K  FEWSHOT_SEED=s  FEWSHOT_INIT=<loco checkpoint>
        Few-shot adaptation curve for a LOCO fold.

        Protocol ("pretrain on donors, adapt on K labelled target sites"):
          * The fold's held-out positions are split ONCE, by coordinate hash,
            into a SUPPORT half and a QUERY half. The split ignores K and the
            seed, so every point on a curve is scored on the SAME query set.
            Splitting by coordinate keeps the matched positive and control
            images of one site on the same side, so no shot's sequence context
            reappears in query.
          * Shots = K positive + K negative sites from SUPPORT, taken in a
            seeded hash order, so the K=10 sites are a subset of the K=100
            sites: the curve moves with K, not with which sites were drawn.
          * Fine-tune FROM the LOCO checkpoint on shots (repeated so they are
            ~25% of the set) plus a replay sample of donor training data, for a
            fixed number of epochs, keeping the FINAL weights. Validation is
            donor replay only; every shot site reaches training.
          * K=0 skips fine-tuning: it is the LOCO model scored on QUERY.
        Adding shots to a full retrain instead would bury ~50 images among
        ~800k and measure noise at small K.
        """
        import zlib
        K = int(os.environ['FEWSHOT_K']); fseed = int(os.environ.get('FEWSHOT_SEED', '0'))
        init = os.environ['FEWSHOT_INIT']
        def _h(i, tag):
            key = f"{tag}:{pool.contig[i]}:{int(pool.ref_pos[i])}".encode()
            return (zlib.crc32(key) & 0xffffffff) / 0xffffffff
        test_idx = np.asarray(test_idx, dtype=np.int64)
        in_sup = np.array([_h(int(i), 'fs_support') < 0.5 for i in test_idx], dtype=bool)
        sup_idx, query_idx = test_idx[in_sup], test_idx[~in_sup]
        # group support images by SOURCE-aware site (positive and control images
        # at one coordinate are different sources, hence different groups)
        skeys = pool.source_keys(sup_idx)
        groups = {}
        for i, k in zip(sup_idx, skeys):
            groups.setdefault(k, []).append(int(i))
        rank = lambda k: zlib.crc32(f"fsrank:{fseed}:{k}".encode())
        gpos = sorted((k for k, v in groups.items() if is_pos[v[0]]), key=rank)
        gneg = sorted((k for k, v in groups.items() if not is_pos[v[0]]), key=rank)
        kp, kn = min(K, len(gpos)), min(K, len(gneg))
        q_pos = int(is_pos[query_idx].sum())
        print(f"  [fewshot] {chem_x} K={K} seed={fseed}: support sites pos={len(gpos):,} "
              f"neg={len(gneg):,} -> shots pos={kp} neg={kn}"
              + ("  (CAPPED: fewer support sites than K)" if kp < K or kn < K else "")
              + f" | query images={len(query_idx):,} (pos={q_pos:,})", flush=True)

        mode = os.environ.get('FEWSHOT_MODE', 'finetune')
        if mode == 'pool':
            # IN-POOL: add the K shot sites (all their images, unrepeated) to the
            # LOCO training pool and train from scratch with the normal recipe
            # (curriculum + BENCH extras). Answers "how many labelled target
            # sites must the training pool contain", and at K=all support sites
            # it is the leak-free supervised ceiling on the SAME query half.
            shot_idx = np.array(sorted(i for k in gpos[:kp] + gneg[:kn] for i in groups[k]),
                                dtype=np.int64)
            tr_pool = np.unique(np.concatenate([np.asarray(train_idx, np.int64), shot_idx]))
            R.assert_disjoint(tr_pool, query_idx, pool, a.fold)
            print(f"  [fewshot-pool] +{len(shot_idx):,} shot images -> train pool "
                  f"{len(tr_pool):,}", flush=True)
            m = fit(tr_pool, f'{a.fold}_pool{K}_s{fseed}', 'loco', extra_idx=clean_extra)
            record(m, f'fewshot{K}_{chem_x}', query_idx, held=chem_x)
            return

        mf = model_factory
        st = torch.load(init, map_location=device)
        sd = st.get('model_state', st.get('model', st))
        if K == 0:
            m = mf(); m.load_state_dict(sd); m = m.to(device).eval()
            record(m, f'fewshot0_{chem_x}', query_idx, held=chem_x)
            return

        shot_idx = np.array(sorted(i for k in gpos[:kp] + gneg[:kn] for i in groups[k]),
                            dtype=np.int64)
        # donor replay: disjoint train/val halves by coordinate hash
        rng = np.random.default_rng(1000 + fseed)
        tr = np.asarray(train_idx, dtype=np.int64)
        rv = np.array([_h(int(i), 'fs_replayval') < 0.1 for i in tr], dtype=bool)
        n_rep = int(os.environ.get('FEWSHOT_REPLAY', '30000'))
        rep_tr = rng.choice(tr[~rv], min(n_rep, int((~rv).sum())), replace=False)
        rep_va = rng.choice(tr[rv], min(n_rep // 6, int(rv.sum())), replace=False)
        reps = max(1, min(50, int(round(0.25 * len(rep_tr) / max(len(shot_idx), 1)))))
        ft_idx = np.sort(np.concatenate([rep_tr, np.tile(shot_idx, reps)]))
        print(f"  [fewshot] shot images={len(shot_idx):,} x{reps} + replay={len(rep_tr):,} "
              f"(val replay={len(rep_va):,})", flush=True)

        hpf = R.HP()
        hpf.epochs = int(os.environ.get('FEWSHOT_EPOCHS', '8'))
        hpf.patience = hpf.epochs
        hpf.lr = float(os.environ.get('FEWSHOT_LR', '2e-4'))
        mdir = out / 'models' / f'{a.fold}_fs{K}_s{fseed}'
        m = R.train_one_model(pool, ft_idx, hpf, device, mdir, f'fs{K}_s{fseed}',
                              model_factory=mf, init_state=sd,
                              val_idx=np.sort(rep_va), keep_last=True)
        record(m, f'fewshot{K}_{chem_x}', query_idx, held=chem_x)

    def record(model, test_name, idx, held=''):
        if a.dry_run:
            y = is_pos[np.asarray(idx, np.int64)]
            print(f"  [dry-run] test {test_name}: n={len(idx):,} pos={int(y.sum()):,} "
                  f"neg={int((~y).sum()):,}", flush=True)
            return
        m = R.evaluate(model, pool, idx, device, hp)
        if getattr(model, 'sad_head', None) is not None:
            m['auroc_sad'] = sad_auroc(model, idx)     # anomaly-score AUROC
        rows.append({'fold': a.fold, 'test_set': test_name, 'held_out': held, **m})
        if R.LAST_EVAL is not None:                 # per-position scores for bootstrap CIs
            sdir = out / 'scores'; sdir.mkdir(parents=True, exist_ok=True)
            yt_, yp_ = R.LAST_EVAL
            np.savez_compressed(sdir / f'{a.fold}__{test_name}.npz', y_true=yt_, y_score=yp_)
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
        train_idx, test_idx, stats = mixed_split(pool, is_pos, neg_mask, hp,
                                                 core_mask=~(is_bench | is_bgctrl))
        print(f"  {stats}", flush=True)
        model = fit(train_idx, a.fold, 'mixed')
        record(model, 'held_out_test', test_idx)

    elif a.fold.startswith('loco_'):
        chem_x = a.fold[len('loco_'):]
        # controls: position-grouped 85/15 over the capped control pool
        ctrl_idx = np.nonzero(neg_mask)[0]
        tr_ctrl, te_ctrl = pos_hash_split(pool, ctrl_idx, test_frac=0.15, seed=SPLIT_SEED)
        # test negatives: from CHEM's organism(s) AND ref-base-matched
        bases = CHEM_BASES[chem_x]; orgs = CHEM_ORGS[chem_x]
        te_ctrl = np.array([i for i in te_ctrl
                            if org_of(pool.names[int(pool.file_of[i])]) in orgs
                            and refbase[i] in bases], dtype=np.int64)
        # BENCH:: images are stage-2 curriculum data only (see build_members /
        # fit()): per-image chemistry typing (chem_array) now resolves them, but
        # that typing exists for the leak guard below, NOT to promote them into
        # the core train/test sets -- the bacterial BENCH:: sets are ~100%
        # positive, so putting them in a test set would hand the model an
        # organism-identity shortcut instead of a chemistry one.
        pos_x = np.nonzero(is_pos & (chem == chem_x) & ~is_bench)[0].astype(np.int64)
        pos_other = np.nonzero(is_pos & (chem != chem_x) & (chem != '')
                               & (chem != 'untyped') & ~is_bench)[0].astype(np.int64)

        # BASE-BALANCED POSITIVES (POS_BASE_CAP_RATIO=R)
        # ------------------------------------------------------------------
        # Measured for loco_6mA: training positives are T=119,710, C=6,231,
        # A=150. SPO1's 5hmU is 95% of everything the model ever sees labelled
        # "modified", so "modified" collapses to "is a T" -- and the fold then
        # asks for A. Capping the dominant base at R x the next-largest keeps
        # the majority chemistry from defining the positive class.
        ratio = float(os.environ.get('POS_BASE_CAP_RATIO', '0'))
        if ratio > 0 and len(pos_other):
            rng = np.random.default_rng(SPLIT_SEED)
            by_base = {}
            for i in pos_other:
                by_base.setdefault(refbase[i], []).append(i)
            counts = sorted((len(v) for v in by_base.values()), reverse=True)
            floor = counts[1] if len(counts) > 1 else counts[0]
            cap = max(int(ratio * floor), 1)
            kept = []
            for b, idx in sorted(by_base.items()):
                idx = np.asarray(idx, dtype=np.int64)
                if len(idx) > cap:
                    idx = rng.choice(idx, cap, replace=False)
                kept.append(idx)
                print(f"  [base-balance] {b.decode()}: "
                      f"{len(by_base[b]):,} -> {len(idx):,}", flush=True)
            pos_other = np.sort(np.concatenate(kept))

        # POS_DROP_BASES=T[,C] : remove training positives at these bases
        # ENTIRELY, not merely cap them. For loco_6mA the core training
        # positives are 95% T (SPO1 5hmU); capping leaves T an equal-sized
        # class, whereas dropping it forces the model to reach A from C
        # evidence alone. Deliberately spends the C folds to buy the two
        # orphan-base folds.
        drop_b = os.environ.get('POS_DROP_BASES', '')
        if drop_b and len(pos_other):
            want = {b.strip().encode() for b in drop_b.split(',') if b.strip()}
            keepm = np.array([refbase[i] not in want for i in pos_other], dtype=bool)
            print(f"  [pos-drop-bases] dropping {sorted(b.decode() for b in want)}: "
                  f"{len(pos_other):,} -> {int(keepm.sum()):,} training positives",
                  flush=True)
            pos_other = pos_other[keepm]

        # POS_CHEM_OVERSAMPLE="5hmC:10,4mC:2"  /  POS_DROP_CHEMS="6mA"
        # ------------------------------------------------------------------
        # Base-level balancing (POS_BASE_CAP_RATIO) cannot reach inside a base:
        # measured pool census is 5mC 73,098 / 6mA 67,036 / 5hmU 119,710 /
        # 4mC 3,490 / 5hmC 1,280, so within C the 5mC:5hmC ratio is 57:1 and
        # 5hmC ends up ~4% of loco_5hmU's 31,155 training positives.
        #
        # That matters because 5hmC is the closest chemical analogue of 5hmU in
        # this corpus -- both are a 5-hydroxymethyl on a pyrimidine ring, i.e.
        # the same (modified-position) family that matched-LOCO transfer was
        # measured to work within. The donor most likely to carry the target is
        # the one training barely sees.
        #
        # OVERSAMPLE repeats indices (duplicates in train_idx are harmless and
        # stay disjoint from test); DROP_CHEMS removes a chemistry outright, e.g.
        # to strip the purine 6mA distractor from a pyrimidine target's fold.
        _drop_ch = os.environ.get('POS_DROP_CHEMS', '')
        if _drop_ch and len(pos_other):
            want = {c.strip() for c in _drop_ch.split(',') if c.strip()}
            keepm = ~np.isin(chem[pos_other], list(want))
            print(f"  [pos-drop-chems] dropping {sorted(want)}: "
                  f"{len(pos_other):,} -> {int(keepm.sum()):,} training positives",
                  flush=True)
            pos_other = pos_other[keepm]
        _os_spec = os.environ.get('POS_CHEM_OVERSAMPLE', '')
        if _os_spec and len(pos_other):
            reps = {}
            for part in _os_spec.split(','):
                if ':' in part:
                    c, n = part.split(':'); reps[c.strip()] = int(n)
            add = []
            for c, n in sorted(reps.items()):
                sel = pos_other[chem[pos_other] == c]
                if len(sel) and n > 1:
                    add.append(np.tile(sel, n - 1))
                    print(f"  [pos-chem-oversample] {c}: {len(sel):,} x{n} "
                          f"(+{len(sel)*(n-1):,} repeats)", flush=True)
            if add:
                pos_other = np.sort(np.concatenate([pos_other] + add))

        # CTRL_TARGET_BASE_TO_TEST=0|orphan|all
        # ------------------------------------------------------------------
        # An orphan-base fold only makes sense if the control positions AT THE
        # TARGET BASE are spent on the measurement rather than on training.
        #
        # Measured for loco_5hmU: pos_x takes ALL 5hmU positives into test
        # un-split, but the negatives are only pos_hash_split's 15% slice, and
        # SPO1 controls are NEG_CAP'd (40k default) of which ~50% are T-centred.
        # After position aggregation that left 807 negatives against 23,942
        # positives (96.7% positive), so AUROC was estimated against <1k points
        # and moved on run-to-run noise alone.
        #
        # The other 85% was not missing -- it was in TRAINING, where (per the
        # orphan-base note below) it is exactly the population that installs
        # "T is unmodified" before the fold asks for modified T. Moving it to
        # test fixes the class balance and removes the harmful prior at once.
        # Whole positions move (org and refbase are constant within a
        # coordinate), so position-grouped disjointness still holds --
        # assert_disjoint below re-checks it.
        #
        # mode=orphan restricts this to folds whose target base carries no
        # training positives, making it a no-op on 5mC/5hmC/4mC (which share C
        # and DO need their C negatives in training). Must run after
        # base-balancing and POS_DROP_BASES, since those decide pos_other.
        #
        # NOTE: this CHANGES the test set. Numbers are not comparable to runs
        # made without it; report as its own series with class balance stated.
        _c2t = os.environ.get('CTRL_TARGET_BASE_TO_TEST', '0')
        if _c2t not in ('0', '', 'off'):
            _share = float(os.environ.get('ORPHAN_BASE_MIN_SHARE', '0.01'))
            _cnt = collections.Counter(refbase[pos_other].tolist())
            _tot = max(sum(_cnt.values()), 1)
            _covered = {b for b, n in _cnt.items() if n / _tot >= _share}
            _orphan = tuple(b for b in bases if b not in _covered)
            if _c2t == 'orphan' and not _orphan:
                print(f"  [ctrl->test] {chem_x}: target base(s) "
                      f"{[b.decode() for b in bases]} already carry training "
                      f"positives; no move (mode=orphan)", flush=True)
            else:
                mv = np.array([org_of(pool.names[int(pool.file_of[i])]) in orgs
                               and refbase[i] in bases for i in tr_ctrl], dtype=bool)
                # Two INDEPENDENT knobs, both position-grouped by coordinate:
                #   CTRL_TARGET_BASE_TEST_FRAC  (default 1.0) fraction of
                #     target-base control positions moved to TEST.
                #   CTRL_TARGET_BASE_TRAIN_KEEP (default 1.0) fraction of the
                #     NOT-moved remainder retained in TRAINING (rest dropped).
                # Keeping these separate is what makes the arms comparable: fix
                # TEST_FRAC and vary TRAIN_KEEP and every arm is scored on the
                # SAME test set, so the only thing changing is how much
                # target-base negative evidence training sees.
                #
                # Why it matters: moving 100% makes the target base vanish from
                # training entirely (no positives -- held out -- and no negatives
                # either), so the model ranks modified vs unmodified T having
                # never seen a T-centred image. Measured that way, loco_5hmU is
                # BELOW chance (A 0.478, C 0.482; best arm B only 0.526).
                import zlib
                def _h(i, tag):
                    key = f"{tag}:{pool.contig[i]}:{int(pool.ref_pos[i])}".encode()
                    return (zlib.crc32(key) & 0xffffffff) / 0xffffffff
                _tf = float(os.environ.get('CTRL_TARGET_BASE_TEST_FRAC', '1'))
                _tk = float(os.environ.get('CTRL_TARGET_BASE_TRAIN_KEEP', '1'))
                if _tf < 1 or _tk < 1:
                    cand = np.flatnonzero(mv)
                    mv = np.zeros(len(tr_ctrl), dtype=bool)   # rebuild: move set
                    drop = np.zeros(len(tr_ctrl), dtype=bool)  # dropped entirely
                    for k in cand:
                        i = int(tr_ctrl[k])
                        if _h(i, 'c2t') < _tf:
                            mv[k] = True
                        elif _h(i, 'keep') >= _tk:
                            drop[k] = True
                    if drop.any():
                        tr_ctrl = tr_ctrl[~drop]
                        mv = mv[~drop]
                        print(f"  [ctrl->test] test_frac={_tf} train_keep={_tk}: "
                              f"dropped {int(drop.sum()):,} target-base control "
                              f"images from training entirely", flush=True)
                if mv.any():
                    te_ctrl = np.sort(np.concatenate([te_ctrl, tr_ctrl[mv]]))
                    tr_ctrl = tr_ctrl[~mv]
                    print(f"  [ctrl->test] moved {int(mv.sum()):,} target-base "
                          f"control images ({[b.decode() for b in bases]} in "
                          f"{list(orgs)}) train->test; test negatives now "
                          f"{len(te_ctrl):,}, train negatives {len(tr_ctrl):,}",
                          flush=True)

        # ORPHAN-BASE GUARD (LOCO_DROP_ORPHAN_BASE_NEGS=1)
        # ------------------------------------------------------------------
        # 5hmU and 6mA are each the ONLY modification in this corpus on their
        # base (T and A). Holding one out therefore leaves training with that
        # base present EXCLUSIVELY as negative evidence -- measured in the SPO1
        # arms: bc01/bc02 controls are 48-50% T-centred with zero positives,
        # while bc06 positives are 99.8% T. So loco_5hmU trains the rule
        # "T is unmodified" and is then scored on modified T, which is why it
        # sits at or below chance (0.481) rather than merely failing to
        # transfer. loco_6mA has the identical structure for A.
        #
        # The C folds never hit this: 5mC/5hmC/4mC share the C base, so holding
        # out one still leaves C positives from the others in training.
        #
        # With the guard on, training negatives at a base that has NO training
        # positives are dropped. They stay in the TEST set, so the measurement
        # is unchanged -- only the installed prior goes away.
        if os.environ.get('LOCO_DROP_ORPHAN_BASE_NEGS', '0') == '1':
            # A base counts as "covered" only if it carries a meaningful share
            # of the training positives. loco_6mA has exactly 150 A-centred
            # positives (SPO1 minus-strand reference-A images typed 5hmU) out
            # of 126,091, and a bare `in` test let those 150 disable the guard
            # for the fold that needed it most.
            share = float(os.environ.get('ORPHAN_BASE_MIN_SHARE', '0.01'))
            cnt = collections.Counter(refbase[pos_other].tolist())
            tot = max(sum(cnt.values()), 1)
            train_bases = {b for b, n in cnt.items() if n / tot >= share}
            orphan = tuple(b for b in CHEM_BASES[chem_x] if b not in train_bases)
            if orphan:
                drop = np.array([refbase[i] in orphan for i in tr_ctrl], dtype=bool)
                print(f"  [orphan-base guard] {chem_x} target base(s) "
                      f"{[b.decode() for b in orphan]} have no training positives; "
                      f"dropping {int(drop.sum()):,} of {len(tr_ctrl):,} training "
                      f"negatives at those bases", flush=True)
                tr_ctrl = tr_ctrl[~drop]
            else:
                print(f"  [orphan-base guard] {chem_x}: training already has "
                      f"positives at every target base; no change", flush=True)

        train_idx = np.sort(np.concatenate([pos_other, tr_ctrl]))
        test_idx = np.sort(np.concatenate([pos_x, te_ctrl]))
        R.assert_disjoint(train_idx, test_idx, pool, a.fold)

        # BENCH:: leak fix (see BENCH_ORG_CHEMS): strip out any BENCH:: organism
        # that biologically carries chem_x before unioning the curriculum data
        # into stage-2 training, so a "held-out" chemistry is actually never
        # seen anywhere in training, not just absent from the core pool.
        leaky_orgs = {o for o, cs in BENCH_ORG_CHEMS.items() if chem_x in cs}
        if USE_STRANDRES:
            # Per-image typing (see chem_array) means an organism carrying two
            # chemistries keeps the one that is NOT held out: withholding 6mA no
            # longer throws away E. coli's Dcm 5mC as collateral. Untyped BENCH
            # images from a leaky organism are still dropped, since their
            # chemistry is unresolved and could be the held-out one.
            drop = is_bench & ((chem == chem_x)
                               | (np.isin(bench_org_of, list(leaky_orgs))
                                  & (chem == 'untyped')))
            clean_extra = np.nonzero(is_bench & ~drop)[0].astype(np.int64)
        else:
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
        if os.environ.get('FEWSHOT_K') is None:
            model = fit(train_idx, a.fold, 'loco', extra_idx=clean_extra)
            record(model, f'zeroshot_{chem_x}', test_idx, held=chem_x)
        else:
            _fewshot(chem_x, train_idx, test_idx)

    elif a.fold.startswith('subset_'):
        include_chems = subset_chems  # validated at top of main()
        held_chems = [c for c in CHEMS if c not in include_chems]
        print(f"  subset training chemistries: {include_chems}  "
              f"(zero-shot held out: {held_chems})", flush=True)

        ctrl_idx = np.nonzero(neg_mask)[0]
        tr_ctrl, te_ctrl_all = pos_hash_split(pool, ctrl_idx, test_frac=0.15, seed=SPLIT_SEED)
        pos_incl = np.nonzero(is_pos & np.isin(chem, include_chems) & ~is_bench)[0].astype(np.int64)
        train_idx = np.sort(np.concatenate([pos_incl, tr_ctrl]))
        print(f"  train={len(train_idx):,} (pos_incl={len(pos_incl):,} neg={len(tr_ctrl):,})",
              flush=True)
        model = fit(train_idx, a.fold, 'subset')

        for chem_x in held_chems:
            bases = CHEM_BASES[chem_x]; orgs = CHEM_ORGS[chem_x]
            te_ctrl = np.array([i for i in te_ctrl_all
                                if org_of(pool.names[int(pool.file_of[i])]) in orgs
                                and refbase[i] in bases], dtype=np.int64)
            pos_x = np.nonzero(is_pos & (chem == chem_x) & ~is_bench)[0].astype(np.int64)
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
        bg_org_of = np.array([dataset_of(pool.names[int(pool.file_of[i])]) if is_bgctrl[i] else ''
                              for i in range(pool.N)])
        bg_mask = is_bgctrl & np.isin(bg_org_of, held_orgs)
        if bg_mask.any():
            # bacterial BENCH:: sets are 100% positive (see BGCTRL_FILES docstring) -- add the
            # held-out organisms' non-motif background negatives, test-only, never trained on.
            # For logo_bacteria this is every BGCTRL:: image, as before.
            bg_idx = np.nonzero(bg_mask)[0].astype(np.int64)
            test_idx = np.sort(np.concatenate([test_idx, bg_idx]))
            print(f"  {a.fold}: +{len(bg_idx):,} BGCTRL:: background negatives "
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
