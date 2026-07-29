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

Chemistries present: 5hmU, 5mC, 5hmC, 6mA, 4mC.

FOLDS (one SLURM job each)
  loco_<CHEM>  leave-one-chemistry-out:
     train = {positives typed != CHEM}  U  {85% of controls, position-grouped}
     test  = {positives typed == CHEM}  U  {15% controls, from the organism(s)
              carrying CHEM AND whose centre ref base matches CHEM's target
              base(s)} -- a pure signal contrast (e.g. modified-T vs unmodified-T
              for 5hmU), not a trivial base-composition split.
     This is the zero-shot "modification-agnostic" test: the model is scored on a
     chemistry it never saw, using causal negatives from the same sample.
  mixed        position-grouped 85/15 split over the whole matched pool
               (in-distribution reference point).

MODEL: ConvFormerV2(supcon_dim=SUPCON_DIM, default 128) trained by
run_pipeline.train_one_model with total loss BCE + SUPCON_WEIGHT*SupCon on the
causal labels. All model/train/eval code is imported from the repo; this file
only assembles the matched pool and defines the LOCO splits.

Usage:
  python run_matched_loco.py --fold {mixed|loco_5hmU|loco_4mC|loco_6mA|loco_5mC|loco_5hmC} \
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
for _p in (REPO / 'experiments' / 'pipeline1', REPO / 'deepmod'):
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

HP_WT  = '/fs/cbcb-scratch/bds062/results/benchmark_results/HP26695_WT_5kHz/features.h5'
HP_WGA = '/fs/cbcb-scratch/bds062/results/benchmark_results/HP26695_WGA_5kHz/features.h5'


def build_members():
    """name -> h5 path for the matched-only pool (prefixes encode the organism)."""
    m = {}
    for mod in R.ONT_ORDER:                 # 5mC,5hmC,6mA,control
        m[f'ONT::{mod}'] = R.ONT_FILES[mod]
    for bc in R.UMCES_ORDER:                # bc06,bc07 (pos) + bc01-05 (neg)
        m[f'SPO1::{bc}'] = R.UMCES_FILES[bc]
    m['HP::WT'] = HP_WT
    m['HP::WGA'] = HP_WGA
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

    valid = ['mixed', 'all'] + [f'loco_{c}' for c in CHEMS]
    if a.fold not in valid:
        raise SystemExit(f"--fold must be one of {valid}, got {a.fold!r}")

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
                                         sad_dim=sad_dim)
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

    def fit(train_idx, fold_dir, runtag):
        mdir = out / 'models' / fold_dir
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
                return R.train_one_model(pool, train_idx, hp, device, mdir, runtag,
                                         model_factory=model_factory, init_state=state)
            print("  [curriculum] too few paired anchors; single-stage fallback",
                  flush=True)
        return R.train_one_model(pool, train_idx, hp, device, mdir, runtag,
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

    else:  # loco_<CHEM>
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
        print(f"  train={len(train_idx):,} (pos_other={len(pos_other):,} "
              f"neg={len(tr_ctrl):,})  test={len(test_idx):,} "
              f"(pos_{chem_x}={len(pos_x):,} neg={len(te_ctrl):,})", flush=True)
        if len(pos_x) == 0 or len(te_ctrl) == 0:
            raise SystemExit(f"empty test for {a.fold}: pos={len(pos_x)} neg={len(te_ctrl)}")
        model = fit(train_idx, a.fold, 'loco')
        record(model, f'zeroshot_{chem_x}', test_idx, held=chem_x)

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
