#!/usr/bin/env python3
"""
deepmod_full_pipeline2 — same ConvFormer-v2 architecture + evaluation
machinery as deepmod_full_pipeline1/results6, retargeted at 7 pooled
training datasets: the 5 new real-genome datasets (bacteria + arabidopsis)
PLUS the original ONT synthetic benchmark and UMCES/SPO1 phage data that
results1-6 were built on.

7 role=train "datasets" (each is one leave-one-out unit, though ONT and
UMCES are themselves pools of several underlying files):
  HP26695_WGA, Ecoli_DM, Ecoli_DM_MSssI, Ecoli_WT, arabidopsis  (single-file,
      from deepmod_genomes/manifest.tsv)
  UMCES  — 7 SPO1 phage barcode files (R.UMCES_FILES/UMCES_ORDER)
  ONT    — 4 synthetic-benchmark condition files (R.ONT_FILES/ONT_ORDER)

role=test (4, from the manifest) : HP26695_WT, HPJ99_WT, Anabaena_WT,
  Tdenticola_WT — motif GT of uncertain penetrance (R-M systems subject to
  phase variation), never trained on anywhere in this pipeline, used purely
  as a fixed external generalization check for every fold below.

Three kinds of runs, one per SLURM job (--fold):

  mixed             Train on all 7 role=train datasets pooled (leakage-safe
                    position-grouped 85/15 train/test split via
                    model.split_position_groups).

  lodo_<DATASET>    Leave-one-dataset-out: train on every OTHER role=train
                    dataset, test on ALL of the held-out one (never seen in
                    training). One job per role=train dataset name (7 total,
                    including lodo_UMCES / lodo_ONT).

  lomo_5mC          Leave-one-modification-out, applied ONLY to the mixed pool
  lomo_5hmC         (not repeated per LODO fold, per instruction), using the
  lomo_6mA          SAME test definition as deepmod_full_pipeline1 so the bars
  lomo_5hmU         are directly comparable to results6:
                      ONT_heldout_<mod>   — that modification's whole ONT file,
                                            dropped from training (5hmU has no
                                            ONT file, so it keeps all of ONT)
                      UMCES_heldout_<mod> — R.umces_lomo_split(): {modified &
                                            type==mod} u {unmodified & in the
                                            held-out region}, where the UMCES
                                            type comes from the modkit dominant
                                            code (T => 5hmU) via mod_types.
                    Bacteria/arabidopsis have no per-file modification identity,
                    so their POSITIVES are typed by reference base (Dam 6mA only
                    methylates A; Dcm/M.SssI/EM-seq 5mC only methylates C; none
                    carry 5hmC/5hmU) and withheld from TRAINING for the matching
                    fold — otherwise "5mC held out" would be false while
                    arabidopsis 5mC is still trained on. They are not added to
                    the test set, keeping the test identical to pipeline1's.

Every fold's final model is additionally scored against the 4 fixed
role=test datasets, so LODO/LOMO/mixed can be compared on how well each
generalizes to organisms nobody trains on.

Run collect2.py after all folds finish to merge metrics and draw figures.

Usage:
  python run_pipeline2.py --fold {mixed|lodo_<name>|lomo_6mA|lomo_5mC} \\
      [--out-dir results1] [--epochs N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PIPE1 = Path('/fs/cbcb-scratch/bds062/results/deepmod_full_pipeline1')
DEEPMOD = Path('/fs/nexus-scratch/bds062/Nanopore-Modification/deepmod')
sys.path.insert(0, str(PIPE1))
sys.path.insert(0, str(DEEPMOD))

import run_pipeline as R                          # noqa: E402 — Group, train_one_model,
                                                    # evaluate, assert_disjoint, HP, set_seed,
                                                    # ONT_FILES, ONT_ORDER, UMCES_FILES, UMCES_ORDER
from run_convformer_v2 import ConvFormerV2         # noqa: E402
from model import split_position_groups            # noqa: E402
from mod_types import build_umces_mod_map          # noqa: E402 — UMCES LOMO typing

# ── Weights & Biases (opt-out with WANDB_DISABLED=1) ───────────────────────────
# Auth is via ~/.netrc (machine api.wandb.ai). Logs to
# bds062-university-of-maryland/rawmod; each run is named
# pipeline2-<8char id>-<fold>  (mirrors the trokens WANDB_ID convention:
# experiment tag + random suffix, so re-runs of the same fold never collide).
# NOTE: the personal `bds062` entity is not writable on this W&B deployment
# (personal projects disabled), so we use the account's UMD team namespace.
# Override with WANDB_ENTITY / WANDB_PROJECT.
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', 'bds062-university-of-maryland')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'rawmod')


def wandb_init(fold: str, hp, train_names, test_names):
    if os.environ.get('WANDB_DISABLED', '').lower() in ('1', 'true', 'yes'):
        return None
    # cbcb compute nodes DO have outbound internet (verified: they resolve and
    # reach api.wandb.ai), so we log ONLINE in real time — no offline/`wandb sync`
    # step. Auth comes from ~/.netrc. Set WANDB_MODE=offline to override on a node
    # without connectivity; the try/except below degrades gracefully either way.
    os.environ.setdefault('WANDB_MODE', 'online')
    os.environ.setdefault('WANDB_DIR', str(Path(__file__).resolve().parent))
    try:
        import wandb
    except Exception as e:
        print(f"  [wandb] import failed ({e}); continuing without logging",
              file=sys.stderr)
        return None
    import secrets
    run_id = secrets.token_hex(4)                      # 8 hex chars
    run_name = f"pipeline2-{run_id}-{fold}"
    try:
        run = wandb.init(
            entity=WANDB_ENTITY, project=WANDB_PROJECT,
            name=run_name, group='pipeline2', job_type=fold.split('_')[0],
            config={
                'fold': fold, 'run_id': run_id, 'architecture': 'ConvFormerV2',
                'train_datasets': train_names, 'external_test_datasets': test_names,
                **{k: getattr(hp, k) for k in dir(hp) if not k.startswith('_')},
            },
            settings=wandb.Settings(init_timeout=180, start_method='thread'),
        )
        print(f"  [wandb] logging to {WANDB_ENTITY}/{WANDB_PROJECT}/{run_name}",
              file=sys.stderr)
        return run
    except Exception as e:
        print(f"  [wandb] init failed ({e}); continuing without logging",
              file=sys.stderr)
        return None

# Override with MANIFEST=/path/to/manifest.tsv to point folds at a different
# feature set (e.g. the balanced-label v2 features used for results2).
MANIFEST = Path(os.environ.get('MANIFEST')  # empty/unset -> default
                or '/fs/cbcb-scratch/bds062/results/deepmod_genomes/manifest.tsv')

# Leave-one-modification-out uses the SAME test definition as
# deepmod_full_pipeline1 (results6-style), so the bars are directly comparable:
#   ONT_heldout_<mod>   — the whole ONT file for that modification
#   UMCES_heldout_<mod> — R.umces_lomo_split(): {modified & type==mod}
#                         u {unmodified & in held-out region}
# Bacteria/arabidopsis have no ONT/UMCES-style per-file modification identity;
# their *positives* are typed by reference base (Dam/Anabaena 6mA only ever
# methylates A; Dcm/M.SssI/EM-seq 5mC only ever methylates C, and none of them
# carry 5hmC/5hmU). Those positives are dropped from TRAINING for the matching
# fold — otherwise "5mC held out" would be false while arabidopsis 5mC is still
# trained on — but they are not added to the test set, to keep the test
# identical to pipeline1's.
LOMO_MODS = tuple(R.UMCES_LOMO_MODS)          # ('5mC', '5hmC', '6mA', '5hmU')
# Bacteria/arabidopsis positive typing only. motif_gt.py searches BOTH strands and
# keys the GT by (contig, pos) with no strand column, so a reverse-strand
# modification lands on a position whose *forward* reference base is the
# complement: 6mA -> forward A (+) or T (-); 5mC -> forward C (+) or G (-).
# Verified empirically at the true window centre: Ecoli_DM positives are
# A=50.0%/T=50.0% (palindromic GATC) and arabidopsis 5mC is C=54%/G=46% (CpG).
# Keying on A/C alone would silently miss half of every dataset's positives.
LOMO_REF_BASES = {'6mA': (b'A', b'T'), '5mC': (b'C', b'G')}


# The manifest also carries a few role=train rows (hg001, hg002, mouse,
# osativa) that are aspirational placeholders for datasets not yet
# featurized (blocked on separate issues — corrupted GT download, refinement
# failure, not started). Those aren't part of this experiment; only these 5
# are the real, current bacteria/plant role=train datasets.
MANIFEST_TRAIN_NAMES = {'HP26695_WGA_5kHz', 'Ecoli_DM_5kHz', 'Ecoli_DM_MSssI_5kHz',
                        'Ecoli_WT_5kHz', 'arabidopsis'}


def load_manifest():
    train_files, test_files = {}, {}
    with open(MANIFEST) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 6 or cols[0] == 'name':
                continue
            name, h5path, _organism, _mod, _gt_source, role = cols[:6]
            if role == 'train' and name in MANIFEST_TRAIN_NAMES:
                train_files[name] = h5path
            elif role == 'test':
                test_files[name] = h5path
    return train_files, test_files


def load_train_dataset_members(manifest_train_files: dict) -> dict:
    """
    dataset_key -> {member_name: h5_path}. The 5 manifest datasets are each a
    single member; UMCES and ONT are pools of several underlying files, but
    still count as ONE leave-one-out unit each. Member names are prefixed
    with their dataset key so they can never collide across datasets when
    flattened into one R.Group's name list.
    """
    members = {name: {name: path} for name, path in manifest_train_files.items()}
    members['UMCES'] = {f'UMCES::{n}': R.UMCES_FILES[n] for n in R.UMCES_ORDER}
    members['ONT'] = {f'ONT::{n}': R.ONT_FILES[n] for n in R.ONT_ORDER}
    return members


def flatten(dataset_keys, members_by_dataset: dict):
    """Concatenate the member (name, path) pairs of the given dataset keys
    into one flat (names_list, files_dict) pair for R.Group()."""
    names, files = [], {}
    for key in dataset_keys:
        for member_name, path in members_by_dataset[key].items():
            names.append(member_name)
            files[member_name] = path
    return names, files


def compute_ref_base_for(group, sel: np.ndarray) -> np.ndarray:
    """
    True reference base ('A'/'C'/'G'/'T' as bytes) for the given global image
    indices, read directly from each image's own tensor (reference row, center
    window position, is_A/is_C/is_G/is_T channels) — the ground truth the model
    itself is built from. Deliberately NOT re-derived from an independently-
    parsed genome coordinate: the per-image reference window is built from
    whatever read(s) contributed that image, and indels in those reads' own
    alignment can locally shift the window-to-genome mapping, so (contig,
    ref_pos) arithmetic against a separate FASTA is not reliable here.
    """
    sel = np.asarray(sel, dtype=np.int64)
    out = np.empty(len(sel), dtype='S1')
    bases = np.array([b'A', b'C', b'G', b'T'])
    offsets = np.concatenate([[0], np.cumsum(group.file_sizes)])
    for fi, path in enumerate(group.paths):
        lo, hi = int(offsets[fi]), int(offsets[fi + 1])
        m = (sel >= lo) & (sel < hi)
        if not m.any():
            continue
        where = np.nonzero(m)[0]
        local = sel[m] - lo
        order = np.argsort(local)              # h5py fancy index must be increasing
        with h5py.File(path, 'r') as hf:
            # The tensor's 210 columns are W=21 window positions x L=10 signal
            # samples, so the CENTRE (the position being called) starts at
            # half_window * L = 100. NOT center_idx * L: center_idx is the k-mer
            # centre used for the level-table lookup, a different quantity that
            # happens to also index columns. Reading center_idx * L returns a
            # base ~4 positions off-centre, i.e. background composition
            # (verified: uniform 25% ACGT vs the true centre's 50/50 A/T on
            # Ecoli_DM GATC).
            L = int(hf.attrs['L'])
            cs = int(hf.attrs['half_window']) * L
            onehot = hf['tensors'][local[order].tolist(), 0, cs, 2:6]   # (k,4)
        out[where[order]] = bases[np.argmax(onehot, axis=1)]
    return out


def lomo_split_pipeline1(pool, members: dict, mod: str):
    """
    Leave-one-modification-out over the full pool, using deepmod_full_pipeline1's
    test definition so the bars are directly comparable to results6.

    Returns (train_idx, test_sets); test_sets is a list of (name, indices):
      ONT_heldout_<mod>    — that modification's whole ONT file (if one exists;
                             5hmU has none, so 5hmU keeps all ONT in training)
      UMCES_heldout_<mod>  — R.umces_lomo_split(): {modified & type==mod}
                             u {unmodified & in the held-out region}
    Everything not tested and not carrying modification `mod` stays in training.
    """
    pool_off = np.concatenate([[0], np.cumsum(pool.file_sizes)])

    def member_range(member_name):
        fi = pool.name_index(member_name)
        return int(pool_off[fi]), int(pool_off[fi + 1])

    drop = np.zeros(pool.N, dtype=bool)        # True = withheld from training
    test_sets = []

    # ── ONT: drop that modification's file entirely; test on all of it ────────
    if mod in R.ONT_FILES and mod != 'control':
        lo, hi = member_range(f'ONT::{mod}')
        drop[lo:hi] = True
        test_sets.append((f'ONT_heldout_{mod}', np.arange(lo, hi, dtype=np.int64)))

    # ── UMCES: pipeline1's region-clean type split, mapped into pool indices ──
    umc = R.Group(list(R.UMCES_ORDER), R.UMCES_FILES)
    mod_map = build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    _, umc_te = R.umces_lomo_split(umc, mod_map, mod)
    umc_off = np.concatenate([[0], np.cumsum(umc.file_sizes)])
    te_pool = np.empty(len(umc_te), dtype=np.int64)
    for j, nm in enumerate(umc.names):         # umc uses pipeline1's bare names
        plo, _ = member_range(f'UMCES::{nm}')
        ulo, uhi = int(umc_off[j]), int(umc_off[j + 1])
        m = (umc_te >= ulo) & (umc_te < uhi)
        te_pool[m] = umc_te[m] - ulo + plo
    # umces_lomo_split partitions UMCES, so dropping the test half leaves
    # exactly its train half in the pool.
    drop[te_pool] = True
    test_sets.append((f'UMCES_heldout_{mod}', np.sort(te_pool)))

    # ── bacteria/arabidopsis: withhold this modification's positives from TRAIN
    bases = LOMO_REF_BASES.get(mod)            # None for 5hmC/5hmU (absent there)
    if bases is not None:
        pos = []
        for key, mm in members.items():
            if key in ('UMCES', 'ONT'):
                continue
            for nm in mm:
                lo, hi = member_range(nm)
                pos.append(np.nonzero(pool.labels[lo:hi] > 0)[0] + lo)
        pos = np.concatenate(pos) if pos else np.zeros(0, dtype=np.int64)
        if len(pos):
            rb = compute_ref_base_for(pool, pos)
            drop[pos[np.isin(rb, list(bases))]] = True

    train_idx = np.nonzero(~drop)[0].astype(np.int64)
    return train_idx, test_sets


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', required=True,
                    help="'mixed', 'lodo_<dataset-name>', 'lomo_6mA', or 'lomo_5mC'")
    ap.add_argument('--out-dir', default=str(Path(__file__).resolve().parent / 'results1'))
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()

    manifest_train_files, test_files = load_manifest()
    members = load_train_dataset_members(manifest_train_files)
    train_names = sorted(manifest_train_files) + ['UMCES', 'ONT']
    valid_folds = ['mixed'] + [f'lodo_{n}' for n in train_names] + [f'lomo_{m}' for m in LOMO_MODS]
    if args.fold not in valid_folds:
        raise SystemExit(f"--fold must be one of {valid_folds}, got {args.fold!r}")

    hp = R.HP()
    if args.epochs:
        hp.epochs = args.epochs
    R.set_seed(hp.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out_dir)
    (out / 'models').mkdir(parents=True, exist_ok=True)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}   fold={args.fold}   out={out}", flush=True)

    # preflight: consistent tensor shape across every file we might touch
    all_paths = ([p for m in members.values() for p in m.values()]
                 + list(test_files.values()))
    shapes = set()
    for p in all_paths:
        with h5py.File(p, 'r') as hf:
            shapes.add(tuple(hf['tensors'].shape[1:]))
    if len(shapes) != 1:
        raise SystemExit(f"PREFLIGHT: inconsistent tensor shapes across H5s: {shapes}")
    print(f"Preflight OK — tensor shape {shapes.pop()} across {len(all_paths)} files", flush=True)

    wandb_run = wandb_init(args.fold, hp, train_names, sorted(test_files))

    if args.fold == 'mixed':
        json.dump({
            'train_role_datasets': train_names,
            'test_role_datasets': sorted(test_files),
            'lomo_mods': list(LOMO_MODS),
            'hyperparams': {k: getattr(R.HP, k) for k in dir(R.HP) if not k.startswith('_')},
        }, open(out / 'config.json', 'w'), indent=2)
        print(f"Wrote {out / 'config.json'}", flush=True)

    model_factory = lambda: ConvFormerV2(dropout=hp.dropout)
    rows = []

    def eval_and_record(model, run, test_name, g, idx, held_out=''):
        m = R.evaluate(model, g, idx, device, hp)
        rows.append({'fold': args.fold, 'run': run, 'test_set': test_name,
                     'held_out': held_out, **m})
        print(f"  EVAL {run} on {test_name}: micro_f1={m['micro_f1']:.3f} "
              f"mod_f1={m['mod_f1']:.3f} mod_rec={m['mod_rec']:.3f} "
              f"auprc={m['auprc']:.3f}", flush=True)
        if wandb_run is not None:
            wandb_run.summary.update({f'eval/{test_name}/{k}': v
                                      for k, v in m.items()})

    def eval_external(model, run):
        for name, path in sorted(test_files.items()):
            g = R.Group([name], {name: path})
            eval_and_record(model, run, f'external_{name}', g,
                            np.arange(g.N, dtype=np.int64), held_out=name)

    if args.fold == 'mixed':
        pool_names, pool_files = flatten(train_names, members)
        pool = R.Group(pool_names, pool_files)
        train_idx, _, test_idx, stats = split_position_groups(
            pool.labels, pool.position_keys, val_frac=0.0, test_frac=0.15, seed=hp.seed)
        R.assert_disjoint(train_idx, test_idx, pool, 'mixed-position-split')
        print(f"  {stats}", flush=True)

        model = R.train_one_model(pool, train_idx, hp, device, out / 'models' / 'mixed',
                                  'base', model_factory=model_factory)
        eval_and_record(model, 'base', 'held_out_test', pool, test_idx)
        for name in train_names:            # per-source breakdown of the held-out test set
            member_fis = {pool.name_index(m) for m in members[name]}
            sel = test_idx[np.isin(pool.file_of[test_idx], list(member_fis))]
            if len(sel):
                eval_and_record(model, 'base', f'held_out_test_{name}', pool, sel,
                                held_out=name)
        eval_external(model, 'base')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    elif args.fold.startswith('lodo_'):
        held = args.fold[len('lodo_'):]
        other_keys = [n for n in train_names if n != held]
        tr_names, tr_files = flatten(other_keys, members)
        te_names, te_files = flatten([held], members)
        g_tr = R.Group(tr_names, tr_files)
        g_te = R.Group(te_names, te_files)
        model = R.train_one_model(g_tr, np.arange(g_tr.N, dtype=np.int64), hp, device,
                                  out / 'models' / args.fold, 'lodo',
                                  model_factory=model_factory)
        eval_and_record(model, 'lodo', f'held_out_{held}', g_te,
                        np.arange(g_te.N, dtype=np.int64), held_out=held)
        eval_external(model, 'lodo')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    else:   # lomo_<mod> — mixed pool, pipeline1's test definition
        mod = args.fold[len('lomo_'):]
        pool_names, pool_files = flatten(train_names, members)
        pool = R.Group(pool_names, pool_files)
        print(f"  LOMO[{mod}] — building pipeline1-style split over "
              f"{pool.N:,} pool images ...", flush=True)
        train_idx, test_sets = lomo_split_pipeline1(pool, members, mod)
        for nm, idx in test_sets:
            R.assert_disjoint(train_idx, idx, pool, f'lomo-{mod}-{nm}')
        print(f"  train={len(train_idx):,}  " +
              "  ".join(f"{nm}={len(i):,}" for nm, i in test_sets), flush=True)

        model = R.train_one_model(pool, train_idx, hp, device, out / 'models' / args.fold,
                                  'lomo', model_factory=model_factory)
        for nm, idx in test_sets:
            eval_and_record(model, 'lomo', nm, pool, idx, held_out=mod)
        eval_external(model, 'lomo')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    cols = ['fold', 'run', 'test_set', 'held_out', 'micro_f1', 'mod_f1', 'unmod_f1',
            'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc', 'threshold', 'n_pos', 'n_test']
    tsv = out / 'metrics' / f'{args.fold}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(
                f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, ''))
                for c in cols) + '\n')
    print(f"\nWrote {tsv}  ({len(rows)} rows)\nDONE [{args.fold}]", flush=True)

    if wandb_run is not None:
        try:
            wandb_run.save(str(tsv))
            wandb_run.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
