#!/usr/bin/env python3
"""Embed the newly-featurized non-motif background sites (real unmod
negatives for the 6 motif-saturated bacterial datasets, see
pipeline/generate_background_sites.py + experiments/pipeline4/
featurize_background.py) and APPEND them to the existing
embeddings_allorg.npz, so the combined figure shows every organism with
its genuine mod/unmod split instead of the 6 bacterial datasets being
100% "mod" islands.

Background points get type='unmod' directly (no chemistry applies to an
unmodified site regardless of base identity) and idx=-1 (they don't come
from the original matched-loco pool, so there's no (contig,ref_pos) to
join back to it -- recompute_bench_types.py's coordinate lookup doesn't
apply to them and shouldn't be run on them).

Usage: python embed_background_sites.py --ckpt <best_model.pt> --npz <embeddings_allorg.npz>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
for _p in (REPO / 'analysis' / 'orca_remake', REPO / 'scripts' / 'train',
           REPO / 'scripts' / 'train', REPO / 'rawmod', REPO / 'scripts' / 'test'):
    sys.path.insert(0, str(_p))

from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402
from score_genome import load_model                                    # noqa: E402

SEED = 0
MAX_PER_GROUP = 200
BG_ROOT = '/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4/features/benchmark'
BG_ORGS = ['Anabaena_WT_5kHz', 'Ecoli_DM_5kHz', 'Ecoli_DM_MSssI_5kHz', 'Ecoli_WT_5kHz',
          'Tdenticola_WT_5kHz', 'HPJ99_WT_5kHz']


def extract_rep(h5_path, n, idx, device, ckpt):
    ds = PileupDataset([h5_path], np.asarray(idx, np.int64), np.array([n], dtype=np.int64),
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(512, 6, device, _worker_init_fn))
    model, arch = load_model(ckpt, device)
    cap = {}
    target = model.head[3] if arch == 'inception' else model.head
    hook = target.register_forward_hook(lambda _m, inp, _o: cap.__setitem__('e', inp[0].detach()))
    reps = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device, non_blocking=True))
            reps.append(cap['e'].float().cpu().numpy())
    hook.remove()
    return np.concatenate(reps, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--npz', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import h5py
    all_rep, all_org, all_type, all_mod = [], [], [], []
    for org in BG_ORGS:
        h5_path = f'{BG_ROOT}/{org}_background/features.h5'
        with h5py.File(h5_path, 'r') as hf:
            n = hf['tensors'].shape[0]
        sel = rng.choice(n, min(MAX_PER_GROUP, n), replace=False)
        sel = np.sort(sel)
        print(f"{org}: embedding {len(sel):,} background sites (of {n:,})", flush=True)
        rep = extract_rep(h5_path, n, sel, device, a.ckpt)
        all_rep.append(rep)
        all_org.extend([org] * len(sel))
        all_type.extend(['unmod'] * len(sel))
        all_mod.extend(['unmod'] * len(sel))

    rep_new = np.concatenate(all_rep, 0)
    orgs_new = np.array(all_org)
    types_new = np.array(all_type)
    mods_new = np.array(all_mod)
    idx_new = np.full(len(orgs_new), -1, dtype=np.int64)

    npz_path = Path(a.npz)
    d = np.load(npz_path, allow_pickle=True)
    rep, orgs, types, mods, idx = d['rep'], d['orgs'], d['types'], d['mods'], d['idx']

    backup = npz_path.with_name('embeddings_allorg.pre_background.npz')
    if not backup.exists():
        np.savez_compressed(backup, rep=rep, orgs=orgs, types=types, mods=mods, idx=idx)
        print(f"backed up pre-background npz -> {backup}")

    rep_all = np.concatenate([rep.astype(np.float16), rep_new.astype(np.float16)], 0)
    orgs_all = np.concatenate([orgs, orgs_new])
    types_all = np.concatenate([types, types_new])
    mods_all = np.concatenate([mods, mods_new])
    idx_all = np.concatenate([idx, idx_new])

    np.savez_compressed(npz_path, rep=rep_all, orgs=orgs_all, types=types_all,
                        mods=mods_all, idx=idx_all)
    print(f"\nappended {len(orgs_new):,} background points; total now {len(orgs_all):,}")
    print(f"wrote {npz_path}")


if __name__ == '__main__':
    main()
