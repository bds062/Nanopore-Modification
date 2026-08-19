#!/usr/bin/env python3
"""
What raw pileup features let a classifier identify the source DATASET/organism
(ONT/SPO1/HP)? Motivated by results7-10 (DANN memory: results7-8-dann-backfire):
organism is trivially decodable from the learned 96-d embedding (unconfounded
ARI up to 0.995) even when directly adversarially attacked. This script asks a
more basic question upstream of any learned representation: which of the 11
raw input channels (+coverage) are themselves most different across organisms,
using the EXACT same 27,900-image selection already used for the embedding/ARI
analysis (same idx, loaded from an existing embeddings.npz -- deterministic,
seed=0 build_selection).

Approach: NOT a deep-model permutation-importance (that would be confounded by
whichever model's own adversarial/representational quirks). Instead: compute
simple, fully interpretable per-image summary statistics of the RAW pileup
tensor (channel mean + std, masked to real reads only, plus read coverage),
then fit a plain multiclass classifier (Random Forest) to predict organism
from those summary stats alone. Feature importances / univariate F-scores on
this simple feature space tell us what physically differs between the
datasets, independent of any particular trained network.

11 channels (deepmod/visualization.py CHANNEL_NAMES + 2 delta channels, see
deepmod/model.py PileupDataset.__getitem__):
  0 raw_signal, 1 dwell_log1p, 2 is_A, 3 is_C, 4 is_G, 5 is_T, 6 strand,
  7 mapq_norm, 8 matches_ref, 9 center_delta, 10 window_delta

Usage: python organism_feature_importance.py --ref-embeddings <path/to/embeddings.npz> --out-dir <dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
for _p in (REPO / 'scripts' / 'train', REPO / 'scripts' / 'train', REPO / 'rawmod'):
    sys.path.insert(0, str(_p))

import run_matched_loco as ML                                   # noqa: E402
from run_matched_loco import R                                  # noqa: E402
from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402

CHANNEL_NAMES = ['raw_signal', 'dwell_log1p', 'is_A', 'is_C', 'is_G', 'is_T',
                 'strand', 'mapq_norm', 'matches_ref', 'center_delta', 'window_delta']
ORG_CLASSES = ['ONT', 'SPO1', 'HP']


def compute_features(pool, idx, batch=256, workers=6):
    """Per-image feature vector: [n_reads (coverage), then mean+std of each
    channel masked to non-padded read rows (reference row 0 always kept,
    included in the mean/std -- it's a real informative row, not padding)]."""
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    device = torch.device('cpu')
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(batch, workers, device, _worker_init_fn))
    feats = []
    n_ch = len(CHANNEL_NAMES)
    for xb, _ in loader:
        xb = xb.numpy()                                    # (B, C, H, W)
        B, C, H, W = xb.shape
        pad = np.abs(xb[:, 0]).sum(axis=2) < 1e-6           # (B, H)
        pad[:, 0] = False                                   # ref row never padding
        keep = (~pad).astype(np.float32)                    # (B, H)
        n_reads = keep[:, 1:].sum(axis=1)                   # exclude ref row from coverage count
        row = [n_reads]
        for c in range(min(n_ch, C)):
            vals = xb[:, c, :, :]                            # (B, H, W)
            w = keep[:, :, None]                             # (B, H, 1) broadcast over W
            denom = np.maximum(keep.sum(axis=1) * W, 1.0)    # kept ROWS * W columns each
            mean = (vals * w).sum(axis=(1, 2)) / denom
            var = ((vals - mean[:, None, None]) ** 2 * w).sum(axis=(1, 2)) / denom
            row.append(mean)
            row.append(np.sqrt(np.maximum(var, 0)))
        feats.append(np.stack(row, axis=1))                  # (B, 1+2*n_ch)
    return np.concatenate(feats, axis=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-embeddings', required=True,
                    help='existing embeddings.npz to reuse idx/orgs/types from (any results dir)')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    ref = np.load(a.ref_embeddings)
    idx, orgs, types = ref['idx'], ref['orgs'], ref['types']
    print(f"Reusing selection from {a.ref_embeddings}: {len(idx):,} images", flush=True)

    members = ML.build_members()
    pool = R.Group(list(members), members)
    print(f"Matched pool: {pool.N:,} images across {len(members)} files", flush=True)

    print("Computing masked per-channel summary features...", flush=True)
    X = compute_features(pool, idx)
    feat_names = ['n_reads'] + [f'{c}_{s}' for c in CHANNEL_NAMES for s in ('mean', 'std')]
    print(f"  X shape {X.shape}, features: {feat_names}", flush=True)
    np.savez_compressed(out / 'features.npz', X=X.astype(np.float32), orgs=orgs,
                        types=types, feat_names=np.array(feat_names))

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.feature_selection import f_classif, mutual_info_classif

    y = np.array([ORG_CLASSES.index(o) for o in orgs])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    rf = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight='balanced',
                                random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    acc = accuracy_score(yte, pred)
    macro_f1 = f1_score(yte, pred, average='macro')
    print(f"\nRandomForest organism classifier (3-way, from {X.shape[1]} interpretable "
          f"summary features only, no learned representation): "
          f"test acc={acc:.4f} macro_f1={macro_f1:.4f} (chance~{1/3:.3f})", flush=True)

    importances = rf.feature_importances_
    fscores, _ = f_classif(Xtr_s, ytr)
    mi = mutual_info_classif(Xtr_s, ytr, random_state=0)

    order = np.argsort(importances)[::-1]
    report = [f"RandomForest organism (3-way) test acc={acc:.4f} macro_f1={macro_f1:.4f} "
             f"(chance={1/3:.3f})\n\n",
             f"{'feature':<20s} {'RF_importance':>14s} {'ANOVA_F':>10s} {'mutual_info':>12s}\n"]
    for i in order:
        report.append(f"{feat_names[i]:<20s} {importances[i]:14.4f} {fscores[i]:10.1f} {mi[i]:12.4f}\n")
    (out / 'feature_importance.txt').write_text(''.join(report))
    print('\n' + ''.join(report))

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    top = order[:20]
    ax.barh(range(len(top)), importances[top][::-1], color='#0072B2')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([feat_names[i] for i in top][::-1], fontsize=9)
    ax.set_xlabel('Random Forest importance (predicting organism from raw pileup summary stats)')
    ax.set_title(f'What raw features identify the dataset/organism?\n'
                f'(3-way RF, {X.shape[1]} interpretable features, test acc={acc:.3f}, chance=0.333)',
                fontsize=11)
    fig.tight_layout()
    (out / 'figures').mkdir(exist_ok=True)
    fig.savefig(out / 'figures' / 'organism_feature_importance.png', bbox_inches='tight')
    print(f"\nwrote {out/'figures'/'organism_feature_importance.png'}")
    print(f"wrote {out/'feature_importance.txt'}")
    print(f"wrote {out/'features.npz'}")


if __name__ == '__main__':
    main()
