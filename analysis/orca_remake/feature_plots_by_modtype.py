#!/usr/bin/env python3
"""
Visualize the raw featurization channels directly (no model involved) --
mod vs unmod, broken out by modification type AND by organism/dataset. Small
balanced sample per (type, organism) group, same pool/selection machinery as
the embedding clustering scripts.

Masking note: per-position (row, col), not just per-row. A read can have real
signal for only PART of the 21-column window (e.g. near a construct/contig
edge) -- the uncovered columns within that row are filled with 0 across every
channel. Masking only at the row level (as organism_feature_importance.py
does) lets those zero-filled columns dilute the mean; masking at (row, col)
avoids that. This mattered concretely: with row-level masking, `strand_mean`
looked variable for unmod/ONT (as low as 0.49) even though the +strand-only
filter has no leaked reverse reads (verified directly against the raw H5 --
no -1 values appear anywhere) -- the variance was coming from partial-window
reads being averaged in with their real, but zero-filled, empty columns.

Uses the CURRENT strand-split (RAWMOD_DATA_GEN=strand15) featurization by
default, since that's the data every recent results dir trains on.

Usage: python feature_plots_by_modtype.py --out-dir <dir> [--per-group N]
"""
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault('RAWMOD_DATA_GEN', 'strand15')

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
for _p in (REPO / 'analysis' / 'orca_remake', REPO / 'experiments' / 'pipeline4',
           REPO / 'experiments' / 'pipeline1', REPO / 'deepmod'):
    sys.path.insert(0, str(_p))

import run_matched_loco as ML                                   # noqa: E402
from run_matched_loco import R                                  # noqa: E402
from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402
import sad_embedding_cluster as SEC                              # noqa: E402
from sad_embedding_cluster import TYPE_COLOR, TYPE_ORDER, ORG_COLOR, ORG_ORDER  # noqa: E402

CHANNEL_NAMES = ['raw_signal', 'dwell_log1p', 'is_A', 'is_C', 'is_G', 'is_T',
                 'strand', 'mapq_norm', 'matches_ref', 'center_delta', 'window_delta']
SEED = 0
MOD_COLOR = {'mod': '#D55E00', 'unmod': '#999999'}


def compute_features(pool, idx, batch=256, workers=4):
    """Per-image feature vector: [n_reads, then mean+std of each channel],
    masked at the (row, col) level -- a window position only counts if THAT
    read has real signal there (raw_signal != 0), not just if the read's row
    has ANY real signal anywhere in the window. Ref row (row 0) always kept
    in full."""
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(batch, workers, torch.device('cpu'), _worker_init_fn))
    feats = []
    n_ch = len(CHANNEL_NAMES)
    for xb, _ in loader:
        xb = xb.numpy()
        B, C, H, W = xb.shape
        keep2d = (np.abs(xb[:, 0, :, :]) > 1e-6)          # (B,H,W) per-position validity
        keep2d[:, 0, :] = True                             # ref row always kept
        row_has_any = keep2d.any(axis=2)                   # (B,H) -- did this read have ANY real position?
        n_reads = row_has_any[:, 1:].sum(axis=1)
        row = [n_reads]
        denom = np.maximum(keep2d.sum(axis=(1, 2)), 1.0)
        for c in range(min(n_ch, C)):
            vals = xb[:, c, :, :]
            mean = (vals * keep2d).sum(axis=(1, 2)) / denom
            var = ((vals - mean[:, None, None]) ** 2 * keep2d).sum(axis=(1, 2)) / denom
            row.append(mean)
            row.append(np.sqrt(np.maximum(var, 0)))
        feats.append(np.stack(row, axis=1))
    return np.concatenate(feats, axis=0)


def _box(ax, data, labels, colors, positions=None, width=0.6):
    bp = ax.boxplot(data, positions=positions, tick_labels=labels, showfliers=False,
                    patch_artist=True, widths=width, medianprops=dict(color='black'))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.78)
    return bp


def box_grid_by_type(X, feat_idx_map, types, stat, out_path, per_group_n):
    n = len(CHANNEL_NAMES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), dpi=130)
    axes = axes.ravel()
    present = [t for t in TYPE_ORDER if (types == t).any()]
    for ci, ch in enumerate(CHANNEL_NAMES):
        ax = axes[ci]
        col = feat_idx_map[f'{ch}_{stat}']
        data = [X[types == t, col] for t in present]
        _box(ax, data, present, [TYPE_COLOR[t] for t in present])
        ax.axhline(0, color='black', lw=1.0, ls='-', zorder=1, alpha=0.6)
        ax.set_title(f'{ch}  ({stat})', fontsize=10)
        ax.tick_params(axis='x', labelrotation=40, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', color='#eee', lw=0.8, zorder=0)
    for ci in range(n, len(axes)):
        axes[ci].axis('off')
    fig.suptitle(f'Raw featurization channels ({stat} over reads, masked) by TRUE '
                f'modification type -- n<={per_group_n}/type/organism group, '
                f'strand-split data', fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def box_grid_by_organism(X, feat_idx_map, orgs, is_mod, stat, out_path, per_group_n):
    """Same channel-panel grid, but x-axis = organism, with mod/unmod as a
    paired sub-box within each organism cluster."""
    n = len(CHANNEL_NAMES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.8 * nrows), dpi=130)
    axes = axes.ravel()
    present_orgs = [o for o in ORG_ORDER if (orgs == o).any()]
    xw = 0.35
    for ci, ch in enumerate(CHANNEL_NAMES):
        ax = axes[ci]
        col = feat_idx_map[f'{ch}_{stat}']
        data, colors, positions = [], [], []
        for oi, org in enumerate(present_orgs):
            for mi, mtag in enumerate(['unmod', 'mod']):
                m = (orgs == org) & (is_mod == mtag)
                if not m.any():
                    continue
                data.append(X[m, col])
                colors.append(MOD_COLOR[mtag])
                positions.append(oi + (mi - 0.5) * xw * 1.3)
        _box(ax, data, [None] * len(data), colors, positions=positions, width=xw)
        ax.set_xticks(range(len(present_orgs)))
        ax.set_xticklabels(present_orgs, fontsize=9)
        ax.axhline(0, color='black', lw=1.0, ls='-', zorder=1, alpha=0.6)
        ax.set_title(f'{ch}  ({stat})', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', color='#eee', lw=0.8, zorder=0)
    for ci in range(n, len(axes)):
        axes[ci].axis('off')
    from matplotlib.patches import Patch
    handles = [Patch(fc=MOD_COLOR['unmod'], alpha=0.78, label='unmod'),
              Patch(fc=MOD_COLOR['mod'], alpha=0.78, label='mod (any chemistry)')]
    fig.legend(handles=handles, loc='upper right', fontsize=9, ncol=2,
              bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f'Raw featurization channels ({stat} over reads, masked) by ORGANISM/dataset '
                f'-- mod vs unmod -- n<={per_group_n}/type/organism group, strand-split data',
                fontsize=12, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def highlight_panel(X, feat_idx_map, group_labels, group_order, colors, out_path,
                    title, per_group_n, group_kind='type'):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    present = [g for g in group_order if (group_labels == g).any()]
    col = feat_idx_map['center_delta_mean']
    data = [X[group_labels == g, col] for g in present]
    _box(ax, data, present, [colors[g] for g in present])
    ax.axhline(0, color='black', lw=1.2, ls='-', zorder=1, alpha=0.7)
    ax.set_ylabel('center_delta (mean over reads)\n= per-read signal deviation from '
                  'expected level AT the site', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', color='#eee', lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def coverage_panel(X, feat_idx_map, group_labels, group_order, colors, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    present = [g for g in group_order if (group_labels == g).any()]
    col = feat_idx_map['n_reads']
    data = [X[group_labels == g, col] for g in present]
    _box(ax, data, present, [colors[g] for g in present])
    ax.axhline(0, color='black', lw=1.2, ls='-', zorder=1, alpha=0.7)
    ax.set_ylabel('n_reads (coverage)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis='y', color='#eee', lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--per-group', type=int, default=150,
                    help='cap per (type, organism) group (small sample)')
    a = ap.parse_args()
    out = Path(a.out_dir); (out / 'figures').mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    SEC.MAX_PER_GROUP = a.per_group

    members = ML.build_members()
    pool = R.Group(list(members), members)
    print(f"Matched pool: {pool.N:,} images across {len(members)} files "
         f"(RAWMOD_DATA_GEN={os.environ.get('RAWMOD_DATA_GEN')})", flush=True)
    mod_map = ML.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = ML.ref_base_center(pool)
    chem = ML.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0

    idx, types, orgs = SEC.build_selection(pool, chem, is_pos, refbase, rng)
    is_mod = np.array(['unmod' if t == 'unmod' else 'mod' for t in types])
    print(f"\nSelected {len(idx):,} images (cap={a.per_group}/group): "
         f"{dict(Counter(types))}", flush=True)
    print("organism breakdown per type:")
    for t in TYPE_ORDER:
        m = types == t
        if m.any():
            print(f"  {t:6}: {dict(Counter(orgs[m]))}")

    print("\nComputing masked per-channel summary features (CPU, no model, "
         "(row,col)-level masking)...", flush=True)
    X = compute_features(pool, idx)
    feat_names = ['n_reads'] + [f'{c}_{s}' for c in CHANNEL_NAMES for s in ('mean', 'std')]
    feat_idx_map = {name: i for i, name in enumerate(feat_names)}
    print(f"  X shape {X.shape}", flush=True)
    np.savez_compressed(out / 'raw_features.npz', X=X.astype(np.float32), types=types,
                        orgs=orgs, is_mod=is_mod, idx=idx.astype(np.int64),
                        feat_names=np.array(feat_names))

    box_grid_by_type(X, feat_idx_map, types, 'mean',
                     out / 'figures' / 'fig_channel_means_by_type.png', a.per_group)
    box_grid_by_type(X, feat_idx_map, types, 'std',
                     out / 'figures' / 'fig_channel_stds_by_type.png', a.per_group)
    box_grid_by_organism(X, feat_idx_map, orgs, is_mod, 'mean',
                         out / 'figures' / 'fig_channel_means_by_organism.png', a.per_group)
    box_grid_by_organism(X, feat_idx_map, orgs, is_mod, 'std',
                         out / 'figures' / 'fig_channel_stds_by_organism.png', a.per_group)

    highlight_panel(X, feat_idx_map, types, TYPE_ORDER, TYPE_COLOR,
                    out / 'figures' / 'fig_center_delta_highlight.png',
                    'Average deviation from reference AT the modification site,\n'
                    f'unmod vs each modification type (n<={a.per_group}/type/organism)',
                    a.per_group)
    highlight_panel(X, feat_idx_map, orgs, ORG_ORDER, ORG_COLOR,
                    out / 'figures' / 'fig_center_delta_highlight_by_organism.png',
                    'Average deviation from reference AT the modification site,\n'
                    f'by organism/dataset (mod+unmod pooled, n<={a.per_group}/type/organism)',
                    a.per_group)

    coverage_panel(X, feat_idx_map, types, TYPE_ORDER, TYPE_COLOR,
                  out / 'figures' / 'fig_coverage_by_type.png',
                  'Read coverage per image, unmod vs each modification type')
    coverage_panel(X, feat_idx_map, orgs, ORG_ORDER, ORG_COLOR,
                  out / 'figures' / 'fig_coverage_by_organism.png',
                  'Read coverage per image, by organism/dataset')

    print(f"\nwrote {out / 'raw_features.npz'}")


if __name__ == '__main__':
    main()
