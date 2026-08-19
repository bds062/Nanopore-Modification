#!/usr/bin/env python3
"""Re-render the results15 all-organism embedding (embeddings_allorg.npz,
already computed by results15_embedding_allorg.py) with a THIRD color
scheme: modification TYPE, not just binary mod/unmod. Type = real
chemistry (5mC/5hmC/6mA/4mC/5hmU) or 'unmod' for EVERY point, including
the 7 BENCH:: organisms -- recompute_bench_types.py recovered their real
per-position chemistry from the same motifs their GT was generated from
(motif_gt.py computes the type internally but only ever wrote out
chrom+position, discarding it -- recoverable by re-matching each
candidate coordinate against its motif). No new embedding/model pass
needed here -- reuses the saved rep/types array and recomputes PCA/t-SNE
with the same seed/params as the original script.

Usage: python results15_embedding_allorg_bytype.py --npz <embeddings_allorg.npz> --out-dir <dir>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SEED = 0

TYPE_ORDER = ['5mC', '5hmC', '6mA', '4mC', '5hmU', 'unmod']
TYPE_COLOR = {'5mC': '#1b9e77', '5hmC': '#d95f02', '6mA': '#7570b3',
             '4mC': '#e7298a', '5hmU': '#66a61e', 'unmod': '#999999'}


def scatter(ax, xy, labels, order, cmap, title, info_box):
    for lab in order:
        m = labels == lab
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, color=cmap[lab],
                  label=f'{lab} (n={m.sum():,})')
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6.5, markerscale=2, loc='upper right', ncol=1)
    ax.text(0.02, 0.02, info_box, transform=ax.transAxes, fontsize=8,
           va='bottom', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#797979'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); (out / 'figures').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = np.load(a.npz, allow_pickle=True)
    rep, types = d['rep'].astype(np.float32), d['types']
    n_types = len(set(types.tolist()))
    info_box = f"n={len(rep):,}\n{n_types} types"

    Xs = (rep - rep.mean(0)) / (rep.std(0) + 1e-8)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    n_sub = min(8000, len(rep))
    sub = rng.choice(len(rep), n_sub, replace=False) if len(rep) > n_sub else np.arange(len(rep))
    tsne_in = PCA(n_components=32, random_state=SEED).fit_transform(Xs[sub]) \
        if Xs.shape[1] > 32 else Xs[sub]
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init='pca').fit_transform(tsne_in)

    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=140)
    scatter(ax, pca, types, TYPE_ORDER, TYPE_COLOR,
           'PCA colored by MODIFICATION TYPE', info_box)
    fig.suptitle('results15 (EXTRA_ORGANISMS=1) embedding: by modification type', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / 'figures' / 'fig_embedding_allorg_pca_bytype.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=140)
    scatter(ax, tsne, types[sub], TYPE_ORDER, TYPE_COLOR,
           't-SNE colored by MODIFICATION TYPE', info_box)
    fig.suptitle('results15 (EXTRA_ORGANISMS=1) embedding: by modification type', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / 'figures' / 'fig_embedding_allorg_tsne_bytype.png', dpi=150)
    plt.close(fig)

    print(f"wrote {out/'figures'/'fig_embedding_allorg_pca_bytype.png'}")
    print(f"wrote {out/'figures'/'fig_embedding_allorg_tsne_bytype.png'}")


if __name__ == '__main__':
    main()
