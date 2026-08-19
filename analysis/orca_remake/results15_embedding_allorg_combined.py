#!/usr/bin/env python3
"""Single combined figure: 2 rows (PCA, t-SNE) x 2 columns (organism,
real modification type) = 4 panels, one file. Reuses the saved
rep/orgs/types from embeddings_allorg.npz (already has real bench types
via recompute_bench_types.py) -- computes PCA and t-SNE ONCE per row and
reuses the same 2D layout across both color columns in that row, so
panels within a row are directly comparable point-for-point.

Convention (apply to future combined embedding figures too): legend only
on the bottom (t-SNE) row -- PCA/t-SNE share the same category colors, so
one legend per column is enough. Each legend entry reads
"label (n=mod/unmod)" -- the mod/unmod split within that category, not a
single total -- since that split is often the interesting fact itself
(e.g. most bench organisms are n=.../0, no unmod at all).

Usage: python results15_embedding_allorg_combined.py --npz <embeddings_allorg.npz> --out <png>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SEED = 0

ORIG_ORGS = ['ONT', 'SPO1', 'HP']
_palette = cm.get_cmap('tab10').colors + cm.get_cmap('Set3').colors

TYPE_ORDER = ['5mC', '5hmC', '6mA', '4mC', '5hmU', 'unmod']
TYPE_COLOR = {'5mC': '#1b9e77', '5hmC': '#d95f02', '6mA': '#7570b3',
             '4mC': '#e7298a', '5hmU': '#66a61e', 'unmod': '#999999'}


def scatter(ax, xy, labels, order, cmap, title, mods, show_legend):
    for lab in order:
        m = labels == lab
        if m.sum() == 0:
            continue
        n_mod = int((mods[m] == 'mod').sum())
        n_unmod = int((mods[m] == 'unmod').sum())
        ax.scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.5, color=cmap[lab],
                  label=f'{lab} (n={n_mod:,}/{n_unmod:,})', linewidths=0)
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    if show_legend:
        ax.legend(fontsize=6, markerscale=1.6, loc='best', ncol=1, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-label', default=None,
                    help='Label for the figure title, e.g. "results16_temp0.20". '
                         'Defaults to the npz grandparent dir name.')
    a = ap.parse_args()
    run_label = a.run_label or Path(a.npz).parents[1].name

    d = np.load(a.npz, allow_pickle=True)
    rep = d['rep'].astype(np.float32)
    orgs, types, mods = d['orgs'], d['types'], d['mods']
    n = len(rep)

    # ORG_ORDER/ORG_COLOR are DATA-DRIVEN (not a hardcoded org list) so a run
    # with a different or larger organism set (e.g. + human, + future
    # datasets) never silently drops points that aren't in some fixed list.
    other_orgs = sorted(set(orgs.tolist()) - set(ORIG_ORGS))
    org_order = [o for o in ORIG_ORGS if o in set(orgs.tolist())] + other_orgs
    org_color = {o: _palette[i % len(_palette)] for i, o in enumerate(org_order)}
    n_orgs = len(org_order)

    Xs = (rep - rep.mean(0)) / (rep.std(0) + 1e-8)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    tsne_in = PCA(n_components=32, random_state=SEED).fit_transform(Xs) if Xs.shape[1] > 32 else Xs
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init='pca').fit_transform(tsne_in)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), dpi=140)
    rows = [('PCA', pca), ('t-SNE', tsne)]
    cols = [(f'ORGANISM ({n_orgs})', orgs, org_order, org_color),
           ('MODIFICATION TYPE', types, TYPE_ORDER, TYPE_COLOR)]
    n_rows = len(rows)
    for r, (rname, xy) in enumerate(rows):
        for c, (cname, labels, order, cmap) in enumerate(cols):
            scatter(axes[r, c], xy, labels, order, cmap, f'{rname} — {cname}',
                   mods, show_legend=(r == n_rows - 1))

    fig.suptitle(f'{run_label} embedding — all {n_orgs} organisms, n={n:,}',
                fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
