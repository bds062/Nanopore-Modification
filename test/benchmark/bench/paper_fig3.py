#!/usr/bin/env python3
"""Main benchmark figure (Fig. 3): what one RawMod model does that a modification-specific model cannot.

  (a) every single model over every row: the mean AUROC over the rows (lollipop, sorted) and, in two number columns,
      the mean and the number of rows with AUROC >= 0.9. A modification-specific model sits near 0.5 on every row
      outside its modification, which pulls its mean down; RawMod is one model across all rows.
  (b) one threshold across all rows. Each tool that covers more than one modification gives one score per site with no
      knowledge of the modification (the maximum over all of its models; RawMod its single score), and the sites of
      every row are pooled (at most 1,000 positives and 1,000 negatives per row, seed 0: exactly the pooled AUROC of
      tables/benchmark_agnostic.tex). The ROC curve asks whether one cut-off works for every modification at once.

Reads bench.paper_table's outputs in --dir (paper_models.tsv, pooled_sites.tsv.gz); no rescoring.

  python -m bench.paper_fig3 --dir DIR [--name fig_benchmark_main]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from sklearn.metrics import roc_auc_score, roc_curve

from . import common as C
from .paper_curves import STY, FIGDIR, style, sty, name
from .paper_table import families

SHORT = {'rawmod': 'RawMod', 'dorado_6mA': 'Dorado\n6mA', 'dorado_4mC_5mC': 'Dorado\n4mC+5mC',
         'dorado_5mC_5hmC': 'Dorado\n5mC+5hmC', 'dorado_5mCG_5hmCG': 'Dorado\n5mCG', 'unimeth_5mC': 'UniMeth\n5mC',
         'unimeth_6mA': 'UniMeth\n6mA', 'unimeth_r81_5hmC': 'FT†\n5hmC', 'unimeth_r81_4mC': 'FT†\n4mC',
         'unimeth_r81_5hmU': 'FT†\n5hmU', 'deepmod2': 'DeepMod2\n5mCG', 'rockfish': 'Rockfish\n5mCG',
         'methynano': 'MethyNano\n5mC'}
INK, OFF, HILITE = '#1f1f1f', '#9a9a9a', '#e3ebf7'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(C.WORK / f'paper_{C.PAPER_TAG}'))
    ap.add_argument('--name', default='fig_benchmark_main' + (f'_holdout-{C.HOLDOUT}' if C.HOLDOUT else ''))
    a = ap.parse_args()
    d = Path(a.dir)
    m = pd.read_csv(d / 'paper_models.tsv', sep='\t')
    m['auroc'] = pd.to_numeric(m['auroc'], errors='coerce')
    m['own'] = m['own'].astype(str) == 'True'
    rows = list(dict.fromkeys(m['row']))
    models = [x for x in dict.fromkeys(m['model']) if m.loc[m['model'] == x, 'auroc'].notna().all()]
    pooled = pd.read_csv(d / 'pooled_sites.tsv.gz', sep='\t')
    fam = families(C.subtools())

    style()
    fig = plt.figure(figsize=(7.15, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.62, left=0.115, right=0.99, top=0.9, bottom=0.17)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) one row per model: mean AUROC over all rows, and how many rows reach 0.9
    summ = (m[m['model'].isin(models)].groupby('model')['auroc']
            .agg(mean='mean', n90=lambda x: int((x >= 0.9).sum()), n='size').sort_values('mean'))
    yy = np.arange(len(summ))
    XM, XN = 1.13, 1.30          # right edges of the two number columns (axes fraction)
    for i, (mod, r) in enumerate(summ.iterrows()):
        hot = mod.startswith('rawmod')
        col = STY['rawmod'][0] if hot else '#7a7a7a'
        ax.plot([0.5, r['mean']], [i, i], color=col, lw=2.2 if hot else 1.2, solid_capstyle='butt', zorder=2)
        ax.scatter(r['mean'], i, s=34 if hot else 20, color=col, zorder=3, edgecolor='white', lw=0.6)
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        kw = dict(va='center', ha='right', fontsize=6.8, transform=tr, color=col if hot else INK)
        ax.text(XM, i, f"{r['mean']:.3f}", **kw)
        ax.text(XN, i, f"{int(r['n90'])}/{int(r['n'])}", **kw)
    lab = {k: v.replace('\n', ' ').replace('FT†', 'UniMeth-FT†') for k, v in SHORT.items()}
    ax.set_yticks(yy)
    ax.set_yticklabels([lab.get(x, x) for x in summ.index], fontsize=6.5)
    for t, mod in zip(ax.get_yticklabels(), summ.index):
        if mod.startswith('rawmod'):
            t.set_color(STY['rawmod'][0])
    ax.set_xlim(0.45, 1.0); ax.set_ylim(-0.6, len(summ) - 0.4)
    ax.axvline(0.5, color='#9a9a9a', lw=0.6, ls=':', zorder=1)
    ax.set_xlabel(f'Mean AUROC over all {len(rows)} datasets')
    ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0, right=False)
    ax.text(XM, 1.012, 'Mean', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.8)
    ax.text(XN, 1.012, r'$\geq$0.9', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.8)
    ax.plot([1.035, XN], [1.008, 1.008], transform=ax.transAxes, color=INK, lw=0.6, clip_on=False)
    ax.plot([1.035, XN], [0.0, 0.0], transform=ax.transAxes, color=INK, lw=0.6, clip_on=False)
    ax.set_title('(a) Each model on every dataset', loc='left', pad=12)

    # (b) one threshold across all datasets: tools that cover more than one modification
    y = pooled['label'].to_numpy()
    for f in ('rawmod', 'dorado', 'unimeth', 'unimeth_r81'):
        if f not in pooled:
            continue
        s = pooled[f].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        col, _, lw = sty(f)
        k = len(fam[f])
        bx.plot(fpr, tpr, color=col, ls='-', lw=1.9 if f == 'rawmod' else 1.2, zorder=4 if f == 'rawmod' else 3,
                label=f"{name(f)} ({k} model{'s' if k > 1 else ''}): {roc_auc_score(y, s):.3f}")
    bx.plot([0, 1], [0, 1], color='#9a9a9a', lw=0.6, ls=':')
    bx.set_xlim(0, 1); bx.set_ylim(0, 1.0)
    bx.set_xlabel('False-positive rate'); bx.set_ylabel('True-positive rate', labelpad=1)
    bx.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=6.3, handlelength=1.8)
    bx.set_title('(b) One threshold across all datasets', loc='left', pad=12)

    out = d
    dirs = [out] + ([FIGDIR] if str(FIGDIR) not in ('', '.') and FIGDIR.is_dir() else [])
    for dd in dirs:
        fig.savefig(dd / f'{a.name}.pdf', bbox_inches='tight', pad_inches=0.02)
        fig.savefig(dd / f'{a.name}.png', dpi=250, bbox_inches='tight', pad_inches=0.02)
    m[m['model'].isin(models)].to_csv(out / f'{a.name}_strip.tsv', sep='\t', index=False, float_format='%.4f')
    print('wrote ' + ' and '.join(str(dd / f'{a.name}.pdf') for dd in dirs))


if __name__ == '__main__':
    main()
