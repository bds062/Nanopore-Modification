#!/usr/bin/env python3
"""Draw a genome map from score_genome.py output: windowed mean RawMod score per
model along one contig, with an optional gene track and the read-coverage track.

Scores are model probabilities, not calibrated modification fractions, and
different checkpoints are calibrated differently -- read the map as RELATIVE
(window vs. the contig median), and compare models by rank, not by value.

  python make_genome_map.py \
      --scores mixed=OUT/scores/NAME_mixed_strand+_scores.tsv.gz \
               loco5hmU=OUT/scores/NAME_loco5hmU_strand+_scores.tsv.gz \
      --contig contig_3762 --gff annotations.gff3 --window 10000 \
      --out OUT/genome_map.png

Also writes <out>.windows.tsv (window start, per-model mean, coverage, n sites).
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

COLORS = ['#2C7FB8', '#D7191C', '#4DAF4A', '#984EA3']
GREY = '#555555'


def read_genes(path, contig):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) >= 7 and p[2] == 'gene' and (contig is None or p[0] == contig):
                rows.append((int(p[3]) - 1, int(p[4]), p[6]))
    return pd.DataFrame(rows, columns=['start', 'end', 'strand'])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores', nargs='+', required=True, help='name=path.tsv.gz (1-4 models)')
    ap.add_argument('--contig', default=None, help='contig to draw (default: the first in the file)')
    ap.add_argument('--gff', default=None, help='GFF3 with gene features (optional)')
    ap.add_argument('--window', type=int, default=10_000)
    ap.add_argument('--min-sites', type=int, default=20, help='drop windows with fewer sites')
    ap.add_argument('--title', default='RawMod genome map')
    ap.add_argument('--out', required=True, help='output .png (or .pdf)')
    a = ap.parse_args()

    names, frames = [], []
    for spec in a.scores[:4]:
        name, path = spec.split('=', 1)
        d = pd.read_csv(path, sep='\t')
        contig = a.contig or d.contig.iloc[0]
        d = d[d.contig == contig]
        frames.append(d[['pos', 'score', 'n_reads']].rename(
            columns={'score': name, 'n_reads': f'n_{name}'}))
        names.append(name)
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on='pos')                       # models compared on shared sites only
    df['n_reads'] = df[f'n_{names[0]}']
    print(f'{contig}: {len(df):,} sites scored by all of {names}')

    df['win'] = df.pos // a.window
    w = df.groupby('win').agg(**{n: (n, 'mean') for n in names},
                              covg=('n_reads', 'mean'), n=('pos', 'size'))
    w = w[w.n >= a.min_sites].astype(float)
    w['mb'] = w.index * a.window / 1e6
    if len(names) > 1:
        print('window Spearman vs', names[0], {n: round(w[names[0]].corr(w[n], method='spearman'), 3)
                                               for n in names[1:]})
    tsv = a.out.rsplit('.', 1)[0] + '.windows.tsv'
    w.assign(start=w.index.astype(int) * a.window).to_csv(tsv, sep='\t', index=False)

    genes = read_genes(a.gff, contig) if a.gff else None
    nrow = len(names) + 1 + (genes is not None)
    ratios = [3] * len(names) + ([0.75] if genes is not None else []) + [1.4]
    fig = plt.figure(figsize=(16, 2.6 * len(names) + 3), dpi=160)
    gs = GridSpec(nrow, 1, height_ratios=ratios, hspace=0.16)
    axes = [fig.add_subplot(gs[i]) for i in range(nrow)]
    xlim = (0, (df.pos.max() + 1) / 1e6)

    for ax, n, col in zip(axes, names, COLORS):
        med = w[n].median()
        ax.fill_between(w.mb, med, w[n], where=w[n] >= med, color=col, alpha=0.85, lw=0, interpolate=True)
        ax.fill_between(w.mb, med, w[n], where=w[n] < med, color=col, alpha=0.28, lw=0, interpolate=True)
        ax.plot(w.mb, w[n], color=col, lw=0.6)
        ax.axhline(med, color='#333', lw=0.9, ls=(0, (5, 4)))
        ax.set_ylabel(f'mean score\n({a.window // 1000} kb window)', fontsize=11, fontweight='bold')
        ax.text(0.008, 0.93, n, transform=ax.transAxes, fontsize=13, fontweight='bold',
                color=col, va='top')
    k = len(names)
    if genes is not None:
        ax = axes[k]
        for g in genes.itertuples():
            ax.add_patch(Rectangle((g.start / 1e6, 0.55 if g.strand == '+' else 0.05),
                                   max((g.end - g.start) / 1e6, xlim[1] / 5000), 0.4, color=GREY, lw=0))
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.75]); ax.set_yticklabels(['gene (−)', 'gene (+)'])
        ax.text(0.008, 1.06, f'genes (n={len(genes)})', transform=ax.transAxes, fontsize=11,
                fontweight='bold', color=GREY, va='bottom')
        for s in ('left',):
            ax.spines[s].set_visible(False)
        k += 1
    ax = axes[k]
    ax.fill_between(w.mb, 0, w.covg, color='#7a7a7a', alpha=0.55, lw=0)
    ax.set_ylim(0, w.covg.quantile(0.999) * 1.05)
    ax.set_ylabel('mean reads\nper site', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{contig} position (Mb)', fontsize=13, fontweight='bold')
    for i, ax in enumerate(axes):
        ax.set_xlim(*xlim)
        ax.tick_params(length=0, labelbottom=(i == nrow - 1))
        for s in ('top', 'right') + (('bottom',) if i < nrow - 1 else ()):
            ax.spines[s].set_visible(False)
        ax.yaxis.grid(True, color='#ececec', lw=0.9); ax.set_axisbelow(True)
    axes[0].set_title(a.title, fontsize=17, fontweight='bold', pad=10, loc='left')
    fig.savefig(a.out, bbox_inches='tight')
    print('wrote', a.out, 'and', tsv)


if __name__ == '__main__':
    main()
