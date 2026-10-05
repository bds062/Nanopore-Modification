#!/usr/bin/env python3
"""ROC curves for the paper benchmark figure (fig:benchmark), one panel per selected row, one curve per TOOL.

A tool's curve uses the matched rule of tables/benchmark.tex (bench.paper_table): only the models it ships for the row's
modification (config/claims.tsv), site score = their maximum, CpG-only tools on mixed-context rows scored on the row's
CpG sites only. Tools with no model for the row's modification are not drawn (they are N/A in the table). Curves are
interpolated onto a log-spaced FPR grid and drawn on a log FPR axis (1e-3..1), where callers with
AUROC > 0.99 still separate; the AUROC in each legend is computed on the full site set and is
identical to the matched table.

Writes <out>/fig_benchmark_roc.{pdf,png}, <out>/roc_curves.tsv (row, tool, fpr, tpr) and <out>/roc_rows.tsv
(row, tool, n_sites, n_pos, auroc).

  python -m bench.paper_curves [--out DIR] [--rows r1,r2,...]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_auc_score, roc_curve

from . import common as C
from . import score as Sc
from . import settings as ST
from .paper_table import TOOLS, GROUPS, claims, row_marks, family_score, is_cpg, families

# optional second output folder for the figures (e.g. a manuscript's figures/ directory); unset = output folder only
FIGDIR = Path(ST.get('BENCH_PAPER_FIGDIR') or '.')
ROWS = ['rice_cpg', 'mouse_cpg', 'hg002_cpg', 'ecoli_dcm_5mC_ko', 'syn_5hmC', 'ecoli_dam_6mA_ko', 'hp26695_4mC',
        'spo1_5hmU']
TITLE = {'rice_cpg': ('5mC CpG', r'$\it{O.\ sativa}$'), 'mouse_cpg': ('5mC CpG', 'Mouse brain'),
         'hg002_cpg': ('5mC CpG', 'HG002'), 'ecoli_dcm_5mC_ko': ('5mC non-CpG', r'$\it{E.\ coli}$ Dcm vs. KO'),
         'syn_5hmC': ('5hmC', 'ONT Benchmark'), 'ecoli_dam_6mA_ko': ('6mA', r'$\it{E.\ coli}$ Dam vs. KO'),
         'hp26695_4mC': ('4mC', r'$\it{H.\ pylori}$ TCTTC vs. WGA'),
         'hp26695_6mA': ('6mA', r'$\it{H.\ pylori}$ GCATG vs. WGA'), 'spo1_5hmU': ('5hmU', 'SPO1 native vs. PCR'),
         'arabidopsis_cpg': ('5mC CpG', r'$\it{A.\ thaliana}$'), 'arabidopsis_noncpg': ('5mC non-CpG', r'$\it{A.\ thaliana}$'),
         'ecoli_mssi_5mC_ko': ('5mC CpG', r'$\it{E.\ coli}$ M.SssI vs. KO'), 'syn_5mC': ('5mC', 'ONT Benchmark'),
         'syn_6mA': ('6mA', 'ONT Benchmark'), 'anabaena_6mA': ('6mA', r'$\it{Anabaena}$'),
         'hpj99_4mC_pb': ('4mC', r'$\it{H.\ pylori}$ J99'), 'hpj99_6mA_pb': ('6mA', r'$\it{H.\ pylori}$ J99'),
         'tden_4mC_pb': ('4mC', r'$\it{T.\ denticola}$'), 'tden_6mA_pb': ('6mA', r'$\it{T.\ denticola}$')}
# figure palette; line style is a second channel so identity is not colour-only
STY = {'rawmod': ('#4b71bb', '-', 1.8), 'dorado': ('#db8043', '-', 1.1), 'unimeth': ('#5FA137', '--', 1.1),
       'unimeth_r81': ('#2f6b1f', '-', 1.1), 'deepmod2': ('#aa4499', '-.', 1.1), 'rockfish': ('#f0c041', '--', 1.1),
       'methynano': ('#a3a3a3', ':', 1.1)}
NAME = {'rawmod': 'RawMod', 'dorado': 'Dorado', 'unimeth': 'UniMeth', 'unimeth_r81': 'UniMeth-FT†', 'deepmod2': 'DeepMod2',
        'rockfish': 'Rockfish', 'methynano': 'MethyNano'}
EXTRA_RAWMOD = ['#1f3f7a', '#7fa3e0', '#3b5f9f', '#9fb9e8']   # further RawMod checkpoints (BENCH_RAWMOD), blue family


def sty(f):
    if f in STY:
        return STY[f]
    i = [t for t, _ in TOOLS if t.startswith('rawmod_')].index(f) if f.startswith('rawmod_') else 0
    return (EXTRA_RAWMOD[i % len(EXTRA_RAWMOD)], '--', 1.5)


def name(f):
    return NAME.get(f) or ('RawMod (' + f.removeprefix('rawmod_') + ')' if f.startswith('rawmod_') else f)


LEG_UL = {'hp26695_4mC', 'hp26695_6mA', 'spo1_5hmU', 'arabidopsis_noncpg'}   # legend would cover the curves
GRID = np.r_[0.0, np.logspace(-3, 0, 500)]   # log-spaced FPR: the near-perfect callers differ only below 1e-2


def style():
    fonts = ST.get('BENCH_FONTS')                  # optional folder of .ttf files (the paper used Times New Roman)
    for f in glob.glob(str(Path(fonts) / '*.[tT][tT][fF]')) if fonts else []:
        fm.fontManager.addfont(f)
    serif = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    tnr = any(f.name == 'Times New Roman' for f in fm.fontManager.ttflist)
    fs = 8
    plt.rcParams.update({
        'figure.facecolor': 'white', 'font.family': serif, 'font.weight': 'bold',
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'font.size': fs, 'axes.titlesize': fs,
        'axes.labelsize': fs, 'xtick.labelsize': fs - 1, 'ytick.labelsize': fs - 1, 'legend.fontsize': fs - 2.5,
        'mathtext.fontset': 'custom' if tnr else 'dejavuserif', 'mathtext.rm': 'Times New Roman:bold',
        'mathtext.it': 'Times New Roman:bold:italic', 'mathtext.bf': 'Times New Roman:bold', 'axes.linewidth': 0.8, 'axes.edgecolor': 'black',
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.grid': True, 'grid.color': '#d0d0d0',
        'grid.linewidth': 0.4, 'axes.axisbelow': True, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'legend.frameon': True, 'legend.framealpha': 1.0, 'legend.edgecolor': 'black', 'legend.fancybox': False,
        'legend.handlelength': 2.0, 'legend.handletextpad': 0.4, 'legend.borderpad': 0.3, 'legend.labelspacing': 0.2})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(C.WORK / f'paper_{C.PAPER_TAG}'))
    ap.add_argument('--rows', default=','.join(ROWS), help="comma list, or 'all' = every row in table order")
    ap.add_argument('--replot', action='store_true', help='redraw from the saved roc_*.tsv, no rescoring')
    ap.add_argument('--name', default='fig_benchmark_roc' + (f'_holdout-{C.HOLDOUT}' if C.HOLDOUT else ''), help='output basename (figure, roc_*.tsv get the same suffix)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S, T, CL = C.samples(), C.subtools(), claims()
    rows = {r['row']: r for r in Sc.rows_cfg()}
    fam = families(T)
    sel = [row for _, m in GROUPS for row, _ in m] if a.rows == 'all' else a.rows.split(',')
    sfx = a.name.replace('fig_benchmark_roc', '')
    if a.replot:
        cdf = pd.read_csv(out / f'roc_curves{sfx}.tsv', sep='\t')
        sdf = pd.read_csv(out / f'roc_rows{sfx}.tsv', sep='\t')
    else:
        cache, curves, summ = {}, [], []
        for row in sel:
            r = rows[row]
            blocks = Sc.row_sites(r, S, cache)
            y = np.concatenate([b[2] for b in blocks])
            cpg = None
            for f, _ in TOOLS:
                use = [t for t in fam[f] if row_marks(r) & CL[t]['marks']
                       and not (CL[t]['context'] == 'cpg' and r['context'] == 'noncpg')]
                if not use:
                    continue
                sc = family_score(blocks, use)
                if sc is None:
                    print(f'[{row}] {f}: output pending'); continue
                keep = np.ones(len(y), bool)
                if all(CL[t]['context'] == 'cpg' for t in use) and r['context'] == 'any':
                    cpg = np.concatenate(is_cpg(blocks, S)) if cpg is None else cpg
                    keep = cpg
                yy, ss = y[keep], sc[keep]
                if not (0 < yy.sum() < len(yy)):
                    continue
                fpr, tpr, _ = roc_curve(yy, ss)
                # step-function interpolation: the curve is right-continuous in FPR
                ti = np.interp(GRID, fpr, tpr)
                curves.append(pd.DataFrame({'row': row, 'tool': f, 'fpr': GRID, 'tpr': ti}))
                auc = roc_auc_score(yy, ss)
                summ.append({'row': row, 'tool': f, 'n_sites': len(yy), 'n_pos': int(yy.sum()), 'auroc': auc,
                             'cpg_only': bool(keep.sum() < len(y))})
                print(f'[{row}] {f:11s} n={len(yy):,} pos={int(yy.sum()):,} AUROC {auc:.3f}')
        cdf, sdf = pd.concat(curves, ignore_index=True), pd.DataFrame(summ)
        cdf.to_csv(out / f'roc_curves{sfx}.tsv', sep='\t', index=False, float_format='%.5f')
        sdf.to_csv(out / f'roc_rows{sfx}.tsv', sep='\t', index=False, float_format='%.4f')

    style()
    nc = 4
    nr = int(np.ceil(len(sel) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7.15, 1.82 * nr), sharex=True, sharey=True)
    for k, (ax, row) in enumerate(zip(axes.flat, sel)):
        ax.plot(GRID[1:], GRID[1:], color='#b0b0b0', lw=0.6, ls='-', zorder=1)
        hs = []
        x = sdf[sdf['row'] == row]
        # RawMod drawn last (on top); legend in table order
        order = [f for f, _ in TOOLS if f in set(x['tool'])]
        for f in order[1:] + order[:1]:
            c = cdf[(cdf['row'] == row) & (cdf['tool'] == f)]
            col, ls, lw = sty(f)
            ax.plot(c['fpr'].iloc[1:], c['tpr'].iloc[1:], color=col, ls=ls, lw=lw, zorder=3 if f == 'rawmod' else 2)
        for f in order:
            col, ls, lw = sty(f)
            v = x[x['tool'] == f].iloc[0]
            hs.append(plt.Line2D([], [], color=col, ls=ls, lw=lw,
                                 label=f"{name(f)}{'$^{c}$' if v['cpg_only'] else ''} {v['auroc']:.3f}"))
        ax.legend(handles=hs, loc='lower right' if row not in LEG_UL else 'upper left', borderaxespad=0.3)
        mark, data = TITLE.get(row, (rows[row]['chem'], row))
        ax.set_title(f'{mark}: {data}', pad=3)
        ax.set_xscale('log'); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.02)
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'])
        ax.minorticks_off(); ax.set_yticks([0, 0.5, 1]); ax.grid(True, axis='both')
        ax.text(-0.02, 1.13, '(' + 'abcdefghijklmnop'[k] + ')', transform=ax.transAxes, fontsize=9, fontweight='bold',
                ha='right', va='bottom')
        if k % nc == 0:
            ax.set_ylabel('True positive rate')
        if k // nc == nr - 1:
            ax.set_xlabel('False positive rate (log)')
    for ax in list(axes.flat)[len(sel):]:
        ax.axis('off')
    for k in range(max(0, len(sel) - nc), len(sel)):     # bottom panel of each column carries the x axis labels
        if k // nc < nr - 1:
            ax = axes.flat[k]
            ax.xaxis.set_tick_params(labelbottom=True); ax.set_xlabel('False positive rate (log)')
    fig.tight_layout(w_pad=0.6, h_pad=0.9)
    dirs = [out] + ([FIGDIR] if str(FIGDIR) not in ('', '.') and FIGDIR.is_dir() else [])
    for d in dirs:
        fig.savefig(d / f'{a.name}.pdf', bbox_inches='tight', pad_inches=0.02)
        fig.savefig(d / f'{a.name}.png', dpi=250, bbox_inches='tight', pad_inches=0.02)
    print('wrote ' + ' and '.join(str(d / f'{a.name}.pdf') for d in dirs))


if __name__ == '__main__':
    main()
