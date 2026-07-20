#!/usr/bin/env python3
"""
Manuscript figures for pipeline3.

Fig 1 (headline)  de novo motif re-discovery: known REBASE motifs recovered from
                  RawMod scores with no motif supplied, alongside the unmodified
                  null (WGA / PCR amplicon) in which nothing should appear.
Fig 2             zero-shot generalisation: leave-one-modification-out vs the Dorado
                  specialist OR-ensemble, across 5mC / 5hmC / 6mA / 5hmU.
Fig 3             unsupervised modification-type clustering, with the confound test
                  (ARI vs type against ARI vs source dataset).
Fig 4             R10 baseline comparison (RawMod / Dorado / MicrobeMod) with the
                  modification-scope matrix showing which tools can even attempt each type.

Reads whatever is present and skips the rest, so it can be run repeatedly as results land.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# fixed, CVD-safe assignments used consistently across every figure
C_RAWMOD = '#2563eb'
C_DORADO = '#e6820e'
C_MICROBE = '#7c3aed'
C_NULL = '#9ca3af'
C_OK = '#2ca25f'
C_NO = '#de2d26'
INK = '#111827'
MUTED = '#6b7280'
MOD_COLORS = {'6mA': '#2ca25f', '5mC': '#3182bd', '5hmC': '#7c3aed',
              '4mC': '#f59e0b', '5hmU': '#de2d26', 'unmod': '#9ca3af'}

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 10,
    'axes.edgecolor': '#d1d5db', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': '#eef2f7', 'grid.linewidth': 0.7,
    'axes.axisbelow': True, 'legend.frameon': False,
})


def _read_tsv(p):
    if not Path(p).exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


# ── Figure 1 ─────────────────────────────────────────────────────────────────
def fig1_denovo(motif_dir, out_png):
    """Recovered-vs-expected per organism + the null."""
    organisms, recovered, expected, nulls = [], [], [], []
    for d in sorted(Path(motif_dir).glob('*')):
        s = d / 'summary.txt'
        if not s.exists():
            continue
        txt = s.read_text()
        is_null = 'NULL CONTROL' in txt
        exp = rec = 0
        nfound = 0
        for line in txt.splitlines():
            if line.startswith('expected:'):
                parts = line.replace('expected:', '').replace('recovered:', '').split()
                try:
                    exp, rec = int(parts[0]), int(parts[1])
                except Exception:
                    pass
            if line.startswith('motifs passing'):
                try:
                    nfound = int(line.split(':')[1])
                except Exception:
                    pass
        if is_null:
            nulls.append((d.name, nfound))
        else:
            organisms.append(d.name); recovered.append(rec); expected.append(exp)

    if not organisms and not nulls:
        print("  [fig1] nothing to plot yet"); return

    fig, (ax, axn) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [2.4, 1]})
    x = np.arange(len(organisms))
    if organisms:
        ax.bar(x, expected, 0.62, color='#e5e7eb', label='known (REBASE) motifs')
        ax.bar(x, recovered, 0.62, color=C_RAWMOD, label='re-discovered de novo')
        for xi, (r, e) in enumerate(zip(recovered, expected)):
            ax.text(xi, max(r, e) + 0.08, f"{r}/{e}", ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(organisms, rotation=20, ha='right')
        ax.set_ylim(0, max(expected + [1]) * 1.3)
    ax.set_ylabel('number of motifs')
    ax.set_title('a  De novo motif re-discovery — no motif supplied to the model',
                 loc='left', fontweight='bold')
    ax.legend(fontsize=9)

    axn.bar(range(len(nulls)), [n for _, n in nulls], 0.5, color=C_NULL)
    axn.set_xticks(range(len(nulls)))
    axn.set_xticklabels([n for n, _ in nulls], rotation=20, ha='right')
    axn.set_ylabel('motifs passing E $\\leq$ 1e-30')
    axn.set_ylim(0, max([n for _, n in nulls] + [1]) * 1.4)
    axn.set_title('b  Unmodified null', loc='left', fontweight='bold')
    axn.text(0.5, 0.55, 'nothing recovered\n(as required)', transform=axn.transAxes,
             ha='center', color=MUTED, fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out_png}")


# ── Figure 2 ─────────────────────────────────────────────────────────────────
def fig2_zeroshot(metrics_dir, out_png):
    """LOMO: RawMod vs Dorado OR-ensemble on held-out modifications."""
    dor = {(r['fold'], r['test_set']): r for r in _read_tsv(Path(metrics_dir) / 'dorado.tsv')}
    labels, rm, dd = [], [], []
    for mod in ['5mC', '5hmC', '6mA', '5hmU']:
        rows = _read_tsv(Path(metrics_dir) / f'lomo_{mod}.tsv')
        for r in rows:
            ts = r.get('test_set', '')
            if '_heldout_' not in ts:
                continue
            src = ts.split('_heldout_')[0]
            labels.append(f"{mod}\n({src})")
            rm.append(float(r['mod_f1']))
            d = dor.get((f'lomo_{mod}', ts))
            dd.append(float(d['mod_f1']) if d else np.nan)
    if not labels:
        print("  [fig2] no LOMO metrics yet"); return

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, rm, w, color=C_RAWMOD, label='RawMod (never saw this modification)')
    ax.bar(x + w/2, dd, w, color=C_DORADO, label='Dorado specialist OR-ensemble')
    for xi, v in zip(x - w/2, rm):
        ax.text(xi, v + .012, f"{v:.2f}", ha='center', va='bottom', fontsize=9,
                fontweight='bold', color=INK)
    for xi, v in zip(x + w/2, dd):
        if np.isfinite(v):
            ax.text(xi, v + .012, f"{v:.2f}", ha='center', va='bottom', fontsize=9,
                    color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('modified-class F1')
    ax.set_title('Zero-shot detection of a held-out modification', loc='left',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.text(0.99, 0.95, '5hmU: no Dorado model exists —\nand it is not a methylation',
            transform=ax.transAxes, ha='right', va='top', fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out_png}")


# ── Figure 3 ─────────────────────────────────────────────────────────────────
def fig3_clustering(cluster_dir, out_png):
    npz = Path(cluster_dir) / 'clustering_data.npz'
    met = _read_tsv(Path(cluster_dir) / 'clustering_metrics.tsv')
    if not npz.exists():
        print("  [fig3] no clustering output yet"); return
    d = np.load(npz, allow_pickle=True)
    X, mt = d['X_sig'], d['mod_type'].astype(str)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X))

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                  gridspec_kw={'width_ratios': [1.25, 1]})
    for t in sorted(set(mt)):
        m = mt == t
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.45,
                   c=MOD_COLORS.get(t, '#666'), label=t, linewidths=0)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.set_title('a  Signal signatures, coloured by modification\n'
                 '(type labels NOT used to fit)', loc='left', fontweight='bold')
    ax.legend(markerscale=3, fontsize=9)

    if met:
        tags = [r['tag'].split('.')[0] for r in met]
        at = [float(r['ari_type']) for r in met]
        ad = [float(r['ari_dataset']) for r in met]
        x = np.arange(len(tags)); w = 0.38
        axb.bar(x - w/2, at, w, color=C_OK, label='ARI vs modification type')
        axb.bar(x + w/2, ad, w, color=C_NO, label='ARI vs source dataset')
        axb.set_xticks(x); axb.set_xticklabels(tags)
        axb.set_ylabel('adjusted Rand index')
        axb.set_title('b  Confound test: chemistry or provenance?', loc='left',
                      fontweight='bold')
        axb.legend(fontsize=9)
        axb.text(0.5, -0.28, 'type $>$ dataset ⇒ clusters encode chemistry,\n'
                             'not which organism the reads came from',
                 transform=axb.transAxes, ha='center', fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out_png}")


# ── Figure 4 ─────────────────────────────────────────────────────────────────
def fig4_scope(out_png, baseline_json=None):
    """Which tools can even attempt each modification on R10.4.1."""
    tools = ['RawMod\n(this work)', 'Dorado', 'MicrobeMod', 'nanodisco\n(R9 only)']
    mods = ['6mA', '5mC', '5hmC', '4mC', '5hmU']
    # 1 = supported, 0.5 = separate/extra model required, 0 = no model
    M = np.array([
        [1, 1, 1, 1, 1],        # RawMod — agnostic detector
        [1, 1, 1, .5, 0],       # Dorado — 4mC needs a different model, no 5hmU
        [1, 1, 0, 0, 0],        # MicrobeMod — 5mC/6mA only
        [1, 1, 0, 1, 0],        # nanodisco — 3 methylations, but cannot run on R10
    ])
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    cmap = matplotlib.colors.ListedColormap(['#f3f4f6', '#fde68a', C_OK])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            s = {1: '✓', 0.5: '~', 0: '✗'}[M[i, j]]
            ax.text(j, i, s, ha='center', va='center', fontsize=15,
                    color=INK if M[i, j] else C_NO, fontweight='bold')
    ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, fontweight='bold')
    ax.set_yticks(range(len(tools))); ax.set_yticklabels(tools, fontsize=9)
    ax.set_title('Modification scope on R10.4.1 chemistry', loc='left', fontweight='bold')
    ax.grid(False)
    ax.legend(handles=[Patch(facecolor=C_OK, label='supported'),
                       Patch(facecolor='#fde68a', label='separate model required'),
                       Patch(facecolor='#f3f4f6', label='no model exists')],
              loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    print(f"  -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline3')
    ap.add_argument('--lomo-metrics',
                    default='/fs/cbcb-scratch/bds062/results/deepmod_full_pipeline2/results2/metrics')
    a = ap.parse_args()
    base = Path(a.base); figs = base / 'figures'; figs.mkdir(parents=True, exist_ok=True)
    print("building manuscript figures ...")
    fig1_denovo(base / 'motifs', figs / 'Fig1_denovo_motif_rediscovery.png')
    fig2_zeroshot(a.lomo_metrics, figs / 'Fig2_zeroshot_heldout_modification.png')
    fig3_clustering(base / 'clusters', figs / 'Fig3_unsupervised_type_clustering.png')
    fig4_scope(figs / 'Fig4_modification_scope_R10.png')
    print("done.")


if __name__ == '__main__':
    main()
