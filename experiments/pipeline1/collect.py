#!/usr/bin/env python3
"""
Collect the per-model metric tables (metrics/{ont_only,umces_only,both,dorado}.tsv)
into one master table and draw the four poster-ready comparison figures:

  figures/base_micro_f1.png   figures/base_mod_f1.png
      x = {ONT test, UMCES test}, grouped by model (cross-dataset, Mode A)

  figures/lomo_micro_f1.png   figures/lomo_mod_f1.png
      x = held-out modification, grouped by model (Mode B).
      For 'both' and 'dorado', the ONT-held-out and UMCES-held-out evaluations
      are averaged into a single bar per held-out modification (falls back to
      the lone available value for 5hmU, which has no ONT counterpart).

Styling is poster-scale by default (large fonts, large vertical bar-top value
labels, minimal chrome) since this script is the shared figure generator for
every resultsN/ directory.

Usage:  python collect.py [--out-dir results1] [--only both,dorado]

  --only  Comma-separated subset of {ont_only,umces_only,both,dorado} to draw
          as an additional, separate set of figures (filenames get a suffix,
          e.g. base_micro_f1_mixed_dorado.png) alongside the standard 4-series
          figures, which are always produced.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODELS = ['ont_only', 'umces_only', 'both', 'dorado']   # dorado = OR-ensemble baseline
MODEL_LABEL = {'ont_only': 'ONT-only', 'umces_only': 'UMCES-only', 'both': 'RawMod',
               'dorado': 'Dorado'}
FLOAT_COLS = {'micro_f1', 'mod_f1', 'unmod_f1', 'macro_f1', 'mod_prec', 'mod_rec',
              'auprc', 'auroc', 'threshold'}
COLORS = {'ONT-only': '#4878CF', 'UMCES-only': '#6ACC65', 'RawMod': '#D65F5F',
        #   'Dorado': '#EAD9A6'}
        'Dorado': '#878682'}
INSIDE_THRESHOLD = 0.9   # bars taller than this get their label inside, near the top
LOMO_ORDER = ['5mC', '5hmC', '6mA', '5hmU']
METRIC_YLABEL = {'mod_f1': 'Modified Site F1'}   # overrides for specific metric y-axis titles

# ── poster-scale typography ───────────────────────────────────────────────────
FS_TITLE  = 24
FS_AXIS   = 20
FS_TICK   = 18
FS_VALUE  = 17
FS_LEGEND = 16
plt.rcParams['font.size'] = FS_TICK


def read_rows(out_dir: Path):
    rows = []
    for m in MODELS:
        p = out_dir / 'metrics' / f'{m}.tsv'
        if not p.exists():
            print(f"  (missing {p.name} — skipping)")
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                for c in FLOAT_COLS:
                    if r.get(c) not in (None, ''):
                        try:
                            r[c] = float(r[c])
                        except ValueError:
                            r[c] = float('nan')
                rows.append(r)
    return rows


def _contrast_text_color(hex_color):
    """Best-contrast label color (black/white) for a given hex bar fill color."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'black'
    # return 'black' if luminance > 150 else 'white'


def grouped_bar(ax, x_labels, series, title, ylabel):
    """series: list of (label, {x_label: value}); missing values are skipped.
    Value labels are large, bold, rotated vertical. Bars taller than
    INSIDE_THRESHOLD get their label inside the bar near the top (contrast-aware
    color); shorter bars get the label just above the bar in black."""
    n = len(series)
    x = np.arange(len(x_labels))
    w = 0.8 / max(n, 1)
    for k, (lab, vals) in enumerate(series):
        ys = [vals.get(xl, np.nan) for xl in x_labels]
        xs = x + (k - (n - 1) / 2) * w
        color = COLORS.get(lab, f'C{k}')
        ax.bar(xs, [0 if np.isnan(v) else v for v in ys], w,
              label=lab, color=color, alpha=0.9)
        for xi, v in zip(xs, ys):
            if np.isnan(v):
                continue
            if v > INSIDE_THRESHOLD:
                ax.text(xi, v - 0.02, f'{v:.2f}', ha='center', va='top',
                        fontsize=FS_VALUE, fontweight='bold', rotation=90,
                        color=_contrast_text_color(color))
            else:
                ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', va='bottom',
                        fontsize=FS_VALUE, fontweight='bold', rotation=90,
                        color='black')
    ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    ax.set_title(title, fontsize=FS_TITLE, pad=14)
    ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=FS_LEGEND, ncol=n, loc='upper center',
              bbox_to_anchor=(0.5, -0.04), frameon=False)


def make_base_fig(rows, metric, title, path, models=MODELS):
    base = [r for r in rows if r['eval_kind'] == 'base']
    xlabels = ['ONT test', 'UMCES test']
    tmap = {'ONT_test': 'ONT test', 'UMCES_test': 'UMCES test'}
    series = []
    for m in models:
        vals = {tmap[r['test_set']]: r[metric] for r in base
                if r['model'] == m and r['test_set'] in tmap}
        if vals:
            series.append((MODEL_LABEL[m], vals))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ylabel = METRIC_YLABEL.get(metric, metric.replace('_', '-').upper())
    grouped_bar(ax, xlabels, series, title, ylabel)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  -> {path}")


def make_lomo_fig(rows, metric, title, path, models=MODELS):
    """One bar per model per held-out modification. For 'both'/'dorado', the
    ONT-held-out and UMCES-held-out evaluations are averaged together."""
    lomo = [r for r in rows if r['eval_kind'] == 'lomo']
    xlabels = [m for m in LOMO_ORDER if any(r['held_out'] == m for r in lomo)]
    series = []
    for model in models:
        vals = {}
        for mod in xlabels:
            matched = [r[metric] for r in lomo
                      if r['model'] == model and r['held_out'] == mod]
            if matched:
                vals[mod] = float(np.mean(matched))
        if vals:
            series.append((MODEL_LABEL[model], vals))
    fig, ax = plt.subplots(figsize=(11, 7))
    ylabel = METRIC_YLABEL.get(metric, metric.replace('_', '-').upper())
    grouped_bar(ax, xlabels, series, title, ylabel)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(Path(__file__).resolve().parent / 'results1'))
    ap.add_argument('--only', default=None,
                    help='Comma-separated subset of {ont_only,umces_only,both,dorado} to draw '
                         'as an additional set of figures (filename suffix reflects the subset), '
                         'alongside the standard 4-series figures.')
    args = ap.parse_args()
    out = Path(args.out_dir)
    fig_dir = out / 'figures'; fig_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(out)
    if not rows:
        raise SystemExit("No metrics found — did the model jobs finish?")

    # master table
    cols = ['model', 'run', 'eval_kind', 'test_set', 'held_out', 'micro_f1', 'mod_f1',
            'unmod_f1', 'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc',
            'threshold', 'n_pos', 'n_test']
    with open(out / 'metrics' / 'all_metrics.tsv', 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(
                f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, ''))
                for c in cols) + '\n')
    print(f"Wrote {out/'metrics'/'all_metrics.tsv'}  ({len(rows)} rows)")

    make_base_fig(rows, 'micro_f1', 'Micro-F1 by Test Set',
                  fig_dir / 'base_micro_f1.png')
    make_base_fig(rows, 'mod_f1', 'Modified-Class F1 by Test Set',
                  fig_dir / 'base_mod_f1.png')
    make_lomo_fig(rows, 'micro_f1', 'Micro-F1 — Held-Out Modification',
                  fig_dir / 'lomo_micro_f1.png')
    make_lomo_fig(rows, 'mod_f1', 'Modified-Class F1 — Held-Out Modification',
                  fig_dir / 'lomo_mod_f1.png')

    if args.only:
        requested = [m.strip() for m in args.only.split(',') if m.strip()]
        invalid = [m for m in requested if m not in MODELS]
        if invalid:
            raise SystemExit(f"--only: unknown model(s) {invalid}; choose from {MODELS}")
        subset = [m for m in MODELS if m in requested]   # preserve canonical order
        suffix = '_' + '_'.join(MODEL_LABEL[m].lower().replace(' ', '') for m in subset)
        print(f"\nAlso drawing --only subset {subset} (suffix '{suffix}'):")
        make_base_fig(rows, 'micro_f1', 'Micro-F1 by Test Set',
                      fig_dir / f'base_micro_f1{suffix}.png', subset)
        make_base_fig(rows, 'mod_f1', 'Modified-Class F1 by Test Set',
                      fig_dir / f'base_mod_f1{suffix}.png', subset)
        make_lomo_fig(rows, 'micro_f1', 'Micro-F1 — Held-Out Modification',
                      fig_dir / f'lomo_micro_f1{suffix}.png', subset)
        make_lomo_fig(rows, 'mod_f1', 'Modified-Class F1 — Held-Out Modification',
                      fig_dir / f'lomo_mod_f1{suffix}.png', subset)

    print("Done.")


if __name__ == '__main__':
    main()