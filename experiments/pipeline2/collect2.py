#!/usr/bin/env python3
"""
Collect deepmod_full_pipeline2's per-fold metric tables
(metrics/{mixed,lodo_*,lomo_*}.tsv) into one master table and draw comparison
figures. Mirrors deepmod_full_pipeline1/collect.py's role, restructured for
this pipeline's fold/run/test_set schema instead of model/run/eval_kind.

Figures:
  figures/lodo_generalization.png
      Leave-one-dataset-out: each lodo_<name> fold's accuracy on its own
      fully held-out dataset (never seen in training) — the headline
      "does this generalize to an unseen organism" result.

  figures/lomo_generalization.png
      Leave-one-modification-out (mixed pool only): mod_rec/mod_f1 on the
      held-out modification's positions.

  figures/external_generalization.png
      All 8 folds' performance on the 4 fixed external role=test organisms
      (HP26695_WT, HPJ99_WT, Anabaena_WT, Tdenticola_WT — never trained on
      by any fold) — shows whether dropping any one training dataset/mod
      changes generalization to organisms nobody trains on.

  figures/mixed_per_source_breakdown.png
      The mixed model's own held-out test split, broken down by which
      source dataset each position came from.

Usage:  python collect2.py [--out-dir results1]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOMO_MODS = ['5mC', '5hmC', '6mA', '5hmU']       # same set/order as pipeline1
FOLDS = (['mixed', 'lodo_HP26695_WGA_5kHz', 'lodo_Ecoli_DM_5kHz',
          'lodo_Ecoli_DM_MSssI_5kHz', 'lodo_Ecoli_WT_5kHz', 'lodo_arabidopsis',
          'lodo_UMCES', 'lodo_ONT'] + [f'lomo_{m}' for m in LOMO_MODS])
FOLD_LABEL = {
    'mixed': 'Mixed', 'lodo_HP26695_WGA_5kHz': 'LODO: HP26695_WGA',
    'lodo_Ecoli_DM_5kHz': 'LODO: Ecoli_DM', 'lodo_Ecoli_DM_MSssI_5kHz': 'LODO: Ecoli_DM_MSssI',
    'lodo_Ecoli_WT_5kHz': 'LODO: Ecoli_WT', 'lodo_arabidopsis': 'LODO: arabidopsis',
    'lodo_UMCES': 'LODO: UMCES', 'lodo_ONT': 'LODO: ONT',
    **{f'lomo_{m}': f'LOMO: {m}' for m in LOMO_MODS},
}
EXTERNAL_SETS = ['HP26695_WT_5kHz', 'HPJ99_WT_5kHz', 'Anabaena_WT_5kHz', 'Tdenticola_WT_5kHz']
FLOAT_COLS = {'micro_f1', 'mod_f1', 'unmod_f1', 'macro_f1', 'mod_prec', 'mod_rec',
              'auprc', 'auroc', 'threshold'}
COLORS = ['#4878CF', '#6ACC65', '#D65F5F', '#956CB4', '#8C613C',
          '#DC7EC0', '#797979', '#D5BB67', '#55A8A1', '#B4513B']


def read_tsv(path):
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            for k in list(row):
                if k in FLOAT_COLS and row[k] not in ('', 'nan'):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = float('nan')
                elif k in ('n_pos', 'n_test'):
                    row[k] = int(row[k]) if row[k] else 0
            rows.append(row)
    return rows


def bar_chart(labels, values_by_series, series_names, title, ylabel, out_path,
              value_fmt='{:.2f}'):
    n_groups = len(labels)
    n_series = len(series_names)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * n_groups * max(n_series, 1)), 6))
    width = 0.8 / max(n_series, 1)
    x = np.arange(n_groups)
    for i, (series, vals) in enumerate(zip(series_names, values_by_series)):
        offs = x + (i - (n_series - 1) / 2) * width
        bars = ax.bar(offs, vals, width=width, label=series,
                      color=COLORS[i % len(COLORS)])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, value_fmt.format(v),
                       ha='center', va='bottom', fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13)
    if n_series > 1:
        ax.legend(fontsize=8, ncol=min(n_series, 4))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=str(Path(__file__).resolve().parent / 'results1'))
    args = ap.parse_args()

    out = Path(args.out_dir)
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    missing = []
    for fold in FOLDS:
        p = out / 'metrics' / f'{fold}.tsv'
        if not p.exists():
            missing.append(fold)
            continue
        all_rows.extend(read_tsv(p))
    if missing:
        print(f"WARNING: missing metrics for folds: {missing} — "
              f"drawing figures from whatever completed.", flush=True)

    merged_path = out / 'metrics' / 'all_metrics.tsv'
    if all_rows:
        cols = list(all_rows[0].keys())
        with open(merged_path, 'w') as fh:
            fh.write('\t'.join(cols) + '\n')
            for r in all_rows:
                fh.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print(f"Wrote {merged_path}  ({len(all_rows)} rows)")

    def rows_where(**kw):
        return [r for r in all_rows if all(r.get(k) == v for k, v in kw.items())]

    # Dorado OR-ensemble baseline (dorado_baseline2.py; ONT + UMCES only — the
    # bacteria/arabidopsis BAMs carry no Dorado mod calls). Absent -> no bars.
    dor_path = out / 'metrics' / 'dorado.tsv'
    dor_rows = read_tsv(dor_path) if dor_path.exists() else []
    if not dor_rows:
        print("NOTE: no metrics/dorado.tsv — run dorado_baseline2.py to add "
              "Dorado comparison bars.", flush=True)

    def dorado_val(fold, test_set, col):
        dd = [r for r in dor_rows
              if r.get('fold') == fold and r.get('test_set') == test_set]
        return dd[0][col] if dd else float('nan')

    # ── LODO generalization: each lodo_<name> fold on its own held-out set ────
    lodo_folds = [f for f in FOLDS if f.startswith('lodo_')]
    labels, micro, modf1, modrec, dor_f1 = [], [], [], [], []
    for fold in lodo_folds:
        name = fold[len('lodo_'):]
        rr = rows_where(fold=fold, test_set=f'held_out_{name}')
        if not rr:
            continue
        r = rr[0]
        labels.append(name)
        micro.append(r['micro_f1']); modf1.append(r['mod_f1']); modrec.append(r['mod_rec'])
        dor_f1.append(dorado_val(fold, f'held_out_{name}', 'mod_f1'))
    if labels:
        series = [micro, modf1, modrec]
        names = ['micro_f1', 'mod_f1', 'mod_rec']
        if any(np.isfinite(v) for v in dor_f1):     # ONT/UMCES only
            series.append(dor_f1); names.append('Dorado mod_f1')
        bar_chart(labels, series, names,
                  'LODO: accuracy on the fully held-out organism', 'score',
                  fig_dir / 'lodo_generalization.png')

    # ── LOMO generalization: held-out modification, mixed pool only ──────────
    # pipeline1's test definition gives up to two test sets per modification
    # (its ONT file and the UMCES held-out region), so bars are grouped
    # mod x source. Dorado (OR-ensemble with the held-out code dropped; 5hmU has
    # no Dorado model at all) is drawn alongside as the agnostic baseline.
    labels, rm_f1, dor_f1, rm_rec, dor_rec = [], [], [], [], []
    for mod in LOMO_MODS:
        for src in ('ONT', 'UMCES'):
            ts = f'{src}_heldout_{mod}'
            rr = rows_where(fold=f'lomo_{mod}', test_set=ts)
            if not rr:
                continue                      # 5hmU has no ONT file
            labels.append(f'{mod}\n({src})')
            rm_f1.append(rr[0]['mod_f1']); rm_rec.append(rr[0]['mod_rec'])
            dor_f1.append(dorado_val(f'lomo_{mod}', ts, 'mod_f1'))
            dor_rec.append(dorado_val(f'lomo_{mod}', ts, 'mod_rec'))
    if labels:
        bar_chart(labels, [rm_f1, dor_f1, rm_rec, dor_rec],
                  ['RawMod mod_f1', 'Dorado mod_f1', 'RawMod mod_rec', 'Dorado mod_rec'],
                  'LOMO: held-out modification — RawMod vs Dorado OR-ensemble',
                  'score', fig_dir / 'lomo_generalization.png')

    # ── External held-out organisms, compared across every fold ──────────────
    series = []
    for fold in FOLDS:
        vals = []
        for ext in EXTERNAL_SETS:
            rr = rows_where(fold=fold, test_set=f'external_{ext}')
            vals.append(rr[0]['mod_f1'] if rr else float('nan'))
        if any(np.isfinite(v) for v in vals):
            series.append((fold, vals))
    if series:
        bar_chart(EXTERNAL_SETS, [v for _, v in series], [FOLD_LABEL[f] for f, _ in series],
                  'External generalization (mod_f1) — 4 organisms never trained on by any fold',
                  'mod_f1', fig_dir / 'external_generalization.png')

    # ── Mixed model's own held-out test, broken down by source dataset ───────
    mixed_source_rows = [r for r in all_rows
                         if r.get('fold') == 'mixed' and r.get('test_set', '').startswith('held_out_test_')]
    if mixed_source_rows:
        labels = [r['test_set'][len('held_out_test_'):] for r in mixed_source_rows]
        micro = [r['micro_f1'] for r in mixed_source_rows]
        modf1 = [r['mod_f1'] for r in mixed_source_rows]
        bar_chart(labels, [micro, modf1], ['micro_f1', 'mod_f1'],
                  'Mixed model: held-out test split, by source dataset', 'score',
                  fig_dir / 'mixed_per_source_breakdown.png')

    print("Done.")


if __name__ == '__main__':
    main()
