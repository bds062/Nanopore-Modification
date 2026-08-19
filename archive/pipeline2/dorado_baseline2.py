#!/usr/bin/env python3
"""
Dorado modification-agnostic baseline for deepmod_full_pipeline2.

Reuses deepmod_full_pipeline1's dorado_baseline machinery (pileup parsing, the
OR-ensemble decision rule, site aggregation) so the bars mean exactly the same
thing in both pipelines:

  Models   : 5mC_5hmC@v2 (codes m,h) + 6mA@v1 (code a), sup@v5.2.0.
  Decision : per test SITE, take the site-level modkit fraction per code; if ANY
             *active* code's fraction > 50% the site is predicted "modified".
             We OR the specialist models together because we are simulating a
             modification-agnostic caller.
  LOMO     : drop the held-out modification's code from the active set. 5hmU has
             no Dorado model at all, so it keeps all three codes — that is the
             "Dorado cannot detect 5hmU" bar, not a bug.

SCOPE — ONT + UMCES only. The bacteria/arabidopsis BAMs were basecalled WITHOUT
modification models (verified: no MM/ML tags), so Dorado has no calls for them;
covering those would require re-basecalling ~87GB of POD5 with
`dorado basecall --modified-bases` + modkit pileup. Folds/test sets with no
Dorado calls simply get no Dorado bar.

Test sets are reconstructed to match run_pipeline2.py exactly:
  lomo_<mod>  -> ONT_heldout_<mod>   (that modification's whole ONT file)
                 UMCES_heldout_<mod> (R.umces_lomo_split test half)
  lodo_ONT    -> held_out_ONT        (all ONT images)
  lodo_UMCES  -> held_out_UMCES      (all UMCES images)

Writes metrics/dorado.tsv in run_pipeline2.py's schema, so collect2.py can draw
Dorado next to each fold's model bar.

Usage:  python dorado_baseline2.py [--out-dir results1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / 'deepmod_full_pipeline1'
sys.path.insert(0, str(P1))
sys.path.insert(0, '/fs/nexus-scratch/bds062/Nanopore-Modification/rawmod')

import run_pipeline as R                     # noqa: E402
import dorado_baseline as D                  # noqa: E402 — pileups, predict, OR-rule
from mod_types import build_umces_mod_map    # noqa: E402

# run_pipeline2.py's tsv schema
COLS = ['fold', 'run', 'test_set', 'held_out', 'micro_f1', 'mod_f1', 'unmod_f1',
        'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc', 'threshold',
        'n_pos', 'n_test']


def row(fold, test_set, held_out, y_true, y_pred):
    """pipeline1's metrics_row, remapped onto pipeline2's (fold, run) schema."""
    m = D.metrics_row('dorado', 'dorado', 'dorado', test_set, held_out,
                      y_true, y_pred)
    m.pop('model', None)
    m.pop('eval_kind', None)
    m['fold'] = fold
    m['run'] = 'dorado'
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=str(HERE / 'results1'))
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)

    ont = R.Group(list(R.ONT_ORDER), R.ONT_FILES)
    umc = R.Group(list(R.UMCES_ORDER), R.UMCES_FILES)
    mod_map = build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)

    dor_ont = D.load_ont()
    dor_umc = D.load_umces()
    ONT_DS = {i: R.ONT_ORDER[i] for i in range(len(R.ONT_ORDER))}
    UMC_DS = {i: (R.UMCES_ORDER[i] if R.UMCES_ORDER[i] in D.UMCES_PILEUP else None)
              for i in range(len(R.UMCES_ORDER))}

    rows = []

    # ── LODO: the whole held-out dataset, every code active ──────────────────
    k, y = D.unique_positions(ont, np.arange(ont.N, dtype=np.int64))
    rows.append(row('lodo_ONT', 'held_out_ONT', 'ONT', y,
                    D.predict(k, ONT_DS, dor_ont, D.ALL_CODES)))
    k, y = D.unique_positions(umc, np.arange(umc.N, dtype=np.int64))
    rows.append(row('lodo_UMCES', 'held_out_UMCES', 'UMCES', y,
                    D.predict(k, UMC_DS, dor_umc, D.ALL_CODES)))

    # ── LOMO: drop the held-out modification's code (5hmU keeps all three) ───
    for mod in R.UMCES_LOMO_MODS:                      # 5mC, 5hmC, 6mA, 5hmU
        active = D.ALL_CODES - ({D.CODE_OF[mod]} if mod in D.CODE_OF else set())
        if mod in R.ONT_LOMO_MODS:
            g = R.Group([mod], R.ONT_FILES)
            k, y = D.unique_positions(g, np.arange(g.N, dtype=np.int64))
            rows.append(row(f'lomo_{mod}', f'ONT_heldout_{mod}', mod, y,
                            D.predict(k, {0: mod}, dor_ont, active)))
        _, te = R.umces_lomo_split(umc, mod_map, mod)
        k, y = D.unique_positions(umc, te)
        rows.append(row(f'lomo_{mod}', f'UMCES_heldout_{mod}', mod, y,
                        D.predict(k, UMC_DS, dor_umc, active)))

    tsv = out / 'metrics' / 'dorado.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(
                f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, ''))
                for c in COLS) + '\n')
    print(f"Wrote {tsv}  ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['fold']:12s} {r['test_set']:22s} held={r['held_out']:5s} "
              f"micro_f1={r['micro_f1']:.3f} mod_f1={r['mod_f1']:.3f} "
              f"mod_rec={r['mod_rec']:.3f} n_pos={r['n_pos']}")


if __name__ == '__main__':
    main()
