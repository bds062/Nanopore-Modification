#!/usr/bin/env python3
"""
Dorado modification-agnostic baseline for deepmod_full_pipeline1.

Runs the "most recent" Dorado specialist models as an OR-ensemble:
  5mC_5hmC@v2 (codes m,h) + 6mA@v1 (code a), sup@v5.2.0.

Per test SITE (aggregated position) the site-level modkit fraction is taken per
code; if ANY *active* code's fraction > 50% the site is predicted "modified",
else "unmodified". Because we are modification-agnostic we OR the models
together. For LOMO we drop the held-out modification's code (5hmU has no Dorado
model, so it keeps all three — the "Dorado can't detect 5hmU" bar).

The test sites are reconstructed from run_pipeline (identical, seeded splits) so
the Dorado bars are directly comparable to the neural models. Sources:
  ONT   : modkit pileups of the dorado_comparison BAMs (dorado_pileups/, v5.2.0)
  UMCES : existing barcode0{6,7}_pileup.bed (v5.0.0); bc01-05 have no Dorado
          calls and score unmodified (matches the dorado_comparison convention).

Writes metrics/dorado.tsv (same schema as the model tsvs).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_pipeline as R

DORADO_DIR   = HERE / 'dorado_pileups'
UMCES_PILEUP = {
    'bc06': '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/modbam/barcode06_pileup.bed',
    'bc07': '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/modbam/barcode07_pileup.bed',
}
CODE_OF   = {'5mC': 'm', '5hmC': 'h', '6mA': 'a'}     # modification -> modkit code
ALL_CODES = {'m', 'h', 'a'}
FRAC_THRESHOLD = 50.0                                  # percent (>0.5 fraction)


# ── load Dorado per-site fractions ────────────────────────────────────────────
def _parse_pileup(path, store):
    """Accumulate max percent-modified per (contig,pos,code) into store."""
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) < 11:
                continue
            code = c[3]
            if code not in ALL_CODES:
                continue
            try:
                frac = float(c[10])
            except ValueError:
                continue
            key = (c[0], int(c[1]))
            d = store.setdefault(key, {})
            if frac > d.get(code, -1.0):
                d[code] = frac


def load_ont():
    """dataset -> {(contig,pos): {code: frac}} from dorado_pileups/."""
    data = {}
    for ds in ['5mC', '5hmC', '6mA', 'control']:
        store = {}
        for mk in ['5mC_5hmC_v2', '6mA_v1']:
            p = DORADO_DIR / f'{ds}__{mk}.bed'
            if p.exists():
                _parse_pileup(p, store)
            else:
                print(f"  WARNING: missing {p}", file=sys.stderr)
        data[ds] = store
    return data


def load_umces():
    data = {}
    for bc, p in UMCES_PILEUP.items():
        store = {}
        if Path(p).exists():
            _parse_pileup(p, store)
        data[bc] = store
    return data


# ── prediction + metrics ──────────────────────────────────────────────────────
def predict(keys, dataset_of_fileidx, dorado_data, active_codes):
    """keys: list of (file_idx, contig, pos). -> np.array of 0/1 predictions."""
    preds = np.zeros(len(keys), dtype=int)
    for i, (fi, contig, pos) in enumerate(keys):
        ds = dataset_of_fileidx.get(int(fi))
        if ds is None:                      # e.g. bc01-05 → no Dorado call → unmod
            continue
        fr = dorado_data[ds].get((contig, int(pos)))
        if not fr:
            continue
        if any(fr.get(code, 0.0) > FRAC_THRESHOLD for code in active_codes):
            preds[i] = 1
    return preds


def unique_positions(group, idx):
    """Aggregate images to unique (file,contig,pos) with max label (mirrors eval)."""
    keys = group.source_keys(idx)
    lab  = group.labels[idx]
    agg = {}
    for k, l in zip(keys, lab):
        agg[k] = max(agg.get(k, 0), int(l > 0))
    ukeys = list(agg.keys())
    return ukeys, np.array([agg[k] for k in ukeys], dtype=int)


def metrics_row(model, run, eval_kind, test_set, held_out, y_true, y_pred):
    yt = y_true.astype(int)
    row = {
        'model': model, 'run': run, 'eval_kind': eval_kind,
        'test_set': test_set, 'held_out': held_out,
        'micro_f1':  float(f1_score(yt, y_pred, average='micro', zero_division=0)),
        'mod_f1':    float(f1_score(yt, y_pred, pos_label=1, zero_division=0)),
        'unmod_f1':  float(f1_score(yt, y_pred, pos_label=0, zero_division=0)),
        'macro_f1':  float(f1_score(yt, y_pred, average='macro', zero_division=0)),
        'mod_prec':  float(precision_score(yt, y_pred, pos_label=1, zero_division=0)),
        'mod_rec':   float(recall_score(yt, y_pred, pos_label=1, zero_division=0)),
        'auprc':     float('nan'), 'auroc': float('nan'),
        'threshold': 0.5, 'n_pos': int(yt.sum()), 'n_test': int(len(yt)),
    }
    return row


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default=str(HERE / 'results1'))
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)

    ont = R.Group(R.ONT_ORDER, R.ONT_FILES)
    umc = R.Group(R.UMCES_ORDER, R.UMCES_FILES)
    mod_map = R.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    tc = R.pick_ont_test_contigs(ont, R.N_ONT_TEST_CONTIGS, R.HP.seed)
    _, ont_test = R.ont_region_split(ont, tc)
    _, umc_test = R.umces_region_split(umc)

    dor_ont = load_ont()
    dor_umc = load_umces()

    ONT_DS   = {i: R.ONT_ORDER[i] for i in range(len(R.ONT_ORDER))}       # base ONT group
    UMC_DS   = {i: (R.UMCES_ORDER[i] if R.UMCES_ORDER[i] in UMCES_PILEUP else None)
                for i in range(len(R.UMCES_ORDER))}

    rows = []

    # ── base (all codes active) ───────────────────────────────────────────────
    k, y = unique_positions(ont, ont_test)
    rows.append(metrics_row('dorado', 'base', 'base', 'ONT_test', '',
                            y, predict(k, ONT_DS, dor_ont, ALL_CODES)))
    k, y = unique_positions(umc, umc_test)
    rows.append(metrics_row('dorado', 'base', 'base', 'UMCES_test', '',
                            y, predict(k, UMC_DS, dor_umc, ALL_CODES)))

    # ── LOMO (drop held-out code; 5hmU keeps all) ─────────────────────────────
    for mod in R.UMCES_LOMO_MODS:                 # 5mC,5hmC,6mA,5hmU
        active = ALL_CODES - ({CODE_OF[mod]} if mod in CODE_OF else set())
        # Dorado -> ONT (only for mods present in ONT)
        if mod in R.ONT_LOMO_MODS:
            g = R.Group([mod], R.ONT_FILES)
            k, y = unique_positions(g, np.arange(g.N, dtype=np.int64))
            rows.append(metrics_row('dorado', f'lomo_{mod}', 'lomo',
                                    f'ONT_heldout_{mod}', mod, y,
                                    predict(k, {0: mod}, dor_ont, active)))
        # Dorado -> UMCES
        _, te = R.umces_lomo_split(umc, mod_map, mod)
        k, y = unique_positions(umc, te)
        rows.append(metrics_row('dorado', f'lomo_{mod}', 'lomo',
                                f'UMCES_heldout_{mod}', mod, y,
                                predict(k, UMC_DS, dor_umc, active)))

    cols = ['model', 'run', 'eval_kind', 'test_set', 'held_out', 'micro_f1', 'mod_f1',
            'unmod_f1', 'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc',
            'threshold', 'n_pos', 'n_test']
    tsv = out / 'metrics' / 'dorado.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(
                f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, ''))
                for c in cols) + '\n')
    print(f"Wrote {tsv}  ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['test_set']:24s} held={r['held_out']:5s} "
              f"micro_f1={r['micro_f1']:.3f} mod_f1={r['mod_f1']:.3f} "
              f"mod_rec={r['mod_rec']:.3f} n_pos={r['n_pos']}")


if __name__ == '__main__':
    main()
