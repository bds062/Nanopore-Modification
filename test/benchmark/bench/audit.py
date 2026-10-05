#!/usr/bin/env python3
"""Robustness checks of the benchmark comparison, on the same sites as bench.paper_table.

  A  two-sample rows: are positives and negatives at the same positions? (each block is filtered by its own sample's
     coverage; the classes are only "matched" on the positions both samples cover). AUROC on all sites vs matched sites.
  B  leakage: distance from every scored site to the nearest RawMod training position of the same organism. Images span
     +-10 bases, so a held-out site next to a trained one shares most of its signal. RawMod AUROC by distance.
  C  missing calls: a selected read without a call counts as 0 (unmodified). Per-tool call rate on its own rows, and
     AUROC when scoring the mean over called reads only.
  D  pooled one-threshold AUROC: bootstrap CI, leave-one-row-out, rows with sequence-only null < 0.75, two-sample rows
     only, and fairer combinations for multi-model tools (models of the row's base only; per-model rank calibration).

  BENCH_HOLDOUT=r81 python -m bench.audit --out DIR
"""
import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import common as C
from . import score as Sc
from .paper_table import GROUPS, TOOLS, families, claims, row_marks

FLANK = 10   # image half-window (featurization --half-window 10)


def auc(y, s):
    return roc_auc_score(y, s) if len(y) and 0 < y.sum() < len(y) else np.nan


def site_table(smp, subtool, sites):
    s = pd.read_csv(C.sites_path(smp, subtool), sep='\t', dtype={'contig': str})
    m = sites[['contig', 'pos', 'strand']].merge(s, on=['contig', 'pos', 'strand'], how='left')
    return m['sum_p'].fillna(0).to_numpy(), m['n_called'].fillna(0).to_numpy()


def trained_positions():
    """organism regex -> {contig: sorted array of trained positions} (bench.holdout source)."""
    from . import holdout
    d = holdout._table()
    out = {}
    for smp, pat in holdout._map().items():
        if pat in out:
            continue
        x = d[d['file'].str.contains(pat, regex=True)]
        out[pat] = {c: np.unique(g['pos'].to_numpy()) for c, g in x.groupby('contig')}
    return out


def nearest(tp, contig, pos):
    a = tp.get(contig)
    if a is None or not len(a):
        return np.full(len(pos), 10**9)
    j = np.searchsorted(a, pos)
    lo = np.abs(pos - a[np.clip(j - 1, 0, len(a) - 1)])
    hi = np.abs(a[np.clip(j, 0, len(a) - 1)] - pos)
    return np.minimum(lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S, T, CL = C.samples(), C.subtools(), claims()
    fam = families(T)
    rows = {r['row']: r for r in Sc.rows_cfg()}
    nulls = pd.read_csv(C.WORK / f'nulls_{C.PAPER_TAG}.tsv', sep='\t').set_index('row')['ref11'].to_dict()
    from . import holdout
    hmap = holdout._map() if C.HOLDOUT else {}
    tp = trained_positions() if C.HOLDOUT else {}
    models = sorted({t for f, _ in TOOLS for t in fam[f]})
    cache, rng = {}, np.random.default_rng(0)
    A, B, Cc, pool = [], [], [], collections.defaultdict(list)
    for _, members in GROUPS:
        for row, _ in members:
            r = rows[row]
            need = [r['pos_sample']] + ([r['neg_sample']] if r['neg_sample'] else [])
            if not all(C.readsel_path(x).exists() for x in need):
                continue
            blocks = Sc.row_sites(r, S, cache)
            y = np.concatenate([b[2] for b in blocks])
            take = np.concatenate([rng.permutation(np.flatnonzero(y == k))[:1000] for k in (0, 1)])   # = paper_table
            sp, nc = {}, {}
            for t in models:
                if all(C.sites_path(smp, t).exists() for smp, _, _ in blocks):
                    v = [site_table(smp, t, sites) for smp, sites, _ in blocks]
                    sp[t] = np.concatenate([x[0] for x in v]); nc[t] = np.concatenate([x[1] for x in v])
            key = np.concatenate([[f'{c}:{p}:{s}' for c, p, s in zip(b[1]['contig'], b[1]['pos'], b[1]['strand'])] for b in blocks])
            # A: matching on two-sample rows
            if r['type'] == 'two':
                kp, kn = set(key[y == 1]), set(key[y == 0])
                both = np.array([k in kp and k in kn for k in key])
                rec = {'row': row, 'n_pos': int((y == 1).sum()), 'n_neg': int((y == 0).sum()), 'n_matched_pos': int(both[y == 1].sum()),
                       'frac_pos_matched': both[y == 1].mean(), 'frac_neg_matched': both[y == 0].mean()}
                for f in ('rawmod', 'dorado', 'unimeth'):
                    use = [t for t in fam[f] if t in sp]
                    if use:
                        s = np.max([sp[t] for t in use], axis=0)
                        rec[f'{f}_all'] = auc(y, s); rec[f'{f}_matched'] = auc(y[both], s[both])
                A.append(rec)
            # B: distance to nearest RawMod training position
            if tp:
                dist = np.concatenate([np.array([nearest(tp[hmap[smp]], c, np.array([p]))[0] if smp in hmap else 10**9
                                                 for c, p in zip(sites['contig'], sites['pos'])]) for smp, sites, _ in blocks])
                s = sp.get('rawmod')
                rec = {'row': row, 'n': len(y), 'median_dist': float(np.median(dist)), 'frac_within_10': float((dist <= FLANK).mean()),
                       'frac_within_100': float((dist <= 100).mean())}
                if s is not None:
                    rec['rawmod_all'] = auc(y, s)
                    for lim in (FLANK, 100, 1000):
                        far = dist > lim
                        rec[f'rawmod_dist_gt{lim}'] = auc(y[far], s[far]); rec[f'n_gt{lim}'] = int(far.sum())
                B.append(rec)
            # C: call rates and called-only scoring
            for t in sp:
                own = bool(row_marks(r) & CL.get(t, {'marks': set()})['marks'])
                called_only = np.where(nc[t] > 0, sp[t] / np.maximum(nc[t], 1), 0.0)
                Cc.append({'row': row, 'model': t, 'own': own, 'call_rate': float(nc[t].mean() / C.K),
                           'auroc_std': auc(y, sp[t] / C.K), 'auroc_called_only': auc(y, called_only)})
            # D: pooled samples per model
            pool['row'] += [row] * len(take); pool['label'] += list(y[take]); pool['base'] += [r['base']] * len(take)
            pool['null'] += [nulls.get(row, np.nan)] * len(take); pool['type'] += [r['type']] * len(take)
            for t in models:
                pool[t] += list(sp[t][take] / C.K) if t in sp else [np.nan] * len(take)
            print(f'[{row}] done', flush=True)
    pd.DataFrame(A).to_csv(out / 'A_matching.tsv', sep='\t', index=False, float_format='%.4f')
    pd.DataFrame(B).to_csv(out / 'B_leakage.tsv', sep='\t', index=False, float_format='%.4f')
    pd.DataFrame(Cc).to_csv(out / 'C_calls.tsv', sep='\t', index=False, float_format='%.4f')
    P = pd.DataFrame(pool)
    P.to_csv(out / 'D_pool_models.tsv.gz', sep='\t', index=False, float_format='%.5f')

    # D: pooled-AUROC variants
    def combine(f, df, how):
        use = [t for t in fam[f] if t in df and df[t].notna().all()]
        if not use:
            return None
        X = df[use].to_numpy()
        if how == 'base':                     # only the models of the row's base (what a user calling that base would run)
            ok = np.array([[df['base'].iloc[i] in T[t]['bases'] or df['base'].iloc[i] == 'N' for t in use] for i in range(len(df))])
            X = np.where(ok, X, 0.0)
        if how == 'rank':                     # each model's scores -> percentiles over all pooled sites (label-free calibration)
            X = np.column_stack([pd.Series(X[:, j]).rank(pct=True).to_numpy() for j in range(X.shape[1])])
        return X.max(axis=1)
    D = []
    y = P['label'].to_numpy()
    rows_all = list(dict.fromkeys(P['row']))
    for f, _ in TOOLS:
        rec = {'tool': f}
        for how in ('max', 'base', 'rank'):
            s = combine(f, P, how)
            rec[f'pooled_{how}'] = auc(y, s) if s is not None else np.nan
        s = combine(f, P, 'max')
        if s is None:
            D.append(rec); continue
        m1 = (P['null'] < 0.75).to_numpy(); m2 = (P['type'] == 'two').to_numpy()
        rec['pooled_null_lt075'] = auc(y[m1], s[m1]); rec['pooled_two_sample'] = auc(y[m2], s[m2])
        loo = [auc(y[P['row'] != rr], s[P['row'] != rr]) for rr in rows_all]
        rec['loo_min'] = np.min(loo); rec['loo_min_row'] = rows_all[int(np.argmin(loo))]
        bs, g = [], np.random.default_rng(1)
        idx_by_row = {rr: np.flatnonzero(P['row'].to_numpy() == rr) for rr in rows_all}
        for _ in range(300):                  # stratified by row: resample sites within each row
            ii = np.concatenate([g.choice(v, len(v)) for v in idx_by_row.values()])
            bs.append(auc(y[ii], s[ii]))
        rec['ci_lo'], rec['ci_hi'] = np.percentile(bs, [2.5, 97.5])
        D.append(rec)
    D = pd.DataFrame(D)
    D.to_csv(out / 'D_pooled.tsv', sep='\t', index=False, float_format='%.4f')
    for name, df in (('A matching', pd.DataFrame(A)), ('B leakage', pd.DataFrame(B)), ('D pooled', D)):
        print(f'\n== {name}\n{df.round(3).to_string(index=False)}')


if __name__ == '__main__':
    main()
