#!/usr/bin/env python3
"""Score every (row, subtool) whose site file exists -> $BENCH_WORK/matrix_long.tsv and matrix_grid.tsv.

For each row: read gt/cand positions, give each its strand from the reference base (see rows.tsv), keep
the positions that are scorable (>= K selected reads on that strand, bench.readsel) in the sample they
are scored on, then score each subtool on exactly those sites:

  mean_P    = sum_p / K   (primary score; AUROC, AUPRC)
  call_freq = n_mod / K   (F1 with a site called modified when call_freq >= 0.5)

Selected reads a subtool did not call count as unmodified. A subtool that called none of a row's scored
sites is N/A (it has no model for that base/context); partial coverage keeps its number and the
fraction of sites it called is reported.

  python -m bench.score [--rows r1,r2] [--subtools t1,t2]
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd
import pysam
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from . import common as C


def rows_cfg():
    out = []
    for line in open(C.CONFIG / 'rows.tsv'):
        if line.strip() and not line.startswith('#'):
            c = line.rstrip('\n').split('\t')
            out.append(dict(row=c[0], chem=c[1], base=c[2], type=c[3], context=c[4], pos_sample=c[5],
                            neg_sample=None if c[6] == '-' else c[6], note=c[7] if len(c) > 7 else ''))
    return out


def read_bed(path):
    d = pd.read_csv(path, sep='\t', header=None, usecols=[0, 1], names=['contig', 'pos'],
                    dtype={'contig': str, 'pos': np.int64}, comment='#')
    return d.drop_duplicates()


class Scorable:
    """Site intervals (contig, strand) that reached K reads in one sample."""
    def __init__(self, sample):
        z = np.load(C.readsel_path(sample))
        ctg = z['contigs'].tolist()
        self.iv = collections.defaultdict(list)
        for c, s, a, b in zip(z['site_contig'], z['site_strand'], z['site_start'], z['site_end']):
            self.iv[(ctg[c], '+-'[s])].append((a, b))
        self.iv = {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v])) for k, v in self.iv.items()}
        self.contigs = {k[0] for k in self.iv}

    def mask(self, contig, strand, pos):
        a = self.iv.get((contig, strand))
        if a is None:
            return np.zeros(len(pos), bool)
        st, en = a
        j = np.searchsorted(st, pos, side='right') - 1
        ok = j >= 0
        return ok & (pos < en[np.where(ok, j, 0)])


def assign_strand(df, ref, base, scorable):
    """Keep positions scorable on the strand their reference base implies; adds 'strand'."""
    fa = pysam.FastaFile(ref)
    keep = []
    for contig, g in df[df['contig'].isin(scorable.contigs)].groupby('contig'):
        p = g['pos'].to_numpy()
        # only fetch bases at positions scorable on either strand
        m = scorable.mask(contig, '+', p) | scorable.mask(contig, '-', p)
        if not m.any():
            continue
        p = p[m]
        lo, hi = int(p.min()), int(p.max()) + 1
        seq = np.frombuffer(fa.fetch(contig, lo, hi).upper().encode(), dtype='S1')
        b = seq[p - lo].astype(str)
        if base == 'N':
            st = np.where(np.isin(b, ['A', 'C']), '+', np.where(np.isin(b, ['G', 'T']), '-', ''))
        else:
            st = np.where(b == base, '+', np.where(b == C.COMP[base], '-', ''))
        out = pd.DataFrame({'contig': contig, 'pos': p, 'strand': st})
        out = out[out['strand'] != '']
        ok = np.zeros(len(out), bool)
        for s in '+-':
            w = (out['strand'] == s).to_numpy()
            ok[w] = scorable.mask(contig, s, out['pos'].to_numpy()[w])
        keep.append(out[ok])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=['contig', 'pos', 'strand'])


def site_scores(sample, subtool, sites):
    """mean_P and call_freq for `sites` (contig, pos, strand) from the subtool's site file; absent = 0."""
    p = C.sites_path(sample, subtool)
    s = pd.read_csv(p, sep='\t', dtype={'contig': str})
    m = sites.merge(s, on=['contig', 'pos', 'strand'], how='left')
    if len(m) != len(sites):
        raise ValueError(f'{p}: duplicated (contig, pos, strand) keys -- re-run bench.aggregate')
    called = m['n_called'].notna().to_numpy()
    mean_p = (m['sum_p'].fillna(0) / C.K).to_numpy()
    freq = (m['n_mod'].fillna(0) / C.K).to_numpy()
    return mean_p, freq, called


def row_sites(r, S, cache):
    """-> list of (sample, sites df, label) blocks for this row."""
    R = C.WORK / 'rows' / r['row']
    gt, cand = read_bed(R / 'gt.bed'), read_bed(R / 'cand.bed')
    sc = lambda s: cache.setdefault(s, Scorable(s))
    if r['type'] == 'one':
        smp = r['pos_sample']
        sites = assign_strand(cand, S[smp]['ref'], r['base'], sc(smp))
        g = set(zip(gt['contig'], gt['pos']))
        lab = np.array([(c, p) in g for c, p in zip(sites['contig'], sites['pos'])], dtype=int)
        neg = np.flatnonzero(lab == 0)
        if len(neg) > C.NEG_CAP:      # same seeded subsample for every tool; positives always kept
            h = np.array([C.read_rank(f'{c}:{p}:{s}') for c, p, s in
                          zip(sites['contig'].to_numpy()[neg], sites['pos'].to_numpy()[neg], sites['strand'].to_numpy()[neg])],
                         dtype=np.uint64)
            keep = np.sort(np.r_[np.flatnonzero(lab == 1), neg[np.argsort(h)[:C.NEG_CAP]]])
            sites, lab = sites.iloc[keep].reset_index(drop=True), lab[keep]
        return _holdout(r, [(smp, sites, lab)])
    pos = assign_strand(gt, S[r['pos_sample']]['ref'], r['base'], sc(r['pos_sample']))
    neg = assign_strand(gt, S[r['neg_sample']]['ref'], r['base'], sc(r['neg_sample']))
    return _holdout(r, [(r['pos_sample'], pos, np.ones(len(pos), int)), (r['neg_sample'], neg, np.zeros(len(neg), int))])


def _holdout(r, blocks):
    """BENCH_HOLDOUT set: drop every position (either strand) that the RawMod checkpoint trained on in the organism of
    any of the row's samples, from every block of the row (bench.holdout)."""
    if not C.HOLDOUT:
        return blocks
    from . import holdout
    seen = holdout.seen_positions([smp for smp, _, _ in blocks])
    out = []
    for smp, sites, lab in blocks:
        keep = ~np.fromiter(((c, int(p)) in seen for c, p in zip(sites['contig'], sites['pos'])), bool, len(sites))
        out.append((smp, sites[keep].reset_index(drop=True), lab[keep]))
    return out


def applicable(r, t):
    """Does subtool t model row r's modified base and context? (subtools.tsv bases/context)"""
    base_ok = r['base'] == 'N' or r['base'] in t['bases']
    ctx_ok = t['context'] == 'any' or r['context'] != 'noncpg'
    return base_ok and ctx_ok


def score_row(r, blocks, subtool):
    if not applicable(r, C.subtools()[subtool]):
        return {'status': 'N/A'}
    mp, fr, cl, y = [], [], [], []
    for smp, sites, lab in blocks:
        if not C.sites_path(smp, subtool).exists():
            return {'status': 'pending'}
        a, b, c = site_scores(smp, subtool, sites)
        mp.append(a); fr.append(b); cl.append(c); y.append(lab)
    mp, fr, cl, y = map(np.concatenate, (mp, fr, cl, y))
    res = {'n_sites': len(y), 'n_pos': int(y.sum()), 'frac_called': cl.mean() if len(cl) else np.nan}
    if len(y) == 0 or y.min() == y.max():
        return {**res, 'status': 'single-class'}
    if not cl.any():
        return {**res, 'status': 'N/A'}
    return {**res, 'status': 'ok', 'auroc': roc_auc_score(y, mp), 'auprc': average_precision_score(y, mp),
            'f1': f1_score(y, fr >= 0.5, zero_division=0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', default='')
    ap.add_argument('--subtools', default='')
    a = ap.parse_args()
    S, T = C.samples(), C.subtools()
    only_r = set(filter(None, a.rows.split(',')))
    only_t = [t for t in (a.subtools.split(',') if a.subtools else T) if t]
    cache, long = {}, []
    for r in rows_cfg():
        if only_r and r['row'] not in only_r:
            continue
        need = [r['pos_sample']] + ([r['neg_sample']] if r['neg_sample'] else [])
        if not all(C.readsel_path(s).exists() for s in need):
            print(f"[{r['row']}] read selection missing for {need}; skipped")
            continue
        blocks = row_sites(r, S, cache)
        n = sum(len(b[1]) for b in blocks); npos = sum(int(b[2].sum()) for b in blocks)
        print(f"[{r['row']}] {n:,} scorable sites ({npos:,} positive)")
        for t in only_t:
            res = score_row(r, blocks, t)
            long.append({'row': r['row'], 'chem': r['chem'], 'subtool': t, **res})
            if res['status'] != 'pending':
                print(f"   {t:22s} {res['status']:12s} AUROC {res.get('auroc', np.nan):.4f}  F1 {res.get('f1', np.nan):.4f}"
                      f"  called {res.get('frac_called', np.nan):.2f}")
    df = pd.DataFrame(long)
    out = C.WORK / f'matrix_long_{C.TAG}.tsv'
    if out.exists() and (only_r or a.subtools):          # partial run: keep the cells it did not touch
        old = pd.read_csv(out, sep='\t')
        old = old[~old.set_index(['row', 'subtool']).index.isin(df.set_index(['row', 'subtool']).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, sep='\t', index=False, float_format='%.4f')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
