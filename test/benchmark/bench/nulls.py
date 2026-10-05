#!/usr/bin/env python3
"""Trivial baselines for every row, on exactly the sites bench.score scores -> $BENCH_WORK/nulls_k*_seed*.tsv.

  ref11     logistic regression on the one-hot reference 11-mer around the site (on the site's own strand),
            5-fold CV AUROC, folds grouped by position (a two-sample row's pos and neg copy of a site never
            straddle folds). Measures how far the label is a property of the sequence alone. On two-sample rows
            both classes are the same positions, so this is ~0.5 by construction; motif-preset rows with
            genome-wide negatives sit near 1.0 -- there a high tool AUROC says little about signal modelling.
  cpg       AUROC of "is the site a CpG" (on its strand).
At most 50,000 sites per row (seeded) go into the regression.

  python -m bench.nulls [--rows r1,r2]
"""
import argparse

import numpy as np
import pandas as pd
import pysam
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from . import common as C
from . import score as Sc

W = 5
IDX = {b: i for i, b in enumerate('ACGT')}
RC = str.maketrans('ACGT', 'TGCA')


def context(fa, contig, pos, strand):
    s = fa.fetch(contig, max(pos - W, 0), pos + W + 1).upper()
    if pos - W < 0:
        s = 'N' * (W - pos) + s
    s = s.ljust(2 * W + 1, 'N')
    return s.translate(RC)[::-1] if strand == '-' else s


def onehot(seqs):
    X = np.zeros((len(seqs), (2 * W + 1) * 4), np.float32)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s):
            if b in IDX:
                X[i, j * 4 + IDX[b]] = 1
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', default='')
    a = ap.parse_args()
    only = set(filter(None, a.rows.split(',')))
    S, cache, out = C.samples(), {}, []
    for r in Sc.rows_cfg():
        if only and r['row'] not in only:
            continue
        blocks = Sc.row_sites(r, S, cache)
        seqs, y, grp = [], [], []
        rng = np.random.default_rng(0)
        for smp, sites, lab in blocks:
            fa = pysam.FastaFile(S[smp]['ref'])
            take = np.arange(len(sites))
            if len(take) > 50000 // len(blocks):
                take = np.sort(rng.choice(len(take), 50000 // len(blocks), replace=False))
            for i in take:
                x = sites.iloc[i]
                seqs.append(context(fa, x['contig'], int(x['pos']), x['strand']))
                y.append(lab[i])
                grp.append(f"{x['contig']}:{x['pos']}:{x['strand']}")
        y = np.array(y)
        res = {'row': r['row'], 'n': len(y), 'n_pos': int(y.sum())}
        if len(y) and 0 < y.sum() < len(y):
            X = onehot(seqs)
            k = min(5, int(min(y.sum(), len(y) - y.sum())))
            if k >= 2:
                p = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=StratifiedGroupKFold(k, shuffle=True, random_state=0),
                                      groups=pd.factorize(pd.Series(grp))[0], method='predict_proba')[:, 1]
                res['ref11'] = roc_auc_score(y, p)
            cpg = np.array([s[W:W + 2] == 'CG' for s in seqs], float)
            res['cpg'] = roc_auc_score(y, cpg) if cpg.min() != cpg.max() else 0.5
        out.append(res)
        print(f"[{r['row']}] n={res['n']:,} ref11={res.get('ref11', np.nan):.3f} cpg={res.get('cpg', np.nan):.3f}")
    path = C.WORK / f'nulls_{C.PAPER_TAG}.tsv'
    df = pd.DataFrame(out)
    if only and path.exists():                            # partial run: keep the rows it did not touch
        old = pd.read_csv(path, sep='\t')
        df = pd.concat([old[~old['row'].isin(df['row'])], df], ignore_index=True)
    df.to_csv(path, sep='\t', index=False, float_format='%.4f')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
