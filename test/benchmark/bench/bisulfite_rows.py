#!/usr/bin/env python3
"""Rebuild the bisulfite-labelled rows under ONE rule -> $BENCH_WORK/rows/<row>/{gt.bed,cand.bed,PROVENANCE.txt}.

  * per-strand bisulfite counts, coverage >= MIN_COV
  * positive: methylated fraction >= POS; negative: <= NEG; anything in between is dropped
  * both strands (a position's strand is re-derived from the reference base by bench.score)
  * the row's context only (cpg: CG; noncpg: CHG + CHH), and cand = positives + negatives, i.e. assayed sites only
  * only positions inside the sample's scorable span (sites with >= K selected reads on either strand), to keep the
    BEDs small; nothing outside it could be scored anyway

Sources are listed in config/bisulfite.tsv (paths relative to $BENCH_DATA). Format 'ontbasemod': the EM-seq dataframes of
Kulkarni et al. (chr start end n_meth n_unmeth strand cov pct ctx kmer, 0-based start). Format 'bismark': Bismark
CpG.bismark.zero.cov (chr start end pct meth unmeth, 0-based start, CpG only, strand from the reference base).

The rows in test/data/rows/ were built with this script; `rawmod_bench.py run` installs those files and does not rerun it.

  python -m bench.bisulfite_rows [--rows r1,r2]
"""
import argparse
import gzip

import numpy as np
import pandas as pd
import pysam

from pathlib import Path

from . import common as C
from . import settings as ST
from .score import Scorable

MIN_COV, POS, NEG = 20, 0.9, 0.1


def read_source(path, fmt, contigs):
    op = gzip.open if path.endswith('.gz') else open
    cols = {'ontbasemod': ([0, 1, 3, 4, 5, 8], ['contig', 'pos', 'meth', 'unmeth', 'strand', 'ctx']),
            'bismark': ([0, 1, 4, 5], ['contig', 'pos', 'meth', 'unmeth'])}[fmt]
    out = []
    with op(path, 'rt') as fh:
        for ch in pd.read_csv(fh, sep='\t', header=None, usecols=cols[0], names=cols[1], dtype={'contig': str},
                              chunksize=5_000_000):
            out.append(ch[ch['contig'].isin(contigs)])
    d = pd.concat(out, ignore_index=True)
    if fmt == 'bismark':
        d['ctx'] = 'CG'
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', default='')
    a = ap.parse_args()
    only = set(filter(None, a.rows.split(',')))
    S = C.samples()
    for row, sample, ctx, fmt, src in C._rows(C.CONFIG / 'bisulfite.tsv'):
        if only and row not in only:
            continue
        if sample not in S:
            print(f'[{row}] {sample} not in the samples list; skipped')
            continue
        if not C.readsel_path(sample).exists():
            print(f'[{row}] no read selection for {sample} yet ({C.readsel_path(sample)}); skipped')
            continue
        sc = Scorable(sample)
        src = str(Path(ST.need('BENCH_DATA', 'bisulfite sources')) / src)
        d = read_source(src, fmt, sc.contigs)
        n0 = len(d)
        d = d[d['ctx'].isin(['CG'] if ctx == 'cpg' else ['CHG', 'CHH'])]
        cov = d['meth'] + d['unmeth']
        d = d[cov >= MIN_COV].assign(frac=lambda x: x['meth'] / (x['meth'] + x['unmeth']))
        keep = []
        for contig, g in d.groupby('contig'):
            p = g['pos'].to_numpy()
            keep.append(g[sc.mask(contig, '+', p) | sc.mask(contig, '-', p)])
        d = pd.concat(keep, ignore_index=True)
        # sanity: the labelled base must be C (+) or G (-) in this sample's reference
        fa, sub = pysam.FastaFile(S[sample]['ref']), d.sample(min(5000, len(d)), random_state=0)
        ok = np.mean([fa.fetch(c, p, p + 1).upper() in 'CG' for c, p in zip(sub['contig'], sub['pos'])])
        pos, neg = d[d['frac'] >= POS], d[d['frac'] <= NEG]
        R = C.WORK / 'rows' / row
        R.mkdir(parents=True, exist_ok=True)
        for name, x in (('gt.bed', pos), ('cand.bed', pd.concat([pos, neg]))):
            x = x[['contig', 'pos']].drop_duplicates().sort_values(['contig', 'pos'])
            x.assign(end=x['pos'] + 1).to_csv(R / name, sep='\t', header=False, index=False)
        (R / 'PROVENANCE.txt').write_text(
            f'built by bench.bisulfite_rows from {Path(src).name} ({fmt})\ncontext {ctx}; coverage >= {MIN_COV}; positive frac >= {POS}; '
            f'negative frac <= {NEG}; both strands; restricted to {sample} scorable span (K={C.K}, seed={C.SEED}, '
            f'min_qs={C.MIN_QS:g})\n{len(pos):,} positives, {len(neg):,} negatives, {len(d) - len(pos) - len(neg):,} '
            f'intermediate dropped; reference base C/G at {ok:.1%} of labelled positions\n')
        print(f'[{row}] {n0:,} source rows on {sample} contigs -> {len(pos):,} pos / {len(neg):,} neg in span '
              f'(C/G check {ok:.1%})')


if __name__ == '__main__':
    main()
