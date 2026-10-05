#!/usr/bin/env python3
"""Positions RawMod must featurize for one sample: every row position scored on that sample, split by the
strand bench.score assigns it (rows.tsv base rule), restricted to sites scorable on that strand.

Writes $BENCH_WORK/<sample>/rawmod/<TAG>/cand_plus.bed and cand_minus.bed (contig, pos, pos+1).
  python -m bench.rawmod_cands --sample ecoli_wt
"""
import argparse
from pathlib import Path

import pandas as pd

from . import common as C
from . import score as Sc


def build(sample, out=None):
    # exactly the sites bench.score scores (score.row_sites, incl. the one-sample negative cap), nothing more
    S, cache, parts = C.samples(), {}, []
    for r in Sc.rows_cfg():
        if sample not in (r['pos_sample'], r['neg_sample']):
            continue
        need = [r['pos_sample']] + ([r['neg_sample']] if r['neg_sample'] else [])
        if not all(C.readsel_path(s).exists() for s in need):
            print(f"[{r['row']}] read selection missing for {need}; skipped")
            continue
        parts += [sites[['contig', 'pos', 'strand']] for smp, sites, _ in Sc.row_sites(r, S, cache) if smp == sample]
    out = Path(out) if out else C.WORK / sample / 'rawmod' / C.TAG
    out.mkdir(parents=True, exist_ok=True)
    allp = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame(columns=['contig', 'pos', 'strand'])
    for st, name in (('+', 'plus'), ('-', 'minus')):
        x = allp[allp['strand'] == st].sort_values(['contig', 'pos'])
        x.assign(end=x['pos'] + 1)[['contig', 'pos', 'end']].to_csv(out / f'cand_{name}.bed', sep='\t', header=False, index=False)
        print(f'{sample} {name}: {len(x):,} positions')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', nargs='+', required=True)
    for s in ap.parse_args().sample:
        build(s)


if __name__ == '__main__':
    main()
