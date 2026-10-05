#!/usr/bin/env python3
"""Per-read calls of one (sample, subtool) -> per-site scores over the sample's K selected reads.

Keeps a call only if its read is one of the K selected reads at that (contig, pos, strand) -- the strand
being the read's orientation in sub.bam -- then sums over those reads. Output (common.sites_path):

  contig  pos  strand  n_called  sum_p  n_mod      (only sites with >= 1 kept call)

n_called <= K. Selected reads with no call are unmodified by convention, so for a scorable site
mean_P = sum_p / K and call_freq = n_mod / K (n_mod = calls with prob > 0.5); sites absent from the
file score 0. Several calls of one read at one site (duplicated output) keep the maximum.

A coordinate check prints the reference base under the kept calls per strand: a C-model must sit on C
for forward reads and G for reverse reads, an A-model on A / T. Anything else is an adapter bug.

  python -m bench.aggregate --sample ecoli_wt --subtool unimeth_6mA
"""
from __future__ import annotations

import argparse
import collections
import time

import numpy as np
import pandas as pd
import pysam

from . import adapters
from . import common as C


class Selection:
    def __init__(self, sample):
        z = np.load(C.readsel_path(sample), allow_pickle=False)
        self.k = int(z['k'])
        self.read_idx = {r: i for i, r in enumerate(z['read_ids'].tolist())}
        self.read_strand = z['read_strand']
        self.contigs = z['contigs'].tolist()
        order = np.lexsort((z['sel_start'], z['sel_read']))
        self.r, self.c = z['sel_read'][order], z['sel_contig'][order]
        self.s, self.e = z['sel_start'][order], z['sel_end'][order]
        self.key = (self.r << 36) + self.s

    def keep(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows of df (read_id, contig, pos, prob) whose read is selected at that position; adds strand."""
        ri = df['read_id'].map(self.read_idx)
        df = df[ri.notna()].copy()
        ri = ri[ri.notna()].astype(np.int64).to_numpy()
        ci = df['contig'].map({c: i for i, c in enumerate(self.contigs)}).fillna(-1).astype(np.int64).to_numpy()
        pos = df['pos'].to_numpy(np.int64)
        j = np.searchsorted(self.key, (ri << 36) + pos, side='right') - 1
        ok = j >= 0
        jj = np.where(ok, j, 0)
        ok &= (self.r[jj] == ri) & (self.c[jj] == ci) & (pos < self.e[jj])
        df = df[ok]
        df['strand'] = np.where(self.read_strand[ri[ok]] == 1, '-', '+')
        return df


def run(sample: str, subtool: str):
    t0 = time.time()
    S, T = C.samples()[sample], C.subtools()[subtool]
    ad = adapters.get(T['family'])
    if getattr(ad, 'PER_SITE', False):
        return run_per_site(S, T, ad, t0)
    sel = Selection(sample)
    acc = []
    n_in = 0
    for df in ad.iter_calls(S, T):
        n_in += len(df)
        kept = sel.keep(df)
        if len(kept):
            # one call per (read, site): keep the max if a tool reported a site twice for a read
            kept = kept.groupby(['read_id', 'contig', 'pos', 'strand'], sort=False)['prob'].max().reset_index()
            acc.append(kept)
    if not acc:
        print(f'{sample}/{subtool}: {n_in:,} calls in, NONE on the selected reads')
        cols = ['contig', 'pos', 'strand', 'n_called', 'sum_p', 'n_mod']
        out = pd.DataFrame(columns=cols)
    else:
        allc = pd.concat(acc, ignore_index=True)
        allc = allc.groupby(['read_id', 'contig', 'pos', 'strand'], sort=False)['prob'].max().reset_index()
        allc['mod'] = (allc['prob'] > 0.5).astype(np.int64)
        out = (allc.groupby(['contig', 'pos', 'strand'], sort=True)
               .agg(n_called=('prob', 'size'), sum_p=('prob', 'sum'), n_mod=('mod', 'sum')).reset_index())
        print(f'{sample}/{subtool}: {n_in:,} calls in, {len(allc):,} on selected reads, {len(out):,} sites')
        coord_check(S['ref'], out)
    out['pos'] = out['pos'].astype(np.int64)        # never let %.6g round a float position
    path = C.sites_path(sample, subtool)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep='\t', index=False, float_format='%.6g')
    print(f'  -> {path}  ({time.time() - t0:.0f}s)')
    return path


def run_per_site(S, T, ad, t0):
    """Tools that score a site from the selected reads themselves (RawMod): write their score in the
    common format, sum_p = score * K and n_mod = K * (score >= 0.5), n_called = reads in the image."""
    df = pd.concat(list(ad.iter_sites(S, T)), ignore_index=True)
    df = df.groupby(['contig', 'pos', 'strand'], sort=True).agg(score=('score', 'max'), n_reads=('n_reads', 'max')).reset_index()
    out = pd.DataFrame({'contig': df['contig'], 'pos': df['pos'], 'strand': df['strand'],
                        'n_called': df['n_reads'], 'sum_p': df['score'] * C.K,
                        'n_mod': np.where(df['score'] >= 0.5, C.K, 0)})
    print(f"{S['sample']}/{T['subtool']}: {len(out):,} sites scored "
          f"(image depth: {(df['n_reads'] == C.K).mean():.1%} have all {C.K} selected reads)")
    coord_check(S['ref'], out)
    out['pos'] = out['pos'].astype(np.int64)        # never let %.6g round a float position
    path = C.sites_path(S['sample'], T['subtool'])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep='\t', index=False, float_format='%.6g')
    print(f'  -> {path}  ({time.time() - t0:.0f}s)')
    return path


def coord_check(ref, sites, n=20000):
    fa = pysam.FastaFile(ref)
    sub = sites.sample(min(n, len(sites)), random_state=0)
    for st in ('+', '-'):
        x = sub[sub['strand'] == st]
        c = collections.Counter(fa.fetch(r.contig, r.pos, r.pos + 1).upper() for r in x.itertuples())
        tot = sum(c.values()) or 1
        print(f'  coord-check strand {st}: ' + ' '.join(f'{b}={v / tot:.2f}' for b, v in sorted(c.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', nargs='+', required=True)
    ap.add_argument('--subtool', nargs='+', required=True)
    a = ap.parse_args()
    for s in a.sample:
        for t in a.subtool:
            run(s, t)


if __name__ == '__main__':
    main()
