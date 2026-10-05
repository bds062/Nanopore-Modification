"""Positions the benchmarked RawMod checkpoint trained on, per benchmark sample (BENCH_HOLDOUT=r81).

Source: test/data/holdout/rawmod_seen_results81_mixed.tsv.gz, written by bench.rawmod_seen (every image of the training
pool with its split). A position counts as SEEN if any training image (split == 'train') of the same organism sits on it, on either
strand and under either label. config/holdout_r81.tsv maps each benchmark sample to a regular expression over the
training-pool file names of its organism; samples with no entry (mouse, rice) have no training data and keep every site.
Organism matching matters because contig names repeat across organisms (human and mouse both have 'chr1').
"""
from functools import lru_cache

import pandas as pd

from . import common as C
from .settings import DATA_DIR

SEEN = {'r81': 'rawmod_seen_results81_mixed.tsv.gz'}


@lru_cache(maxsize=1)
def _table():
    d = pd.read_csv(DATA_DIR / 'holdout' / SEEN[C.HOLDOUT], sep='\t', usecols=['file', 'contig', 'pos', 'split'],
                    dtype={'file': str, 'contig': str, 'pos': 'int64', 'split': str})
    return d[d['split'] == 'train']


@lru_cache(maxsize=1)
def _map():
    return {r[0]: r[1] for r in C._rows(C.CONFIG / f'holdout_{C.HOLDOUT}.tsv')}


@lru_cache(maxsize=None)
def _seen_for(sample):
    pat = _map().get(sample)
    if not pat:
        return frozenset()
    d = _table()
    d = d[d['file'].str.contains(pat, regex=True)]
    return frozenset(zip(d['contig'], d['pos'].astype(int)))


def seen_positions(samples):
    """Union of trained (contig, pos) over the organisms of `samples`."""
    out = set()
    for s in samples:
        out |= _seen_for(s)
    return out
