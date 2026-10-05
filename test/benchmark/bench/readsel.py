#!/usr/bin/env python3
"""Choose, for every (contig, pos, strand) of a sample, the K reads that every tool is scored on.

Reads are the primary, mapped alignments of the sample's sub.bam. They are visited in increasing
read_rank order and each one claims every aligned (match/mismatch) reference position on its strand
where fewer than K reads have claimed it so far -- so each site ends up with its K lowest-ranked
covering reads, independently of which tool is being scored.

Output (npz, see common.readsel_path):
  read_ids                 str[R]     read index -> read id
  sel_read, sel_contig,    int[N]     the positions each read was selected at, as half-open
  sel_start, sel_end                  intervals [start, end) on that read's own contig
  read_strand              int8[R]    0 = forward, 1 = reverse
  contigs                  str[C]
  site_contig, site_strand,int[M]     intervals of sites that reached K reads (the scorable sites)
  site_start, site_end

  python -m bench.readsel --sample ecoli_wt
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pysam

from . import common as C


def runs(mask: np.ndarray, pos: np.ndarray):
    """Half-open intervals of consecutive positions where mask is True (pos sorted, unique)."""
    if not mask.any():
        return []
    p = pos[mask]
    brk = np.nonzero(np.diff(p) != 1)[0]
    starts = np.r_[p[0], p[brk + 1]]
    ends = np.r_[p[brk], p[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def build(sample: str, k: int = C.K, seed: int = C.SEED):
    s = C.samples()[sample]
    t0 = time.time()
    reads = []                                     # (rank, read_id, contig, strand, blocks)
    n_lowq = 0
    with pysam.AlignmentFile(s['sub_bam'], 'rb') as bam:
        contigs = list(bam.references)
        lens = dict(zip(bam.references, bam.lengths))
        for r in bam.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if r.get_tag('qs') < C.MIN_QS:
                n_lowq += 1
                continue
            reads.append((C.read_rank(r.query_name, seed), r.query_name, r.reference_name,
                          int(r.is_reverse), r.get_blocks()))
    reads.sort(key=lambda x: x[0])
    if len({x[1] for x in reads}) != len(reads):
        sys.exit(f'{sample}: duplicate primary read ids in {s["sub_bam"]}')
    # one dense uint8 counter per touched contig and strand, over that contig's covered span
    span = {}
    for _, _, ctg, _, blocks in reads:
        lo, hi = blocks[0][0], blocks[-1][1]
        a, b = span.get(ctg, (lo, hi))
        span[ctg] = (min(a, lo), max(b, hi))
    cnt = {c: np.zeros((2, b - a), dtype=np.uint8) for c, (a, b) in span.items()}
    ci = {c: i for i, c in enumerate(contigs)}
    sel = []
    for idx, (_, rid, ctg, st, blocks) in enumerate(reads):
        off = span[ctg][0]
        pos = np.concatenate([np.arange(a, b) for a, b in blocks]) if blocks else np.zeros(0, int)
        pos = np.unique(pos)
        c = cnt[ctg][st, pos - off]
        m = c < k
        cnt[ctg][st, pos[m] - off] += 1
        for a, b in runs(m, pos):
            sel.append((idx, ci[ctg], a, b))
    site = []
    for ctg, arr in cnt.items():
        off = span[ctg][0]
        for st in (0, 1):
            full = np.nonzero(arr[st] >= k)[0]
            for a, b in runs(np.ones(len(full), bool), full):
                site.append((ci[ctg], st, a + off, b + off))
    sel = np.array(sel, dtype=np.int64).reshape(-1, 4)
    site = np.array(site, dtype=np.int64).reshape(-1, 4)
    out = C.readsel_path(sample)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, read_ids=np.array([x[1] for x in reads]), read_strand=np.array([x[3] for x in reads], np.int8),
        contigs=np.array(contigs), sel_read=sel[:, 0], sel_contig=sel[:, 1], sel_start=sel[:, 2],
        sel_end=sel[:, 3], site_contig=site[:, 0], site_strand=site[:, 1], site_start=site[:, 2],
        site_end=site[:, 3], k=k, seed=seed, min_qs=C.MIN_QS)
    n_sites = int((site[:, 3] - site[:, 2]).sum()) if len(site) else 0
    print(f'{sample}: {len(reads):,} primary reads with qs >= {C.MIN_QS:g} ({n_lowq:,} below dropped), {n_sites:,} site/strands with >= {k} reads '
          f'-> {out}  ({time.time() - t0:.0f}s)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', nargs='+', required=True)
    a = ap.parse_args()
    for s in a.sample:
        build(s)


if __name__ == '__main__':
    main()
