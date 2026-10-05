#!/usr/bin/env python3
"""Per-read UniMeth fine-tuning labels at EXACTLY the positions the benchmarked RawMod checkpoint trained on.

Writes MM/ML tags into a copy of a benchmark sample's BAM (the benchmark's basecall, which carries the move tables UniMeth
needs). Only bases of --base (in the read's own orientation) whose (contig, pos, strand) is a RawMod training position
of the sample's organism (bench.holdout source table, split == 'train') are listed: ML 255 if the sample is the
modified library of the pair and the position is a positive for this modification, ML 0 if it is a training negative.
Every other base is left out of the MM tag ('?' mode), which UniMeth's fine-tuner turns into [MASK] with zero loss
weight, so the model learns from RawMod's training positions only and the benchmark's held-out positions stay unseen.

  --state modified   : label 255 at training positions that are positives (label > 0) in the pool files matching
                       --pos-files and, if --gt is given, are listed in it (e.g. the 4mC annotation: H. pylori WT
                       positives also include 6mA/5mC, which are left unlabelled); label 0 at every other training
                       position of the organism
  --state unmodified : label 0 at every training position of the organism (control library)

  python -m bench.unimeth_ft_labels --sample syn_5hmC --state modified --base C --code h \
      --org-files '^ONT::' --pos-files '^ONT::5hmC' --out syn_5hmC.ft.bam
"""
import argparse
import os
from array import array

import numpy as np
import pandas as pd
import pysam

from . import common as C
from .holdout import SEEN
from .settings import DATA_DIR

COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sample', required=True)
    ap.add_argument('--state', choices=['modified', 'unmodified'], required=True)
    ap.add_argument('--base', required=True, choices=list('ACGT'))
    ap.add_argument('--code', required=True, help='SAM mod code written to MM (h, 21839, g, ...)')
    ap.add_argument('--org-files', required=True, help='regex: every training-pool file of this organism')
    ap.add_argument('--pos-files', default='', help='regex: files whose label>0 training images are positives')
    ap.add_argument('--gt', default='', help='optional BED (contig, pos): positives must also be listed here')
    ap.add_argument('--all-modified', action='store_true', help='modified library where EVERY --base is modified (native SPO1: every T is 5hmU): label 255 at every training position of the organism')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    d = pd.read_csv(DATA_DIR / 'holdout' / SEEN['r81'], sep='\t', dtype={'contig': str})
    d = d[(d['split'] == 'train') & d['file'].str.contains(a.org_files, regex=True)]
    allpos = set(zip(d['contig'], d['pos'].astype(int), d['strand']))
    pos = set()
    if a.state == 'modified':
        p = d[d['file'].str.contains(a.pos_files, regex=True) & (d['label'] > 0)]
        pos = set(zip(p['contig'], p['pos'].astype(int), p['strand']))
        if a.gt:
            g = pd.read_csv(a.gt, sep='\t', header=None, usecols=[0, 1], names=['c', 'p'], dtype={'c': str})
            gt = set(zip(g['c'], g['p'].astype(int)))
            pos = {k for k in pos if (k[0], k[1]) in gt}
    # a training position that is a positive of this library but not of this modification (e.g. a 6mA site when
    # training 4mC) is left unlabelled rather than called negative
    other_pos = set()
    if a.state == 'modified':
        q = d[d['file'].str.contains(a.pos_files, regex=True) & (d['label'] > 0)]
        other_pos = set(zip(q['contig'], q['pos'].astype(int), q['strand'])) - pos
    if a.all_modified:
        pos, other_pos = set(allpos), set()
    neg = allpos - pos - other_pos
    print(f'{a.sample}: {len(allpos):,} training (contig,pos,strand) of the organism; labelling {len(pos):,} as '
          f'modified, {len(neg):,} as unmodified, {len(other_pos):,} other-modification positives left out', flush=True)

    src = C.samples()[a.sample]['sub_bam']
    tmp = a.out + '.unsorted.bam'
    n_reads = n_lab = n_mod = n_mv = 0
    with pysam.AlignmentFile(src, 'rb') as fin, pysam.AlignmentFile(tmp, 'wb', template=fin) as fout:
        for r in fin:
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            if r.has_tag('mv') and sum(r.get_tag('mv')[1:]) != r.query_length:
                n_mv += 1; continue                       # UniMeth's extractor rejects these (move table != sequence)
            fseq = (r.get_forward_sequence() or '').upper()
            if not fseq:
                continue
            L, st, ctg = len(fseq), '-' if r.is_reverse else '+', r.reference_name
            q2r = dict(r.get_aligned_pairs(matches_only=True))
            idx = [i for i, b in enumerate(fseq) if b == a.base]
            keep, ml = [], []
            for j, i in enumerate(idx):
                p = q2r.get(L - 1 - i if r.is_reverse else i)
                if p is None:
                    continue
                k = (ctg, p, st)
                if k in pos:
                    keep.append(j); ml.append(255); n_mod += 1
                elif k in neg:
                    keep.append(j); ml.append(0)
            r.set_tag('MM', None); r.set_tag('ML', None)
            if keep:
                skips = np.diff(np.r_[-1, keep]) - 1
                r.set_tag('MM', f"{a.base}+{a.code}?," + ','.join(map(str, skips)) + ';', 'Z')
                r.set_tag('ML', array('B', ml))
                n_lab += len(keep)
            fout.write(r); n_reads += 1
    pysam.sort('-@', '4', '-o', a.out, tmp); os.remove(tmp); pysam.index(a.out)
    print(f'{a.out}: {n_reads:,} reads, {n_lab:,} labelled {a.base} ({n_mod:,} modified); {n_mv:,} reads dropped '
          f'for a move table not matching the sequence')


if __name__ == '__main__':
    main()
