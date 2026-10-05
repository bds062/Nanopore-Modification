#!/usr/bin/env python3
"""Motif-labelled rows built from the reference alone -> $BENCH_WORK/rows/<row>/{gt.bed,cand.bed,PROVENANCE.txt}.

config/motifs.tsv: row, ref_sample (whose reference gives the coordinates), motif (IUPAC), offset (0-based index
of the modified base in the motif as written), design.

  gt   = every occurrence of the motif on either strand; on the + strand the modified base is at start+offset, on
         the - strand (motif found as its reverse complement) at the mirrored position, whose + strand reference
         base is the complement.
  cand = design 'one':  every reference position whose base is the modified base or its complement (the row's
                        negatives are then every other such base in the same sample; sequence-confounded, see
                        bench.nulls)
         design 'two':  = gt (the same positions are scored in a control sample: knockout, WGA, ...)

  python -m bench.motif_rows [--rows r1,r2]
"""
import argparse
import re

import numpy as np
import pysam

from . import common as C

IUPAC = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'W': '[AT]', 'S': '[CG]', 'R': '[AG]', 'Y': '[CT]', 'K': '[GT]',
         'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]', 'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'}
RC = str.maketrans('ACGTWSRYKMBDHVN', 'TGCAWSYRMKVHDBN')


def positions(seq, motif, off):
    L = len(motif)
    out = set()
    for m in re.finditer(f'(?=({"".join(IUPAC[b] for b in motif)}))', seq):
        out.add(m.start() + off)
    rc = motif.translate(RC)[::-1]
    for m in re.finditer(f'(?=({"".join(IUPAC[b] for b in rc)}))', seq):
        out.add(m.start() + L - 1 - off)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', default='')
    a = ap.parse_args()
    only = set(filter(None, a.rows.split(',')))
    S = C.samples()
    for row, smp, motif, off, design in C._rows(C.CONFIG / 'motifs.tsv'):
        if only and row not in only:
            continue
        off = int(off)
        base = motif[off]
        fa = pysam.FastaFile(S[smp]['ref'])
        R = C.WORK / 'rows' / row
        R.mkdir(parents=True, exist_ok=True)
        n_gt = n_c = 0
        with open(R / 'gt.bed', 'w') as g, open(R / 'cand.bed', 'w') as c:
            for ctg in fa.references:
                seq = fa.fetch(ctg).upper()
                gt = np.array(sorted(positions(seq, motif, off)), dtype=np.int64)
                for p in gt:
                    g.write(f'{ctg}\t{p}\t{p + 1}\n')
                n_gt += len(gt)
                if design == 'two':
                    cand = gt
                else:
                    arr = np.frombuffer(seq.encode(), dtype='S1')
                    cand = np.flatnonzero((arr == base.encode()) | (arr == base.translate(RC).encode()))
                for p in cand:
                    c.write(f'{ctg}\t{p}\t{p + 1}\n')
                n_c += len(cand)
        (R / 'PROVENANCE.txt').write_text(
            f'built by bench.motif_rows from the reference of {smp} ({S[smp]["ref"]})\nmotif {motif}, modified base '
            f'{base} at offset {off}, both strands; design {design}\n{n_gt:,} motif positions, {n_c:,} candidates\n')
        print(f'[{row}] {motif}[{off}] on {smp}: {n_gt:,} gt, {n_c:,} cand ({design})')


if __name__ == '__main__':
    main()
