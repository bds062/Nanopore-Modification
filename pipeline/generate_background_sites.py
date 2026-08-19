#!/usr/bin/env python3
"""Generate non-motif "background" candidate sites for the 6 motif-saturated
bacterial benchmark datasets (results15's EXTRA_ORGANISMS pool). These
datasets are 100% positive because motif_gt.py's candidate.bed only ever
lists motif-matching (near-100%-methylated) positions -- there's no
complementary "confidently unmethylated" population. This script builds
one by reusing the exact motif hit-set each dataset's GT was derived from
(same regex/offset/strand as motif_gt.py) to find every occurrence of the
SAME target base (A for 6mA systems, C for 5mC/4mC systems) that is NOT
inside any known motif -- a genuine background/negative population, same
base chemistry as the real candidates, just outside the recognition motif.

Output: candidate_background.bed (chrom\\tpos) in each dataset's GT dir,
capped and uniformly subsampled to CAP_PER_ORG positions. No
gt_background.bed is written -- featurization.py with no --gt sets every
label to 0, which is exactly what we want here.

Usage: python generate_background_sites.py
"""
import gzip
import re
import sys
from pathlib import Path

import numpy as np

BENCH_REF = '/fs/cbcb-lab/storm/bds062/data/benchmark/references'
GT_ROOT = '/fs/cbcb-scratch/bds062/data/gt'
SEED = 0
CAP_PER_ORG = 20000

# gt_name -> (reference fasta, [(motif_iupac, offset, strand, type), ...])
# (identical to recompute_bench_types.py's TYPED_MOTIFS -- kept in sync)
TYPED_MOTIFS = {
    'anabaena':       (f'{BENCH_REF}/anabaena_sp_PCC7120_ATCC27893.fa.gz',
                       [('GATC', 1, 'both', '6mA')]),
    'Ecoli_DM':       (f'{BENCH_REF}/ecoli.fa.gz',
                       [('GATC', 1, 'both', '6mA')]),
    'Ecoli_DM_MSssI': (f'{BENCH_REF}/ecoli.fa.gz',
                       [('GATC', 1, 'both', '6mA'), ('CG', 0, 'both', '5mC')]),
    'Ecoli_WT':       (f'{BENCH_REF}/ecoli.fa.gz',
                       [('GATC', 1, 'both', '6mA'), ('CCWGG', 1, 'both', '5mC')]),
    'tdenticola':     (f'{BENCH_REF}/treponema_denticola_ATCC35405.fa.gz',
                       [('GATC', 1, 'both', '6mA'), ('TATAC', 1, '+', '6mA'),
                        ('GTATA', 3, '+', '6mA')]),
    'hpylori_j99':    (f'{BENCH_REF}/hpylori_J99_ATCC700824.fa.gz',
                       [('GTNNNNNNAC', 1, '+', '6mA'), ('TCNNNNNNNGC', 1, '+', '4mC')]),
}
TYPE_TO_BASE = {'6mA': 'A', '5mC': 'C', '4mC': 'C'}

_IUPAC = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'R': '[AG]', 'Y': '[CT]',
         'S': '[GC]', 'W': '[AT]', 'K': '[GT]', 'M': '[AC]', 'B': '[CGT]',
         'D': '[AGT]', 'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'}
_COMP = str.maketrans('ACGT', 'TGCA')


def iupac_to_regex(motif):
    return ''.join(_IUPAC.get(c.upper(), c) for c in motif)


def revcomp(seq):
    return seq.translate(_COMP)[::-1]


def open_fasta(path):
    return gzip.open(path, 'rt') if path.endswith(('.gz', '.bgz')) else open(path)


def motif_hit_positions(seq, motifs):
    """All forward-strand coordinates hit by ANY of this organism's motifs."""
    hits = set()
    for motif_str, offset, strand, _t in motifs:
        rx = re.compile(iupac_to_regex(motif_str), re.IGNORECASE)
        for m in rx.finditer(seq):
            hits.add(m.start() + offset)
        if strand == 'both':
            rc = revcomp(seq)
            rc_len = len(rc)
            for m in rx.finditer(rc):
                hits.add(rc_len - (m.start() + offset) - 1)
    return hits


def main():
    rng = np.random.default_rng(SEED)
    for gt_name, (ref_path, motifs) in TYPED_MOTIFS.items():
        target_bases = {TYPE_TO_BASE[t] for *_x, t in motifs}
        print(f"=== {gt_name}  (target bases: {sorted(target_bases)}) ===", flush=True)

        bg = []  # (chrom, pos)
        chrom, seq_parts = None, []

        def flush():
            if chrom is None:
                return
            seq = ''.join(seq_parts).upper()
            hits = motif_hit_positions(seq, motifs)
            for i, base in enumerate(seq):
                if base in target_bases and i not in hits:
                    bg.append((chrom, i))

        with open_fasta(ref_path) as fh:
            for line in fh:
                line = line.rstrip('\n')
                if line.startswith('>'):
                    flush()
                    chrom = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line)
            flush()

        print(f"  {len(bg):,} non-motif background positions found", flush=True)
        if len(bg) > CAP_PER_ORG:
            idx = rng.choice(len(bg), CAP_PER_ORG, replace=False)
            bg = [bg[i] for i in sorted(idx)]
        bg.sort()

        outdir = Path(GT_ROOT) / gt_name
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / 'candidate_background.bed'
        with open(out_path, 'w') as f:
            for c, p in bg:
                f.write(f"{c}\t{p}\n")
        print(f"  wrote {len(bg):,} -> {out_path}\n", flush=True)


if __name__ == '__main__':
    main()
