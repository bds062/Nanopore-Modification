#!/usr/bin/env python3
"""Write every candidate site of one strand as contiguous chunk files for featurization.

A genome map scores every position whose base matches the target, with no ground
truth, so the candidate list is just the reference scanned for that base. The
output (contig<TAB>0-based pos, one site per line) is what
`rawmod/featurization.py --candidate-bed` reads.

Strand convention (matches the strand-resolved training data):
  + strand: sites are reference positions whose base IS the target
            (5hmU/T -> ref T, 6mA -> ref A, 5mC -> ref C).
  - strand: the read carries the target, so the reference shows its complement
            (T -> ref A, A -> ref T, C -> ref G). Featurize with
            `--strand - --orient read` and `--target-base <complement>`.

Usage:
  python make_candidates.py --ref genome.fa --target-base T --strand + \
      --chunks 24 --out-dir candidates_plus [--contigs contig_1,contig_2]
"""
import argparse
from pathlib import Path

import pysam

COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True, help='reference FASTA (indexed or indexable)')
    ap.add_argument('--target-base', required=True, choices=list(COMP),
                    help='base carrying the modification on the read (T for 5hmU)')
    ap.add_argument('--strand', default='+', choices=['+', '-'])
    ap.add_argument('--chunks', type=int, default=24,
                    help='number of contiguous chunk files (= featurization array size)')
    ap.add_argument('--contigs', default=None,
                    help='comma-separated contigs to include (default: all)')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    ref_base = a.target_base if a.strand == '+' else COMP[a.target_base]
    fa = pysam.FastaFile(a.ref)
    contigs = a.contigs.split(',') if a.contigs else list(fa.references)
    sites = []
    for c in contigs:
        seq = fa.fetch(c).upper()
        sites.extend((c, i) for i, b in enumerate(seq) if b == ref_base)
    if not sites:
        raise SystemExit(f'no {ref_base} positions found in {contigs}')

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = min(a.chunks, len(sites))
    size = -(-len(sites) // n)
    for k in range(n):
        with open(out / f'chunk_{k:02d}.tsv', 'w') as f:
            for c, p in sites[k * size:(k + 1) * size]:
                f.write(f'{c}\t{p}\n')
    print(f'{len(sites):,} reference-{ref_base} sites ({a.strand} strand, target {a.target_base}) '
          f'on {len(contigs)} contig(s) -> {n} chunks in {out}')


if __name__ == '__main__':
    main()
