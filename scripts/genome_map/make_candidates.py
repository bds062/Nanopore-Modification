#!/usr/bin/env python3
"""Write the sites to be scored, split into chunk files for parallel featurization.

By default every position of the target base is used (e.g. every T for 5hmU). With
--sites, only the listed positions are used. Output files contain one site per line
(contig<TAB>0-based position), the format read by `rawmod/featurization.py
--candidate-bed`.

Strand convention (matches the training data):
  + strand: positions where the reference base is the target base (T for 5hmU).
  - strand: the modified base is on the reverse strand, so the reference shows its
            complement (A for 5hmU). These sites are featurized with
            `--strand - --orient read`.

--sites file: tab-separated, 0-based positions. The first two columns are contig and
position; a header line is optional. An optional column named `strand` (+ or -)
assigns sites to strands; without it every listed site is treated as + strand.
Listed sites whose reference base does not match the target base for their strand
are dropped and counted.

Examples:
  python make_candidates.py --ref genome.fa --target-base T --strand + --out-dir cand_plus
  python make_candidates.py --ref genome.fa --target-base T --strand + --sites my_sites.tsv --out-dir cand_plus
"""
import argparse
import csv
from pathlib import Path

import pysam

COMP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def read_sites(path, strand):
    rows = []
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        header = None
        for k, r in enumerate(reader):
            if not r or r[0].startswith('#'):
                continue
            if header is None and k == 0 and not r[1].strip().lstrip('-').isdigit():
                header = [c.strip().lower() for c in r]
                continue
            s_col = header.index('strand') if header and 'strand' in header else None
            s = r[s_col].strip() if s_col is not None else '+'
            if s == strand:
                rows.append((r[0].strip(), int(r[1])))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', required=True, help='reference FASTA')
    ap.add_argument('--target-base', required=True, choices=list(COMP),
                    help='base that carries the modification (T for 5hmU, A for 6mA, C for 5mC)')
    ap.add_argument('--strand', default='+', choices=['+', '-'])
    ap.add_argument('--sites', default=None, help='optional list of positions to score (see above)')
    ap.add_argument('--contigs', default=None, help='comma-separated contigs (default: all)')
    ap.add_argument('--chunks', type=int, default=24, help='number of chunk files')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    ref_base = a.target_base if a.strand == '+' else COMP[a.target_base]
    fa = pysam.FastaFile(a.ref)
    contigs = a.contigs.split(',') if a.contigs else list(fa.references)
    if a.sites:
        keep, bad = [], 0
        for c, p in read_sites(a.sites, a.strand):
            if c in contigs and 0 <= p < fa.get_reference_length(c) and fa.fetch(c, p, p + 1).upper() == ref_base:
                keep.append((c, p))
            else:
                bad += 1
        sites = sorted(set(keep))
        if bad:
            print(f'dropped {bad:,} listed sites that are not a reference {ref_base} on the listed contigs')
    else:
        sites = []
        for c in contigs:
            seq = fa.fetch(c).upper()
            sites.extend((c, i) for i, b in enumerate(seq) if b == ref_base)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('chunk_*.tsv'):
        old.unlink()
    if not sites:
        print(f'no {a.strand}-strand sites to score (reference base {ref_base})')
        return
    n = min(a.chunks, len(sites))
    size = -(-len(sites) // n)
    for k in range(n):
        with open(out / f'chunk_{k:02d}.tsv', 'w') as f:
            for c, p in sites[k * size:(k + 1) * size]:
                f.write(f'{c}\t{p}\n')
    print(f'{len(sites):,} {a.strand}-strand sites (reference base {ref_base}, target {a.target_base}) '
          f'-> {n} chunks in {out}')


if __name__ == '__main__':
    main()
