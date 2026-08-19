#!/usr/bin/env python3
"""
Extract 5mC ground-truth BED files from a Bismark bisulfite BAM.

Parses the XM tag per read to accumulate CpG methylation counts, then
applies the confidence filter (Rockfish-style):
  modified  : coverage >= MIN_COV and fraction >= HI_FRAC  → label=1
  unmodified: coverage >= MIN_COV and fraction <= LO_FRAC  → label=0

Outputs:
  gt_modified.bed  — tab: chrom, 0-based pos  (positives for featurization --gt)
  candidate.bed    — high-confidence positive + negative sites (for --candidate-bed)

XM tag format (Bismark convention):
  Z = methylated CpG     z = unmethylated CpG
  X = methylated CHH     x = unmethylated CHH
  H = methylated CHG     h = unmethylated CHG
  . = non-cytosine position

Usage:
  python extract_gt_bismark.py \\
      --bam Arabidopsis.deduplicated.bam \\
      --outdir /path/to/gt/arabidopsis \\
      [--context CpG]   \\   # CpG (default), CHG, CHH, or all
      [--min-cov 30]    \\
      [--hi-frac 0.90]  \\
      [--lo-frac 0.10]  \\
      [--threads 16]
"""

import argparse
import collections
import os
import sys

import pysam
from tqdm import tqdm


# XM tag characters for each context
_CONTEXTS = {
    'CpG': ('Z', 'z'),
    'CHG': ('H', 'h'),
    'CHH': ('X', 'x'),
    'all': ('Z', 'z', 'H', 'h', 'X', 'x'),
}


def main():
    parser = argparse.ArgumentParser(
        description='Extract 5mC GT BEDs from a Bismark bisulfite BAM')
    parser.add_argument('--bam', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--context', default='CpG',
                        choices=['CpG', 'CHG', 'CHH', 'all'],
                        help='Cytosine context to extract (default: CpG)')
    parser.add_argument('--min-cov', type=int, default=30)
    parser.add_argument('--hi-frac', type=float, default=0.90)
    parser.add_argument('--lo-frac', type=float, default=0.10)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gt_path   = os.path.join(args.outdir, 'gt_modified.bed')
    cand_path = os.path.join(args.outdir, 'candidate.bed')

    mod_chars, unmod_chars = set(), set()
    for ctx_key in (_CONTEXTS.get(args.context, _CONTEXTS['CpG']),):
        pass
    ctx_chars = _CONTEXTS.get(args.context, _CONTEXTS['CpG'])
    mod_chars   = {c for c in ctx_chars if c.isupper()}
    unmod_chars = {c for c in ctx_chars if c.islower()}
    all_chars   = mod_chars | unmod_chars

    print(f"BAM           : {args.bam}", file=sys.stderr)
    print(f"Output dir    : {args.outdir}", file=sys.stderr)
    print(f"Context       : {args.context} "
          f"(mod={sorted(mod_chars)}, unmod={sorted(unmod_chars)})",
          file=sys.stderr)
    print(f"Filters       : cov>={args.min_cov}, "
          f"hi>={args.hi_frac}, lo<={args.lo_frac}", file=sys.stderr)

    # (chrom, pos) → [n_mod, n_unmod]
    counts: dict = collections.defaultdict(lambda: [0, 0])

    bam = pysam.AlignmentFile(args.bam, 'rb', threads=args.threads)

    n_reads = n_skip = 0
    for read in tqdm(bam.fetch(until_eof=True), desc="Parsing reads",
                     file=sys.stderr):
        if read.is_unmapped or read.is_supplementary or read.is_secondary:
            n_skip += 1
            continue

        xm = read.get_tag('XM') if read.has_tag('XM') else None
        if xm is None:
            n_skip += 1
            continue

        # XM string is aligned to the read sequence (same length after soft clips).
        # We need to map read positions to reference positions via the alignment.
        ref_name = read.reference_name
        pairs = read.get_aligned_pairs()  # (query_pos, ref_pos)

        for qpos, rpos in pairs:
            if qpos is None or rpos is None:
                continue
            if qpos >= len(xm):
                break
            c = xm[qpos]
            if c not in all_chars:
                continue
            key = (ref_name, rpos)
            if c in mod_chars:
                counts[key][0] += 1
            else:
                counts[key][1] += 1

        n_reads += 1

    bam.close()
    print(f"\nReads processed: {n_reads:,}  skipped: {n_skip:,}", file=sys.stderr)
    print(f"Positions with data: {len(counts):,}", file=sys.stderr)

    n_mod = n_unmod = n_ambig = n_lowcov = 0

    with open(gt_path, 'w') as fgt, open(cand_path, 'w') as fcand:
        for (chrom, pos), (nm, nu) in sorted(counts.items()):
            total = nm + nu
            if total < args.min_cov:
                n_lowcov += 1
                continue
            frac = nm / total
            if frac >= args.hi_frac:
                line = f"{chrom}\t{pos}\n"
                fgt.write(line)
                fcand.write(line)
                n_mod += 1
            elif frac <= args.lo_frac:
                fcand.write(f"{chrom}\t{pos}\n")
                n_unmod += 1
            else:
                n_ambig += 1

    print(f"\nModified (label=1) : {n_mod:,}", file=sys.stderr)
    print(f"Unmodified in cand : {n_unmod:,}", file=sys.stderr)
    print(f"Ambiguous (skipped): {n_ambig:,}", file=sys.stderr)
    print(f"Low coverage       : {n_lowcov:,}", file=sys.stderr)
    print(f"Candidate sites    : {n_mod + n_unmod:,}", file=sys.stderr)
    print(f"\ngt_modified.bed → {gt_path}", file=sys.stderr)
    print(f"candidate.bed   → {cand_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
