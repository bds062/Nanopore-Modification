#!/usr/bin/env python3
"""
Extract 5mC ground-truth BED files from a pre-computed methylation pileup
(ENCODE WGBS bed9+2, or Bismark .cov/.cov.gz coverage file).

Unlike extract_gt_bismark.py (which parses XM tags out of a raw Bismark BAM),
this script consumes an already-aggregated per-position pileup — used for
hg001 (ENCFF*.bed.gz from ENCODE) and hg002 (CpG.bismark.zero.cov.gz), where
no raw bisulfite BAM is available locally, only the finished pileup.

Applies the same Rockfish-style confidence filter as the other two GT
scripts:
  modified  : coverage >= MIN_COV and fraction >= HI_FRAC  -> label=1
  unmodified: coverage >= MIN_COV and fraction <= LO_FRAC  -> label=0
  discarded : everything else (ambiguous / low coverage)

Outputs (in OUTDIR), same contract as extract_gt_bismark.py / extract_gt_emseq.sh:
  gt_modified.bed  — tab: chrom, 0-based pos   (positives for --gt)
  candidate.bed    — tab: chrom, 0-based pos   (high-confidence +/- for --candidate-bed)

Formats
-------
encode_bed  (ENCODE WGBS bed9+2, e.g. ENCFF835NTC.bed.gz):
  col0=chrom col1=start(0-based) ... col9=coverage col10=pct_methylated(0-100)

bismark_cov (Bismark .cov / .cov.gz, e.g. CpG.bismark.zero.cov.gz):
  col0=chrom col1=start(0-based) col2=end col3=pct_methylated(0-100)
  col4=count_methylated col5=count_unmethylated

Usage:
  python extract_gt_from_pileup.py \\
      --input ENCFF835NTC.bed.gz --format encode_bed \\
      --outdir /path/to/gt/hg001 \\
      [--valid-contigs REF.fna.fai] \\
      [--min-cov 30] [--hi-frac 0.90] [--lo-frac 0.10]
"""

import argparse
import gzip
import os
import sys


def open_maybe_gzip(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path)


def load_valid_contigs(fai_path):
    if not fai_path:
        return None
    contigs = set()
    with open(fai_path) as fh:
        for line in fh:
            contigs.add(line.split('\t', 1)[0])
    return contigs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='Pileup file (.bed[.gz] or .cov[.gz])')
    p.add_argument('--format', required=True, choices=['encode_bed', 'bismark_cov'])
    p.add_argument('--outdir', required=True)
    p.add_argument('--valid-contigs', default=None,
                   help='Optional .fai of the reference used for nanopore alignment; '
                        'positions on any other contig are dropped.')
    p.add_argument('--min-cov', type=int, default=30)
    p.add_argument('--hi-frac', type=float, default=0.90)
    p.add_argument('--lo-frac', type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    gt_path = os.path.join(args.outdir, 'gt_modified.bed')
    cand_path = os.path.join(args.outdir, 'candidate.bed')

    valid_contigs = load_valid_contigs(args.valid_contigs)

    print(f"Input         : {args.input}", file=sys.stderr)
    print(f"Format        : {args.format}", file=sys.stderr)
    print(f"Valid contigs : {'all' if valid_contigs is None else len(valid_contigs)}",
          file=sys.stderr)
    print(f"Filters       : cov>={args.min_cov}, hi>={args.hi_frac}, lo<={args.lo_frac}",
          file=sys.stderr)

    n_seen = n_wrong_contig = n_lowcov = n_mod = n_unmod = 0

    with open_maybe_gzip(args.input) as fin, \
         open(gt_path, 'w') as fgt, open(cand_path, 'w') as fcand:
        for line in fin:
            cols = line.rstrip('\n').split('\t')
            chrom = cols[0]
            pos = int(cols[1])
            n_seen += 1

            if valid_contigs is not None and chrom not in valid_contigs:
                n_wrong_contig += 1
                continue

            if args.format == 'encode_bed':
                cov = int(cols[9])
                pct = float(cols[10])
            else:  # bismark_cov
                pct = float(cols[3])
                cov = int(cols[4]) + int(cols[5])

            if cov < args.min_cov:
                n_lowcov += 1
                continue

            frac = pct / 100.0
            line_out = f"{chrom}\t{pos}\n"
            if frac >= args.hi_frac:
                fgt.write(line_out)
                fcand.write(line_out)
                n_mod += 1
            elif frac <= args.lo_frac:
                fcand.write(line_out)
                n_unmod += 1

    print(f"\nRows read: {n_seen:,}  wrong-contig: {n_wrong_contig:,}  "
          f"low-cov: {n_lowcov:,}", file=sys.stderr)
    print(f"Modified (>= hi-frac): {n_mod:,}", file=sys.stderr)
    print(f"Unmodified (<= lo-frac): {n_unmod:,}", file=sys.stderr)
    print(f"Wrote {gt_path} and {cand_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
