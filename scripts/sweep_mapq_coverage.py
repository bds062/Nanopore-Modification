#!/usr/bin/env python3
"""Sweep MAPQ thresholds and report how many ground-truth sites keep enough reads
for a full pileup image. Helps pick the highest MAPQ that still yields at least
one complete image (min-reads) per modified site.

  python scripts/sweep_mapq_coverage.py \
    --bam READS_md.bam --peaks peaks_refined.tsv --gt sites.bed --min-reads 30
"""
import argparse
from collections import defaultdict
import numpy as np
import pysam


def load_peak_ids(path):
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.add(line.split('\t')[0])
    return ids


def load_gt(path):
    sites = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                p = line.split('\t')
                if len(p) >= 2:
                    sites.append((p[0], int(p[1])))
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bam', required=True)
    ap.add_argument('--peaks', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--min-reads', type=int, default=30)
    ap.add_argument('--thresholds', default='20,25,30,35,40,45,50,55,60')
    args = ap.parse_args()

    peak_ids = load_peak_ids(args.peaks)
    gt = load_gt(args.gt)
    gt_by_ref = defaultdict(list)
    for rn, pos in gt:
        gt_by_ref[rn].append(pos)

    reads = defaultdict(list)   # ref_name -> [(start, end, mapq), ...]
    bam = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)
    for r in bam:
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if r.query_name not in peak_ids:
            continue
        reads[r.reference_name].append(
            (r.reference_start, r.reference_end, r.mapping_quality))
    bam.close()

    thresholds = [int(x) for x in args.thresholds.split(',')]
    print(f"GT sites: {len(gt)}   full-image threshold: >= {args.min_reads} reads/site")
    print(f"{'MAPQ>=':>7} {'sites_full':>12} {'median_cov':>11}")
    for T in thresholds:
        full, covs = 0, []
        for rn, positions in gt_by_ref.items():
            rlist = [(s, e) for (s, e, m) in reads.get(rn, []) if m >= T]
            for pos in positions:
                c = sum(1 for (s, e) in rlist if s <= pos < e)
                covs.append(c)
                if c >= args.min_reads:
                    full += 1
        med = float(np.median(covs)) if covs else 0.0
        print(f"{T:>7} {full:>5}/{len(gt):<6} {med:>11.0f}")


if __name__ == '__main__':
    main()
