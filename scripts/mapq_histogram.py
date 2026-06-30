#!/usr/bin/env python3
"""MAPQ histogram across one or more BAMs.

Used to diagnose why the RNA featurization kept few reads: synthetic reads should
map at high MAPQ, so a low-MAPQ distribution points at the alignment step.

  python scripts/mapq_histogram.py \
    --bam "AWS:/path/basecalls/m6A_rep1.bam" \
    --bam "ours:/path/basecalls_moves/m6A_rep1_aligned.bam" \
    --out outputs/mapq_m6A_rep1.png
"""
import argparse
import numpy as np
import pysam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_mapq(path):
    mapq = []
    bam = pysam.AlignmentFile(path, "rb", check_sq=False)
    for r in bam:
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        mapq.append(r.mapping_quality)
    bam.close()
    return np.array(mapq, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bam", action="append", required=True,
                    help="label:path  (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plt.figure(figsize=(9, 5))
    bins = np.arange(0, 62, 2)
    print(f"{'label':<16} {'n_primary':>10} {'median':>7} {'%>=20':>7} {'%==60':>7}")
    for spec in args.bam:
        label, path = spec.split(":", 1)
        mq = collect_mapq(path)
        if len(mq) == 0:
            print(f"{label:<16} {'0':>10}  (no primary alignments)")
            continue
        med = float(np.median(mq))
        pct20 = 100.0 * np.mean(mq >= 20)
        pct60 = 100.0 * np.mean(mq == 60)
        print(f"{label:<16} {len(mq):>10,} {med:>7.0f} {pct20:>6.1f}% {pct60:>6.1f}%")
        plt.hist(mq, bins=bins, alpha=0.55, label=f"{label} (n={len(mq):,}, med={med:.0f})")

    plt.xlabel("MAPQ")
    plt.ylabel("primary alignments")
    plt.title("MAPQ distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
