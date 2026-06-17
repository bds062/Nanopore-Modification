#!/usr/bin/env python3
"""Write CpG cytosine BED rows using reference bases recovered from BAM MD tags."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import pysam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="m")
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument(
        "--include-minus-strand",
        action="store_true",
        help="Also write the reverse-strand CpG cytosine, which is the forward-reference G in CG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_counts: Dict[Tuple[str, int], Counter] = defaultdict(Counter)

    with pysam.AlignmentFile(str(args.bam), "rb", check_sq=False) as bam:
        for read in bam:
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if not read.has_tag("MD"):
                continue
            for _qpos, rpos, ref_base in read.get_aligned_pairs(with_seq=True):
                if rpos is None or ref_base is None:
                    continue
                base = ref_base.upper()
                if base in {"A", "C", "G", "T"}:
                    base_counts[(read.reference_name, int(rpos))][base] += 1

    consensus: Dict[Tuple[str, int], str] = {}
    for key, counts in base_counts.items():
        base, support = counts.most_common(1)[0]
        if support >= args.min_support:
            consensus[key] = base

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_plus = 0
    n_minus = 0
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        contigs = sorted({ref_name for ref_name, _pos in consensus})
        for ref_name in contigs:
            positions = sorted(pos for rn, pos in consensus if rn == ref_name)
            pos_set = set(positions)
            for pos in positions:
                if consensus.get((ref_name, pos)) == "C" and (pos + 1) in pos_set:
                    if consensus.get((ref_name, pos + 1)) == "G":
                        writer.writerow([ref_name, pos, pos + 1, args.name, ".", "+"])
                        n_plus += 1
                        if args.include_minus_strand:
                            writer.writerow([ref_name, pos + 1, pos + 2, args.name, ".", "-"])
                            n_minus += 1

    total = n_plus + n_minus
    print(
        f"Wrote {total:,} CpG cytosine rows to {args.output} "
        f"({n_plus:,} plus-strand, {n_minus:,} minus-strand)."
    )


if __name__ == "__main__":
    main()
