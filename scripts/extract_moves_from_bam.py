#!/usr/bin/env python3
"""Extract Dorado move-table tags from a BAM into DeepMod's moves TSV format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pysam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--add-sp-to-ts",
        action="store_true",
        help="Add the optional sp tag to ts when present. Leave off for standard Dorado mv/ts tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0
    n_missing_mv = 0
    with pysam.AlignmentFile(str(args.bam), "rb", check_sq=False) as bam, open(args.output, "w") as out:
        for read in bam:
            n_total += 1
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if not read.has_tag("mv"):
                n_missing_mv += 1
                continue

            mv = [int(x) for x in read.get_tag("mv")]
            if not mv:
                n_missing_mv += 1
                continue

            ts = int(read.get_tag("ts")) if read.has_tag("ts") else 0
            if args.add_sp_to_ts and read.has_tag("sp"):
                ts += int(read.get_tag("sp"))

            print(
                read.query_name,
                "mv:B:c," + ",".join(str(x) for x in mv),
                f"ts:i:{ts}",
                sep="\t",
                file=out,
            )
            n_written += 1

    print(
        f"Scanned {n_total:,} BAM records; wrote {n_written:,} move rows; "
        f"missing mv tag on {n_missing_mv:,} records.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
