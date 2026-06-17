#!/usr/bin/env python3
"""Write a BED file for forward-reference CpG cytosines."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Optional


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def load_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    name: Optional[str] = None
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                seqs.setdefault(name, [])
            elif name is not None:
                seqs[name].append(line.upper())
    return {name: "".join(parts) for name, parts in seqs.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="5mC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fasta = load_fasta(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_sites = 0
    with open(args.output, "w") as out:
        for ref_name, seq in fasta.items():
            for pos in range(0, max(len(seq) - 1, 0)):
                if seq[pos : pos + 2] == "CG":
                    print(ref_name, pos, pos + 1, args.name, 0, "+", sep="\t", file=out)
                    n_sites += 1

    print(f"Wrote {n_sites:,} forward-reference CpG cytosines to {args.output}")


if __name__ == "__main__":
    main()
