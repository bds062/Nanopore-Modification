#!/usr/bin/env python3
"""Build a DeepMod-only canonical-vs-M.SssI benchmark table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Set, Tuple


def parse_float_or_none(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    return float(value)


def load_bed(path: Path) -> Set[Tuple[str, int]]:
    sites: Set[Tuple[str, int]] = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            sites.add((parts[0], int(parts[1])))
    return sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="remora_msssi_5mc_deepmod_all_positions")
    parser.add_argument("--cpg-bed", type=Path, default=None)
    parser.add_argument("--can", type=Path, required=True)
    parser.add_argument("--mod", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-col", default="mean_prob")
    parser.add_argument(
        "--no-cpg-filter",
        action="store_true",
        help="Keep every scored DeepMod row instead of filtering to cpg-bed coordinates.",
    )
    parser.add_argument(
        "--use-source-labels",
        action="store_true",
        help="Use gt_label from each prediction table. Default labels all can rows 0 and all mod rows 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpg_bed is None and not args.no_cpg_filter:
        raise SystemExit("--cpg-bed is required unless --no-cpg-filter is set")
    cpg_sites = load_bed(args.cpg_bed) if args.cpg_bed is not None else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "sample",
        "ref_name",
        "ref_pos",
        "start",
        "end",
        "ref_base",
        "gt_label",
        "mean_prob",
        "source_mean_prob",
        "n_images",
        "frac_mod",
        "n_mod_images",
        "n_unmod_images",
        "min_prob",
        "max_prob",
    ]

    n_in = 0
    n_written = 0
    n_missing_score = 0
    with open(args.output, "w", newline="") as out:
        writer = csv.DictWriter(out, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample, label, path in (("can", 0, args.can), ("mod", 1, args.mod)):
            with open(path, newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    n_in += 1
                    ref_name = row["ref_name"]
                    ref_pos = int(row["ref_pos"])
                    if not args.no_cpg_filter and (ref_name, ref_pos) not in cpg_sites:
                        continue
                    source_score = row.get(args.score_col, "NA")
                    score = parse_float_or_none(source_score)
                    if score is None:
                        n_missing_score += 1
                        continue
                    out_label = int(float(row.get("gt_label", label))) if args.use_source_labels else label
                    writer.writerow(
                        {
                            "dataset": args.dataset,
                            "sample": sample,
                            "ref_name": ref_name,
                            "ref_pos": ref_pos,
                            "start": ref_pos,
                            "end": ref_pos + 1,
                            "ref_base": row.get("ref_base", "N"),
                            "gt_label": out_label,
                            "mean_prob": f"{score:.8g}",
                            "source_mean_prob": source_score,
                            "n_images": row.get("n_images", "NA"),
                            "frac_mod": row.get("frac_mod", "NA"),
                            "n_mod_images": row.get("n_mod_images", "NA"),
                            "n_unmod_images": row.get("n_unmod_images", "NA"),
                            "min_prob": row.get("min_prob", "NA"),
                            "max_prob": row.get("max_prob", "NA"),
                        }
                    )
                    n_written += 1

    print(
        f"CpG sites in BED: {len(cpg_sites):,}; cpg_filter={not args.no_cpg_filter}; "
        f"use_source_labels={args.use_source_labels}; input prediction rows: {n_in:,}; "
        f"wrote DeepMod rows: {n_written:,}; skipped missing scores: {n_missing_score:,}."
    )


if __name__ == "__main__":
    main()
