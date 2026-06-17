#!/usr/bin/env python3
"""Build a Rockfish-only callable-CpG canonical-vs-M.SssI table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


Key = Tuple[str, int]
SampleKey = Tuple[str, str, int]


def parse_float_or_none(value: Optional[str]) -> Optional[float]:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    return float(value)


def load_bed(path: Path) -> Set[Key]:
    sites: Set[Key] = set()
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            sites.add((parts[0], int(parts[1])))
    return sites


def load_callable_rows(path: Path, sample: str, label: int, cpg_sites: Set[Key], score_col: str) -> Dict[Key, Dict[str, str]]:
    rows: Dict[Key, Dict[str, str]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (row["ref_name"], int(row["ref_pos"]))
            if key not in cpg_sites:
                continue
            score = parse_float_or_none(row.get(score_col))
            if score is None:
                continue
            out = dict(row)
            out["sample"] = sample
            out["gt_label"] = str(label)
            out["mean_prob"] = f"{score:.8g}"
            out["source_mean_prob"] = row.get(score_col, "NA")
            rows[key] = out
    return rows


def write_table(path: Path, dataset: str, rows: Dict[SampleKey, Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "coverage",
        "n_calls",
        "n_no_call",
        "frac_mod",
        "n_mod_calls",
        "n_unmod_calls",
        "min_prob",
        "max_prob",
        "fwd_coverage",
        "rev_coverage",
        "n_fwd_calls",
        "n_rev_calls",
        "score_note",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample, ref_name, ref_pos in sorted(rows):
            row = rows[(sample, ref_name, ref_pos)]
            writer.writerow(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "ref_name": ref_name,
                    "ref_pos": ref_pos,
                    "start": ref_pos,
                    "end": ref_pos + 1,
                    "ref_base": row.get("ref_base", "N"),
                    "gt_label": row["gt_label"],
                    "mean_prob": row["mean_prob"],
                    "source_mean_prob": row["source_mean_prob"],
                    "coverage": row.get("coverage", "NA"),
                    "n_calls": row.get("n_calls", "NA"),
                    "n_no_call": row.get("n_no_call", "NA"),
                    "frac_mod": row.get("frac_mod", "NA"),
                    "n_mod_calls": row.get("n_mod_calls", "NA"),
                    "n_unmod_calls": row.get("n_unmod_calls", "NA"),
                    "min_prob": row.get("min_prob", "NA"),
                    "max_prob": row.get("max_prob", "NA"),
                    "fwd_coverage": row.get("fwd_coverage", "NA"),
                    "rev_coverage": row.get("rev_coverage", "NA"),
                    "n_fwd_calls": row.get("n_fwd_calls", "NA"),
                    "n_rev_calls": row.get("n_rev_calls", "NA"),
                    "score_note": "rockfish_called_cpg",
                }
            )


def write_summary(
    path: Path,
    cpg_sites: Set[Key],
    can_rows: Dict[Key, Dict[str, str]],
    mod_rows: Dict[Key, Dict[str, str]],
    kept_keys: Iterable[Key],
    callable_mode: str,
) -> None:
    kept = set(kept_keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerow(["callable_mode", callable_mode])
        writer.writerow(["cpg_sites_in_bed", len(cpg_sites)])
        writer.writerow(["canonical_callable_cpg_sites", len(can_rows)])
        writer.writerow(["modified_callable_cpg_sites", len(mod_rows)])
        writer.writerow(["paired_callable_cpg_sites", len(set(can_rows) & set(mod_rows))])
        writer.writerow(["kept_cpg_sites", len(kept)])
        writer.writerow(["output_rows", len(kept) * 2 if callable_mode == "paired" else len(can_rows) + len(mod_rows)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="remora_msssi_5mc_rockfish_callable_cpg")
    parser.add_argument("--cpg-bed", type=Path, required=True)
    parser.add_argument("--can", type=Path, required=True)
    parser.add_argument("--mod", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--score-col", default="mean_prob")
    parser.add_argument(
        "--callable-mode",
        choices=("paired", "sample"),
        default="paired",
        help="paired keeps CpG positions with Rockfish scores in both samples; sample keeps each sample's own callable CpGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpg_sites = load_bed(args.cpg_bed)
    can_rows = load_callable_rows(args.can, "can", 0, cpg_sites, args.score_col)
    mod_rows = load_callable_rows(args.mod, "mod", 1, cpg_sites, args.score_col)

    out_rows: Dict[SampleKey, Dict[str, str]] = {}
    if args.callable_mode == "paired":
        kept_keys = set(can_rows) & set(mod_rows)
        for ref_name, ref_pos in kept_keys:
            out_rows[("can", ref_name, ref_pos)] = can_rows[(ref_name, ref_pos)]
            out_rows[("mod", ref_name, ref_pos)] = mod_rows[(ref_name, ref_pos)]
    else:
        kept_keys = set(can_rows) | set(mod_rows)
        for (ref_name, ref_pos), row in can_rows.items():
            out_rows[("can", ref_name, ref_pos)] = row
        for (ref_name, ref_pos), row in mod_rows.items():
            out_rows[("mod", ref_name, ref_pos)] = row

    write_table(args.output, args.dataset, out_rows)
    if args.summary is not None:
        write_summary(args.summary, cpg_sites, can_rows, mod_rows, kept_keys, args.callable_mode)

    print(
        f"CpG sites in BED: {len(cpg_sites):,}; canonical callable: {len(can_rows):,}; "
        f"modified callable: {len(mod_rows):,}; mode={args.callable_mode}; wrote rows: {len(out_rows):,}."
    )


if __name__ == "__main__":
    main()
