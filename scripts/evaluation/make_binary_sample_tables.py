#!/usr/bin/env python3
"""Build matched canonical-vs-M.SssI comparison tables for Rockfish and DeepMod."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


Key = Tuple[str, str, int]  # sample, ref_name, ref_pos


@dataclass
class Row:
    sample: str
    ref_name: str
    ref_pos: int
    ref_base: str
    label: int
    score: float
    source_score: str
    coverage: str = "NA"
    n_calls: str = "NA"
    n_no_call: str = "NA"
    n_images: str = "NA"
    score_note: str = ""


def parse_float_or_none(value: str) -> Optional[float]:
    if value in ("", "NA", "nan", "NaN", None):
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


def load_method_rows(
    can_path: Path,
    mod_path: Path,
    cpg_sites: Set[Tuple[str, int]],
    method: str,
) -> Dict[Key, Row]:
    rows: Dict[Key, Row] = {}
    for sample, label, path in (("can", 0, can_path), ("mod", 1, mod_path)):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for raw in reader:
                ref_name = raw["ref_name"]
                ref_pos = int(raw["ref_pos"])
                if (ref_name, ref_pos) not in cpg_sites:
                    continue

                coverage = raw.get("coverage", "NA")
                if method == "Rockfish" and coverage not in ("", "NA") and int(float(coverage)) <= 0:
                    continue

                source_score = raw.get("mean_prob", "NA")
                parsed = parse_float_or_none(source_score)
                if parsed is None:
                    score = 0.0
                    score_note = "no_call_set_to_0" if method == "Rockfish" else "missing_score_set_to_0"
                else:
                    score = parsed
                    score_note = "called" if method == "Rockfish" else "scored"

                key = (sample, ref_name, ref_pos)
                rows[key] = Row(
                    sample=sample,
                    ref_name=ref_name,
                    ref_pos=ref_pos,
                    ref_base=raw.get("ref_base", "N"),
                    label=label,
                    score=score,
                    source_score=source_score,
                    coverage=coverage,
                    n_calls=raw.get("n_calls", "NA"),
                    n_no_call=raw.get("n_no_call", "NA"),
                    n_images=raw.get("n_images", "NA"),
                    score_note=score_note,
                )
    return rows


def write_rows(path: Path, dataset: str, rows: Dict[Key, Row], keys: Iterable[Key]) -> int:
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
        "n_images",
        "score_note",
    ]
    n = 0
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for key in sorted(keys):
            row = rows[key]
            writer.writerow(
                {
                    "dataset": dataset,
                    "sample": row.sample,
                    "ref_name": row.ref_name,
                    "ref_pos": row.ref_pos,
                    "start": row.ref_pos,
                    "end": row.ref_pos + 1,
                    "ref_base": row.ref_base,
                    "gt_label": row.label,
                    "mean_prob": f"{row.score:.8g}",
                    "source_mean_prob": row.source_score,
                    "coverage": row.coverage,
                    "n_calls": row.n_calls,
                    "n_no_call": row.n_no_call,
                    "n_images": row.n_images,
                    "score_note": row.score_note,
                }
            )
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="remora_msssi_cpg_5mc")
    parser.add_argument("--cpg-bed", type=Path, required=True)
    parser.add_argument("--rockfish-can", type=Path, required=True)
    parser.add_argument("--rockfish-mod", type=Path, required=True)
    parser.add_argument("--deepmod-can", type=Path, required=True)
    parser.add_argument("--deepmod-mod", type=Path, required=True)
    parser.add_argument("--rockfish-out", type=Path, required=True)
    parser.add_argument("--deepmod-out", type=Path, required=True)
    parser.add_argument(
        "--allow-method-specific-sites",
        action="store_true",
        help="Keep each method's own callable CpG sites. Default writes the strict method intersection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpg_sites = load_bed(args.cpg_bed)
    rockfish = load_method_rows(args.rockfish_can, args.rockfish_mod, cpg_sites, "Rockfish")
    deepmod = load_method_rows(args.deepmod_can, args.deepmod_mod, cpg_sites, "DeepMod")

    if args.allow_method_specific_sites:
        rockfish_keys = set(rockfish)
        deepmod_keys = set(deepmod)
    else:
        rockfish_keys = deepmod_keys = set(rockfish) & set(deepmod)

    n_rf = write_rows(args.rockfish_out, args.dataset, rockfish, rockfish_keys)
    n_dm = write_rows(args.deepmod_out, args.dataset, deepmod, deepmod_keys)

    print(
        f"CpG sites in BED: {len(cpg_sites):,}; Rockfish rows: {len(rockfish):,}; "
        f"DeepMod rows: {len(deepmod):,}; wrote Rockfish={n_rf:,}, DeepMod={n_dm:,}."
    )


if __name__ == "__main__":
    main()
