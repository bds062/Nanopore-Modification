#!/usr/bin/env python3
"""Aggregate Dorado MM/ML modified-base probabilities by reference position."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pysam


MOD_DEFAULTS = {
    "5mC": ("C", {"m"}),
    "5hmC": ("C", {"h"}),
    "6mA": ("A", {"a"}),
    # Dorado/SAM can encode 4mC with its ChEBI identifier.
    "4mC": ("C", {"21839"}),
}


@dataclass
class PosAgg:
    n_calls: int = 0
    n_mod_calls: int = 0
    sum_prob: float = 0.0
    min_prob: float = math.inf
    max_prob: float = -math.inf

    def add(self, prob: float, threshold: float) -> None:
        self.n_calls += 1
        self.n_mod_calls += int(prob >= threshold)
        self.sum_prob += prob
        self.min_prob = min(self.min_prob, prob)
        self.max_prob = max(self.max_prob, prob)

    @property
    def mean_prob(self) -> float:
        return self.sum_prob / max(self.n_calls, 1)

    @property
    def frac_mod(self) -> float:
        return self.n_mod_calls / max(self.n_calls, 1)


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def load_fasta(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
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
    return {key: "".join(parts) for key, parts in seqs.items()}


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.8g}"


def normalize_mod_code(value) -> str:
    return str(value).lower()


def mod_spec(args: argparse.Namespace) -> Tuple[str, set[str]]:
    if args.canonical_base or args.mod_code:
        if not args.canonical_base or not args.mod_code:
            raise SystemExit("--canonical-base and --mod-code must be used together")
        return args.canonical_base.upper(), {normalize_mod_code(args.mod_code)}
    canonical, codes = MOD_DEFAULTS[args.mod_type]
    return canonical, {normalize_mod_code(code) for code in codes}


def read_passes_filters(read: pysam.AlignedSegment, min_mapq: int, include_secondary: bool, include_supplementary: bool) -> bool:
    if read.is_unmapped:
        return False
    if read.mapping_quality < min_mapq:
        return False
    if read.is_secondary and not include_secondary:
        return False
    if read.is_supplementary and not include_supplementary:
        return False
    return True


def iter_matching_calls(
    read: pysam.AlignedSegment,
    canonical_base: str,
    mod_codes: set[str],
    observed_keys: Counter,
) -> Iterable[Tuple[int, float]]:
    try:
        modified = read.modified_bases or {}
    except Exception as exc:  # Malformed tags should not kill the whole run.
        print(f"WARNING: could not parse MM/ML tags for {read.query_name}: {exc}", file=sys.stderr)
        return

    for key, calls in modified.items():
        canonical, strand, mod_code = key
        key_name = f"{canonical}:{strand}:{mod_code}"
        observed_keys[key_name] += len(calls)
        if str(canonical).upper() != canonical_base:
            continue
        if normalize_mod_code(mod_code) not in mod_codes:
            continue
        for query_pos, qual in calls:
            if qual is None or qual < 0:
                continue
            prob = max(0.0, min(1.0, float(qual) / 255.0))
            yield int(query_pos), prob


def aggregate(args: argparse.Namespace) -> Tuple[Dict[Tuple[str, int], PosAgg], Counter]:
    canonical_base, mod_codes = mod_spec(args)
    agg: Dict[Tuple[str, int], PosAgg] = defaultdict(PosAgg)
    observed_keys: Counter = Counter()
    n_reads = 0
    n_reads_with_tags = 0

    with pysam.AlignmentFile(str(args.bam), "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if not read_passes_filters(read, args.min_mapq, args.include_secondary, args.include_supplementary):
                continue
            n_reads += 1
            try:
                modified = read.modified_bases
            except Exception:
                modified = None
            if modified:
                n_reads_with_tags += 1

            ref_positions = read.get_reference_positions(full_length=True)
            ref_name = bam.get_reference_name(read.reference_id)
            for query_pos, prob in iter_matching_calls(read, canonical_base, mod_codes, observed_keys):
                if query_pos < 0 or query_pos >= len(ref_positions):
                    continue
                ref_pos = ref_positions[query_pos]
                if ref_pos is None:
                    continue
                agg[(ref_name, int(ref_pos))].add(prob, args.threshold)

    print(
        f"[{args.dataset}] reads passing filters={n_reads:,}; reads with MM/ML tags={n_reads_with_tags:,}; "
        f"{args.mod_type} reference positions={len(agg):,}",
        file=sys.stderr,
    )
    if not agg:
        print(f"[{args.dataset}] observed modified-base keys: {dict(observed_keys.most_common(20))}", file=sys.stderr)
    return agg, observed_keys


def write_output(args: argparse.Namespace, agg: Dict[Tuple[str, int], PosAgg], observed_keys: Counter) -> None:
    fasta = load_fasta(args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_name = args.mod_model_name or args.bam.stem

    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "mod_model",
                "mod_type",
                "ref_name",
                "ref_pos",
                "start",
                "end",
                "ref_base",
                "gt_label",
                "n_calls",
                "mean_prob",
                "frac_mod",
                "n_mod_calls",
                "n_unmod_calls",
                "min_prob",
                "max_prob",
            ]
        )
        for ref_name, pos in sorted(agg, key=lambda item: (item[0], item[1])):
            item = agg[(ref_name, pos)]
            if item.n_calls < args.min_calls:
                continue
            seq = fasta.get(ref_name, "")
            ref_base = seq[pos] if 0 <= pos < len(seq) else "N"
            label_score = item.mean_prob if args.label_mode == "mean_prob" else item.frac_mod
            writer.writerow(
                [
                    args.dataset,
                    model_name,
                    args.mod_type,
                    ref_name,
                    pos,
                    pos,
                    pos + 1,
                    ref_base,
                    int(label_score >= args.threshold),
                    item.n_calls,
                    fmt(item.mean_prob),
                    fmt(item.frac_mod),
                    item.n_mod_calls,
                    item.n_calls - item.n_mod_calls,
                    fmt(item.min_prob),
                    fmt(item.max_prob),
                ]
            )

    if args.key_summary:
        args.key_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(args.key_summary, "w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["modified_base_key", "calls"])
            for key, count in observed_keys.most_common():
                writer.writerow([key, count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam", type=Path, required=True, help="Aligned Dorado BAM with MM/ML tags.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mod-type", choices=sorted(MOD_DEFAULTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--mod-model-name", default="")
    parser.add_argument("--canonical-base", default="")
    parser.add_argument("--mod-code", default="")
    parser.add_argument("--threshold", type=float, default=0.5, help="Dorado probability threshold.")
    parser.add_argument("--label-mode", choices=["mean_prob", "frac_mod"], default="mean_prob")
    parser.add_argument("--min-mapq", type=int, default=0)
    parser.add_argument("--min-calls", type=int, default=1)
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--include-supplementary", action="store_true")
    parser.add_argument("--key-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agg, observed_keys = aggregate(args)
    write_output(args, agg, observed_keys)
    print(f"[{args.dataset}] wrote Dorado reference labels: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
