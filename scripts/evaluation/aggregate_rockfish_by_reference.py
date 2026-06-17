#!/usr/bin/env python3
"""Aggregate Rockfish read-level predictions onto reference coordinates.

Rockfish writes read-local calls (`read_id`, `pos`, `prob`).  This script uses
the BAM alignment to pile those calls up by reference base, then adds binary
ground-truth labels from the same BED convention used by
results/deep_modification/deep_mod_featurization.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pysam


MATCH_OPS = {0, 7, 8}  # M, =, X
QUERY_ONLY_OPS = {1, 4}  # I, S
REF_ONLY_OPS = {2, 3}  # D, N
SKIP_OPS = {5, 6}  # H, P


@dataclass
class PredAgg:
    sum_prob: float = 0.0
    n: int = 0
    n_mod: int = 0
    min_prob: float = math.inf
    max_prob: float = -math.inf

    def add(self, prob: float, threshold: float) -> None:
        self.sum_prob += prob
        self.n += 1
        self.n_mod += int(prob > threshold)
        self.min_prob = min(self.min_prob, prob)
        self.max_prob = max(self.max_prob, prob)


@dataclass
class SiteAgg:
    coverage: int = 0
    fwd_coverage: int = 0
    rev_coverage: int = 0
    n_calls: int = 0
    n_mod_calls: int = 0
    sum_prob: float = 0.0
    min_prob: float = math.inf
    max_prob: float = -math.inf
    n_fwd_calls: int = 0
    n_rev_calls: int = 0

    def add_coverage(self, is_reverse: bool) -> None:
        self.coverage += 1
        if is_reverse:
            self.rev_coverage += 1
        else:
            self.fwd_coverage += 1

    def add_prediction(self, pred: PredAgg, is_reverse: bool) -> None:
        self.n_calls += pred.n
        self.n_mod_calls += pred.n_mod
        self.sum_prob += pred.sum_prob
        self.min_prob = min(self.min_prob, pred.min_prob)
        self.max_prob = max(self.max_prob, pred.max_prob)
        if is_reverse:
            self.n_rev_calls += pred.n
        else:
            self.n_fwd_calls += pred.n

    @property
    def n_no_call(self) -> int:
        return max(self.coverage - self.n_calls, 0)

    @property
    def mean_prob(self) -> Optional[float]:
        if self.n_calls == 0:
            return None
        return self.sum_prob / self.n_calls

    @property
    def frac_mod(self) -> Optional[float]:
        if self.n_calls == 0:
            return None
        return self.n_mod_calls / self.n_calls


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "r", newline="")


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.8g}"


def load_predictions(
    path: Path,
    threshold: float,
    input_logits: bool = False,
) -> Tuple[Dict[str, Dict[int, PredAgg]], int]:
    by_read: Dict[str, Dict[int, PredAgg]] = defaultdict(dict)
    n_rows = 0

    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        fields = set(reader.fieldnames)
        if "read_id" not in fields or "pos" not in fields:
            raise ValueError(f"{path} must contain read_id and pos columns")

        value_col = "logit" if input_logits else "prob"
        if value_col not in fields:
            fallback = "prob" if "prob" in fields else "logit" if "logit" in fields else None
            if fallback is None:
                raise ValueError(f"{path} must contain prob or logit column")
            value_col = fallback

        for row in reader:
            read_id = row["read_id"]
            pos = int(row["pos"])
            value = float(row[value_col])
            prob = sigmoid(value) if input_logits or value_col == "logit" else value

            pred = by_read[read_id].setdefault(pos, PredAgg())
            pred.add(prob, threshold)
            n_rows += 1

    return dict(by_read), n_rows


def load_gt_bed(path: Optional[Path]) -> Set[Tuple[str, int]]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"GT BED does not exist: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"GT BED path is a directory, not a file: {path}")

    labels: Set[Tuple[str, int]] = set()
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            labels.add((parts[0], int(parts[1])))
    return labels


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
    return {name: "".join(parts) for name, parts in seqs.items()}


def iter_forward_query_to_ref(read: pysam.AlignedSegment) -> Iterator[Tuple[int, int]]:
    """Yield (forward_read_pos, ref_pos) for aligned query/reference bases.

    Rockfish uses `bam_read.get_forward_sequence()`, so prediction `pos` is in
    the original read orientation.  This mirrors Rockfish's extract_ref_pos.py
    CIGAR traversal for reverse alignments.
    """

    if read.cigartuples is None or read.reference_start is None or read.reference_end is None:
        return

    if read.is_reverse:
        qpos = 0
        rpos = read.reference_end - 1
        ref_step = -1
        cigars: Iterable[Tuple[int, int]] = reversed(read.cigartuples)
    else:
        qpos = 0
        rpos = read.reference_start
        ref_step = 1
        cigars = read.cigartuples

    for op, length in cigars:
        if op in MATCH_OPS:
            for _ in range(length):
                yield qpos, rpos
                qpos += 1
                rpos += ref_step
        elif op in QUERY_ONLY_OPS:
            qpos += length
        elif op in REF_ONLY_OPS:
            rpos += ref_step * length
        elif op in SKIP_OPS:
            continue
        else:
            raise ValueError(f"Unsupported CIGAR op {op} in {read.query_name}")


def collect_reference_order(
    bam_path: Path,
) -> List[Tuple[str, int]]:
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        return list(zip(bam.references, bam.lengths))


def aggregate(
    predictions: Dict[str, Dict[int, PredAgg]],
    bam_path: Path,
    min_mapq: int,
    include_qcfail: bool,
    include_secondary: bool,
    include_supplementary: bool,
    threads: int,
) -> Tuple[Dict[Tuple[str, int], SiteAgg], int, int]:
    sites: Dict[Tuple[str, int], SiteAgg] = defaultdict(SiteAgg)
    n_alignments = 0
    n_mapped_prediction_rows = 0

    with pysam.AlignmentFile(
        str(bam_path), "rb", check_sq=False, threads=max(1, threads)
    ) as bam:
        for read in bam:
            if read.is_unmapped:
                continue
            if read.is_secondary and not include_secondary:
                continue
            if read.is_supplementary and not include_supplementary:
                continue
            if read.is_qcfail and not include_qcfail:
                continue
            if read.mapping_quality < min_mapq:
                continue

            n_alignments += 1
            read_preds = predictions.get(read.query_name, {})
            ref_name = read.reference_name
            is_reverse = read.is_reverse

            for qpos, rpos in iter_forward_query_to_ref(read):
                key = (ref_name, rpos)
                site = sites[key]
                site.add_coverage(is_reverse)

                pred = read_preds.get(qpos)
                if pred is not None:
                    site.add_prediction(pred, is_reverse)
                    n_mapped_prediction_rows += pred.n

    return dict(sites), n_alignments, n_mapped_prediction_rows


def write_sites(
    output: Path,
    dataset: str,
    ref_order: Sequence[Tuple[str, int]],
    fasta: Dict[str, str],
    gt: Set[Tuple[str, int]],
    sites: Dict[Tuple[str, int], SiteAgg],
    covered_only: bool,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    header = [
        "dataset",
        "ref_name",
        "ref_pos",
        "start",
        "end",
        "ref_base",
        "gt_label",
        "coverage",
        "n_calls",
        "n_no_call",
        "mean_prob",
        "frac_mod",
        "n_mod_calls",
        "n_unmod_calls",
        "min_prob",
        "max_prob",
        "fwd_coverage",
        "rev_coverage",
        "n_fwd_calls",
        "n_rev_calls",
    ]

    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)

        if covered_only:
            keys_by_ref: Dict[str, Set[int]] = defaultdict(set)
            ref_lengths = dict(ref_order)
            for ref_name, pos in gt:
                if 0 <= pos < ref_lengths.get(ref_name, pos + 1):
                    keys_by_ref[ref_name].add(pos)
            for (ref_name, pos), site in sites.items():
                if site.coverage > 0 and 0 <= pos < ref_lengths.get(ref_name, pos + 1):
                    keys_by_ref[ref_name].add(pos)
            pos_iter_by_ref = (
                (ref_name, sorted(keys_by_ref.get(ref_name, set())))
                for ref_name, _ in ref_order
            )
        else:
            pos_iter_by_ref = ((ref_name, range(ref_len)) for ref_name, ref_len in ref_order)

        for ref_name, positions in pos_iter_by_ref:
            seq = fasta.get(ref_name, "")
            for pos in positions:
                key = (ref_name, pos)
                site = sites.get(key, SiteAgg())
                gt_label = int(key in gt)
                if covered_only and site.coverage == 0 and gt_label == 0:
                    continue

                mean_prob = site.mean_prob
                frac_mod = site.frac_mod
                min_prob = None if site.n_calls == 0 else site.min_prob
                max_prob = None if site.n_calls == 0 else site.max_prob
                ref_base = seq[pos] if pos < len(seq) else "N"

                writer.writerow(
                    [
                        dataset,
                        ref_name,
                        pos,
                        pos,
                        pos + 1,
                        ref_base,
                        gt_label,
                        site.coverage,
                        site.n_calls,
                        site.n_no_call,
                        fmt_float(mean_prob),
                        fmt_float(frac_mod),
                        site.n_mod_calls,
                        site.n_calls - site.n_mod_calls,
                        fmt_float(min_prob),
                        fmt_float(max_prob),
                        site.fwd_coverage,
                        site.rev_coverage,
                        site.n_fwd_calls,
                        site.n_rev_calls,
                    ]
                )
                n_written += 1

    return n_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map Rockfish read-level predictions onto reference bases and add "
            "Deep Modification-style BED labels."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--gt-bed", type=Path, default=None)
    parser.add_argument("--dataset", default="rockfish")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mapq-min", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--input-logits", action="store_true")
    parser.add_argument("--covered-only", action="store_true")
    parser.add_argument("--include-qcfail", action="store_true")
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--include-supplementary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[{args.dataset}] loading predictions: {args.predictions}", file=sys.stderr)
    predictions, n_pred_rows = load_predictions(
        args.predictions, threshold=args.threshold, input_logits=args.input_logits
    )
    print(
        f"[{args.dataset}] prediction rows={n_pred_rows:,} reads={len(predictions):,}",
        file=sys.stderr,
    )

    print(f"[{args.dataset}] loading labels: {args.gt_bed or 'all zero'}", file=sys.stderr)
    gt = load_gt_bed(args.gt_bed)
    print(f"[{args.dataset}] gt positive reference bases={len(gt):,}", file=sys.stderr)

    fasta = load_fasta(args.reference)
    ref_order = collect_reference_order(args.bam)

    print(f"[{args.dataset}] aggregating through BAM: {args.bam}", file=sys.stderr)
    sites, n_alignments, n_mapped_pred_rows = aggregate(
        predictions,
        args.bam,
        min_mapq=args.mapq_min,
        include_qcfail=args.include_qcfail,
        include_secondary=args.include_secondary,
        include_supplementary=args.include_supplementary,
        threads=args.threads,
    )

    print(f"[{args.dataset}] writing: {args.output}", file=sys.stderr)
    n_written = write_sites(
        args.output,
        dataset=args.dataset,
        ref_order=ref_order,
        fasta=fasta,
        gt=gt,
        sites=sites,
        covered_only=args.covered_only,
    )

    n_unmapped_pred_rows = n_pred_rows - n_mapped_pred_rows
    print(
        f"[{args.dataset}] alignments={n_alignments:,} rows={n_written:,} "
        f"mapped_prediction_rows={n_mapped_pred_rows:,} "
        f"unmapped_prediction_rows={n_unmapped_pred_rows:,}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
