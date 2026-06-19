#!/usr/bin/env python3
"""Summarize reference-level DeepMod predictions from an unlabeled dataset."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List


SVG_STYLE = '<style>text { font-family: DejaVu Sans, Arial, Helvetica, sans-serif; }</style>'


def parse_float(value: str) -> float | None:
    if value in ("", "NA", "nan", "NaN", None):
        return None
    return float(value)


def quantile(values: List[float], frac: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = round(frac * (len(values) - 1))
    return values[idx]


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.8g}"


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, body: str, size: int = 12, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="#111111">'
        f"{svg_escape(body)}</text>"
    )


def maybe_write_png(svg_path: Path) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return
    subprocess.run(
        [renderer, "-o", str(svg_path.with_suffix(".png")), str(svg_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def write_histogram(scores: List[float], output: Path, bins: int = 40) -> None:
    width, height = 900, 470
    x, y, w, h = 80, 85, 760, 285
    counts = [0] * bins
    for score in scores:
        idx = min(bins - 1, max(0, int(score * bins)))
        counts[idx] += 1
    max_count = max(counts) if counts else 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, "DeepMod Score Distribution", 18, "700", "middle"),
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="black" stroke-width="1"/>',
        text(x + w / 2, y + h + 48, "Mean modified probability", 12, "400", "middle"),
        f'<text x="{x - 48}" y="{y + h / 2}" font-size="12" transform="rotate(-90 {x - 48},{y + h / 2})" text-anchor="middle">Reference bases</text>',
    ]
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        px = x + t * w
        parts.append(f'<line x1="{px:.1f}" y1="{y+h:.1f}" x2="{px:.1f}" y2="{y+h+5:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(text(px, y + h + 20, f"{t:g}", 10, "400", "middle"))
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        py = y + h - frac * h
        label = int(round(frac * max_count))
        parts.append(f'<line x1="{x-5:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{py:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(text(x - 8, py + 4, f"{label:,}", 10, "400", "end"))

    bar_w = w / bins
    for i, count in enumerate(counts):
        bh = 0 if max_count == 0 else count / max_count * h
        bx = x + i * bar_w
        by = y + h - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w * 0.92:.1f}" height="{bh:.1f}" fill="steelblue"/>')

    parts.append(text(x, y + h + 82, f"N={len(scores):,}; max bin count={max_count:,}", 11))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def write_summary(path: Path, scores: List[float], thresholds: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerow(["scored_reference_bases", len(scores)])
        writer.writerow(["mean_probability", fmt(sum(scores) / len(scores) if scores else float("nan"))])
        writer.writerow(["median_probability", fmt(quantile(scores, 0.5))])
        writer.writerow(["p90_probability", fmt(quantile(scores, 0.9))])
        writer.writerow(["p95_probability", fmt(quantile(scores, 0.95))])
        writer.writerow(["p99_probability", fmt(quantile(scores, 0.99))])
        writer.writerow(["max_probability", fmt(max(scores) if scores else float("nan"))])
        for threshold in thresholds:
            n = sum(score >= threshold for score in scores)
            frac = n / len(scores) if scores else float("nan")
            writer.writerow([f"calls_at_or_above_{threshold:g}", n])
            writer.writerow([f"fraction_at_or_above_{threshold:g}", fmt(frac)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--top-n", type=int, default=1000)
    args = parser.parse_args()

    rows = []
    scores = []
    with open(args.predictions, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = parse_float(row.get("mean_prob"))
            if score is None:
                continue
            row["_score"] = score
            rows.append(row)
            scores.append(score)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.out_dir / "prediction_summary.tsv", scores, [args.threshold, args.high_confidence_threshold])
    write_histogram(scores, args.out_dir / "score_histogram.svg")

    with open(args.out_dir / f"modified_calls_threshold_{args.threshold:g}.bed", "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for row in rows:
            if row["_score"] >= args.threshold:
                writer.writerow([row["ref_name"], row["start"], row["end"], f"DeepMod:{row['_score']:.6g}"])

    rows_sorted = sorted(rows, key=lambda row: row["_score"], reverse=True)
    fieldnames = [name for name in rows_sorted[0].keys() if name != "_score"] if rows_sorted else []
    with open(args.out_dir / "top_predictions.tsv", "w", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in rows_sorted[: args.top_n]:
                writer.writerow({name: row[name] for name in fieldnames})

    print(f"Wrote DeepMod prediction summary to {args.out_dir}")


if __name__ == "__main__":
    main()
