#!/usr/bin/env python3
"""Plot precision-recall and F1 summaries for Rockfish and DeepMod tables."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


COLORS = {
    "Rockfish": "darkorange",
    "DeepMod": "steelblue",
    "Our method": "steelblue",
}
RED = "#d62728"
SEAGREEN = "seagreen"
MAX_POLYLINE_POINTS = 1200
BAR_Y_MAX = 1.1
SVG_STYLE = '<style>text { font-family: DejaVu Sans, Arial, Helvetica, sans-serif; }</style>'


@dataclass
class Curve:
    method: str
    dataset: str
    labels: List[int]
    scores: List[float]
    pr_points: List[Tuple[float, float]]
    f1_points: List[Tuple[float, float]]
    n_sites: int
    n_pos: int
    n_neg: int
    auprc: Optional[float]
    auroc: Optional[float]
    best_threshold: Optional[float]
    best_precision: Optional[float]
    best_recall: Optional[float]
    best_f1: Optional[float]


def parse_float(value: str) -> Optional[float]:
    if value in ("", "NA", "nan", "NaN"):
        return None
    return float(value)


def fmt(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.8g}"


def load_table(path: Path, method: str, dataset: str, score_col: str) -> Curve:
    labels: List[int] = []
    scores: List[float] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = parse_float(row.get(score_col, "NA"))
            if score is None:
                continue
            labels.append(int(float(row["gt_label"])))
            scores.append(score)
    return compute_curve(method, dataset, labels, scores)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    tp = 0
    precision_sum = 0.0
    for rank, idx in enumerate(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), start=1):
        if labels[idx] == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / n_pos


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def compute_curve(method: str, dataset: str, labels: Sequence[int], scores: Sequence[float]) -> Curve:
    labels = list(labels)
    scores = list(scores)
    n_sites = len(labels)
    n_pos = int(sum(labels))
    n_neg = n_sites - n_pos
    if n_sites == 0 or n_pos == 0:
        return Curve(method, dataset, labels, scores, [], [], n_sites, n_pos, n_neg, None, None, None, None, None, None)

    order = sorted(range(n_sites), key=lambda i: scores[i], reverse=True)
    pr_points: List[Tuple[float, float]] = [(0.0, 1.0)]
    f1_points: List[Tuple[float, float]] = []
    tp = 0
    fp = 0
    best_threshold = None
    best_precision = None
    best_recall = None
    best_f1 = -1.0

    i = 0
    while i < len(order):
        threshold = scores[order[i]]
        while i < len(order) and scores[order[i]] == threshold:
            if labels[order[i]] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / n_pos
        precision = tp / max(tp + fp, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        pr_points.append((recall, precision))
        f1_points.append((threshold, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return Curve(
        method=method,
        dataset=dataset,
        labels=labels,
        scores=scores,
        pr_points=pr_points,
        f1_points=f1_points,
        n_sites=n_sites,
        n_pos=n_pos,
        n_neg=n_neg,
        auprc=average_precision(labels, scores),
        auroc=roc_auc(labels, scores),
        best_threshold=best_threshold,
        best_precision=best_precision,
        best_recall=best_recall,
        best_f1=best_f1,
    )


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def display_label(value: str) -> str:
    labels = {
        "remora_msssi_cpg_5mc": "Remora M.SssI CpG 5mC",
        "remora_msssi_5mc_deepmod_all_positions": "Remora M.SssI 5mC All Positions",
        "remora_msssi_5mc_rockfish_callable_cpg": "Remora M.SssI Rockfish-Callable CpG 5mC",
    }
    return labels.get(value, value.replace("_", " "))


def label_lines(value: str, max_chars: int = 28) -> List[str]:
    value = display_label(value)
    if len(value) <= max_chars:
        return [value]
    words = value.split()
    if len(words) < 2:
        return [value]
    lines = [words[0]]
    for word in words[1:]:
        if len(lines[-1]) + len(word) + 1 <= max_chars:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines


def thin_points(points: Sequence[Tuple[float, float]], max_points: int = MAX_POLYLINE_POINTS) -> List[Tuple[float, float]]:
    if len(points) <= max_points:
        return list(points)
    if max_points < 2:
        return [points[0]]
    n = len(points)
    thinned: List[Tuple[float, float]] = []
    last_idx = -1
    for i in range(max_points):
        idx = round(i * (n - 1) / (max_points - 1))
        if idx != last_idx:
            thinned.append(points[idx])
            last_idx = idx
    return thinned


def line(points: Sequence[Tuple[float, float]], color: str, width: float = 2.0, dash: str = "") -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in thin_points(points))
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'


def text(x: float, y: float, body: str, size: int = 12, weight: str = "400", anchor: str = "start", fill: str = "#111111") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{svg_escape(body)}</text>'


def multiline_text(x: float, y: float, lines: Sequence[str], size: int = 10, weight: str = "400", anchor: str = "middle", fill: str = "#111111") -> str:
    if not lines:
        return ""
    spans = []
    for i, line_body in enumerate(lines):
        dy = "0" if i == 0 else f"{size + 2}"
        spans.append(f'<tspan x="{x:.1f}" dy="{dy}">{svg_escape(line_body)}</tspan>')
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">' + "".join(spans) + "</text>"


def circle(x: float, y: float, r: float = 4.0, fill: str = RED) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="1"/>'


def maybe_write_png(svg_path: Path) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return
    png_path = svg_path.with_suffix(".png")
    subprocess.run([renderer, "-o", str(png_path), str(svg_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def panel_axes(x: float, y: float, w: float, h: float, xlabel: str, ylabel: str, title: str) -> List[str]:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="white" stroke="black" stroke-width="1"/>',
        text(x + w / 2, y - 14, title, 14, "600", "middle"),
        text(x + w / 2, y + h + 42, xlabel, 12, "400", "middle"),
        f'<text x="{x - 42:.1f}" y="{y + h / 2:.1f}" font-size="12" transform="rotate(-90 {x - 42:.1f},{y + h / 2:.1f})" text-anchor="middle">{svg_escape(ylabel)}</text>',
    ]
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        px = x + t * w
        py = y + h - t * h
        parts.append(f'<line x1="{px:.1f}" y1="{y+h:.1f}" x2="{px:.1f}" y2="{y+h+5:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(f'<line x1="{x-5:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{py:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(text(px, y + h + 18, f"{t:g}", 10, "400", "middle"))
        parts.append(text(x - 8, py + 4, f"{t:g}", 10, "400", "end"))
    return parts


def bar_axes(x: float, y: float, w: float, h: float, xlabel: str, ylabel: str, title: str) -> List[str]:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="white" stroke="black" stroke-width="1"/>',
        text(x + w / 2, y - 14, title, 14, "600", "middle"),
        text(x + w / 2, y + h + 56, xlabel, 12, "400", "middle"),
        f'<text x="{x - 48:.1f}" y="{y + h / 2:.1f}" font-size="12" transform="rotate(-90 {x - 48:.1f},{y + h / 2:.1f})" text-anchor="middle">{svg_escape(ylabel)}</text>',
    ]
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        py = y + h - (t / BAR_Y_MAX) * h
        if t not in (0.0, 1.0):
            parts.append(f'<line x1="{x:.1f}" y1="{py:.1f}" x2="{x + w:.1f}" y2="{py:.1f}" stroke="#d9d9d9" stroke-width="1" stroke-dasharray="1,2"/>')
        parts.append(f'<line x1="{x-5:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{py:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(text(x - 8, py + 4, f"{t:.1f}", 10, "400", "end"))
    return parts


def transform_points(points: Sequence[Tuple[float, float]], x: float, y: float, w: float, h: float) -> List[Tuple[float, float]]:
    return [(x + max(0.0, min(1.0, px)) * w, y + h - max(0.0, min(1.0, py)) * h) for px, py in points]


def write_dataset_svg(dataset: str, curves: Sequence[Curve], output: Path) -> None:
    width = 1200
    height = 600
    pr_x, pr_y, pr_w, panel_h = 80, 98, 650, 300
    f1_x, f1_y, f1_w = 845, 98, 290
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, f"{display_label(dataset)} — Precision-Recall / Threshold Sweep", 18, "700", "middle"),
    ]
    parts.extend(panel_axes(pr_x, pr_y, pr_w, panel_h, "Recall", "Precision", "Precision-Recall Curve"))
    parts.extend(panel_axes(f1_x, f1_y, f1_w, panel_h, "Threshold", "F1 Score", "F1 vs. Threshold"))

    legend_x = 100
    legend_y = 468
    for i, curve in enumerate(curves):
        color = COLORS.get(curve.method, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][i % 4])
        if curve.pr_points:
            parts.append(line(transform_points(curve.pr_points, pr_x, pr_y, pr_w, panel_h), color, 2.2))
            if curve.best_precision is not None and curve.best_recall is not None:
                bx, by = transform_points([(curve.best_recall, curve.best_precision)], pr_x, pr_y, pr_w, panel_h)[0]
                parts.append(circle(bx, by, 4.5, RED))
                parts.append(f'<line x1="{bx:.1f}" y1="{pr_y:.1f}" x2="{bx:.1f}" y2="{pr_y + panel_h:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
                parts.append(f'<line x1="{pr_x:.1f}" y1="{by:.1f}" x2="{pr_x + pr_w:.1f}" y2="{by:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
        if curve.f1_points:
            parts.append(line(transform_points(sorted(curve.f1_points), f1_x, f1_y, f1_w, panel_h), SEAGREEN if len(curves) == 1 else color, 2.2))
            if curve.best_threshold is not None and curve.best_f1 is not None:
                bx, by = transform_points([(curve.best_threshold, curve.best_f1)], f1_x, f1_y, f1_w, panel_h)[0]
                parts.append(circle(bx, by, 4.5, RED))
                parts.append(f'<line x1="{bx:.1f}" y1="{f1_y:.1f}" x2="{bx:.1f}" y2="{f1_y + panel_h:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
                parts.append(f'<line x1="{f1_x:.1f}" y1="{by:.1f}" x2="{f1_x + f1_w:.1f}" y2="{by:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
        ly = legend_y + i * 38
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+28}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(text(legend_x + 36, ly + 4, f"{curve.method}: AUPRC={fmt(curve.auprc)}  best F1={fmt(curve.best_f1)}", 11))
        detail = (
            f"threshold={fmt(curve.best_threshold)}  precision={fmt(curve.best_precision)}  "
            f"recall={fmt(curve.best_recall)}  N={curve.n_sites:,}  modified={curve.n_pos:,}  unmodified={curve.n_neg:,}"
        )
        parts.append(text(legend_x + 36, ly + 20, detail, 10, "400", "start", "#444444"))

    if not any(c.pr_points for c in curves):
        parts.append(text(width / 2, 250, "No positive labels with scored sites; PR/F1 undefined.", 14, "600", "middle"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def write_metrics(metrics: Sequence[Curve], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "method",
                "scored_sites",
                "positive_sites",
                "negative_sites",
                "auprc",
                "auroc",
                "best_threshold",
                "best_precision",
                "best_recall",
                "best_f1",
            ]
        )
        for c in metrics:
            writer.writerow(
                [
                    c.dataset,
                    c.method,
                    c.n_sites,
                    c.n_pos,
                    c.n_neg,
                    fmt(c.auprc),
                    fmt(c.auroc),
                    fmt(c.best_threshold),
                    fmt(c.best_precision),
                    fmt(c.best_recall),
                    fmt(c.best_f1),
                ]
            )


def write_summary_svg(metrics: Sequence[Curve], datasets: Sequence[str], output: Path) -> None:
    methods = []
    for c in metrics:
        if c.method not in methods:
            methods.append(c.method)
    width = 1300
    height = 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, "Method Summary by Dataset", 18, "700", "middle"),
    ]
    panels = [("AUPRC", "auprc", 95, 92), ("Best F1", "best_f1", 725, 92)]
    lookup = {(c.dataset, c.method): c for c in metrics}
    for title, attr, x, y in panels:
        w, h = 500, 320
        parts.extend(bar_axes(x, y, w, h, "Dataset", title, title))
        group_w = w / max(len(datasets), 1)
        bar_w = group_w / max(len(methods) + 1, 2)
        for di, dataset in enumerate(datasets):
            cx = x + di * group_w + group_w / 2
            parts.append(multiline_text(cx, y + h + 24, label_lines(dataset, 30), 10, "400", "middle"))
            for mi, method in enumerate(methods):
                c = lookup.get((dataset, method))
                value = getattr(c, attr) if c is not None else None
                if value is None:
                    continue
                color = COLORS.get(method, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][mi % 4])
                bx = x + di * group_w + (mi + 0.6) * bar_w
                bh = max(0.0, min(BAR_Y_MAX, value)) / BAR_Y_MAX * h
                by = y + h - bh
                parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.8:.1f}" height="{bh:.1f}" fill="{color}"/>')
                parts.append(text(bx + bar_w * 0.4, by - 5, f"{value:.3f}", 10, "400", "middle", "#5f5f5f"))
        for mi, method in enumerate(methods):
            color = COLORS.get(method, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][mi % 4])
            lx = x + 10 + mi * 145
            parts.append(f'<rect x="{lx}" y="{y+h+86}" width="14" height="14" fill="{color}"/>')
            parts.append(text(lx + 20, y + h + 98, method, 11))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rockfish-dir", type=Path, default=None)
    parser.add_argument("--deepmod-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["control", "5mC", "5hmC", "6mA"])
    parser.add_argument("--score-col", default="mean_prob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_curves: List[Curve] = []
    by_dataset: Dict[str, List[Curve]] = {dataset: [] for dataset in args.datasets}

    for dataset in args.datasets:
        if args.rockfish_dir is not None:
            path = args.rockfish_dir / f"{dataset}.rockfish_reference_labels.tsv"
            if path.exists():
                c = load_table(path, "Rockfish", dataset, args.score_col)
                all_curves.append(c)
                by_dataset[dataset].append(c)
        if args.deepmod_dir is not None:
            path = args.deepmod_dir / f"{dataset}.deepmod_reference_predictions.tsv"
            if path.exists():
                c = load_table(path, "DeepMod", dataset, args.score_col)
                all_curves.append(c)
                by_dataset[dataset].append(c)

    plot_dir = args.out_dir / "plots"
    for dataset, curves in by_dataset.items():
        if curves:
            write_dataset_svg(dataset, curves, plot_dir / f"{dataset}.pr_f1.svg")
    write_metrics(all_curves, args.out_dir / "comparison_metrics.tsv")
    write_summary_svg(all_curves, args.datasets, plot_dir / "method_summary.svg")
    print(f"Wrote comparison metrics and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
