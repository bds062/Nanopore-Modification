#!/usr/bin/env python3
"""Class- and sample-split plots for binary modification evaluations."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CLASS_COLORS = {
    "modified": "steelblue",
    "unmodified": "darkorange",
}
RED = "#d62728"
SEAGREEN = "seagreen"
MAX_POLYLINE_POINTS = 1200
BAR_Y_MAX = 1.1
SVG_STYLE = '<style>text { font-family: DejaVu Sans, Arial, Helvetica, sans-serif; }</style>'


@dataclass
class Curve:
    sample: str
    class_name: str
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


def compute_curve(sample: str, class_name: str, labels: Sequence[int], scores: Sequence[float], require_both_classes: bool = False) -> Curve:
    labels = list(labels)
    scores = list(scores)
    n_sites = len(labels)
    n_pos = int(sum(labels))
    n_neg = n_sites - n_pos
    if n_sites == 0 or n_pos == 0 or (require_both_classes and n_neg == 0):
        return Curve(sample, class_name, labels, scores, [], [], n_sites, n_pos, n_neg, None, None, None, None, None, None)

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
        sample=sample,
        class_name=class_name,
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
        "all": "All",
        "can": "Canonical",
        "mod": "Modified",
        "modified": "Modified",
        "unmodified": "Unmodified",
    }
    return labels.get(value, value.replace("_", " "))


def text(x: float, y: float, body: str, size: int = 12, weight: str = "400", anchor: str = "start", fill: str = "#111111") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{svg_escape(body)}</text>'


def maybe_write_png(svg_path: Path) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return
    png_path = svg_path.with_suffix(".png")
    subprocess.run([renderer, "-o", str(png_path), str(svg_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def circle(x: float, y: float, r: float = 4.0, fill: str = RED) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="1"/>'


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


def write_pr_f1_svg(sample: str, curves: Sequence[Curve], output: Path, method_name: str) -> None:
    width = 1200
    height = 600
    pr_x, pr_y, pr_w, panel_h = 80, 98, 650, 300
    f1_x, f1_y, f1_w = 845, 98, 290
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, f"{display_label(sample)} — {method_name} Class Precision-Recall / Threshold Sweep", 18, "700", "middle"),
    ]
    parts.extend(panel_axes(pr_x, pr_y, pr_w, panel_h, "Recall", "Precision", "Precision-Recall Curve"))
    parts.extend(panel_axes(f1_x, f1_y, f1_w, panel_h, "Class score threshold", "F1 Score", "F1 vs. Threshold"))

    legend_x = 100
    legend_y = 468
    for i, curve in enumerate(curves):
        color = CLASS_COLORS.get(curve.class_name, "#444444")
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
        parts.append(text(legend_x + 36, ly + 4, f"{display_label(curve.class_name)}: AUPRC={fmt(curve.auprc)}  best F1={fmt(curve.best_f1)}", 11))
        detail = (
            f"threshold={fmt(curve.best_threshold)}  precision={fmt(curve.best_precision)}  "
            f"recall={fmt(curve.best_recall)}  N={curve.n_sites:,}  class positives={curve.n_pos:,}  class negatives={curve.n_neg:,}"
        )
        parts.append(text(legend_x + 36, ly + 20, detail, 10, "400", "start", "#444444"))

    if not any(c.pr_points for c in curves):
        parts.append(text(width / 2, 250, "No positive labels with scored sites; PR/F1 undefined.", 14, "600", "middle"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def write_summary_svg(curves: Sequence[Curve], samples: Sequence[str], output: Path, method_name: str) -> None:
    width = 1300
    height = 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, f"{method_name} Class Metrics by Sample", 18, "700", "middle"),
    ]
    lookup = {(c.sample, c.class_name): c for c in curves}
    panels = [("AUPRC", "auprc", 95, 92), ("Best F1", "best_f1", 725, 92)]
    classes = ["modified", "unmodified"]
    for title, attr, x, y in panels:
        w, h = 500, 320
        parts.extend(bar_axes(x, y, w, h, "Sample", title, title))
        group_w = w / max(len(samples), 1)
        bar_w = group_w / 3.0
        for si, sample in enumerate(samples):
            cx = x + si * group_w + group_w / 2
            parts.append(text(cx, y + h + 24, display_label(sample), 10, "400", "middle"))
            for ci, class_name in enumerate(classes):
                c = lookup.get((sample, class_name))
                value = getattr(c, attr) if c is not None else None
                if value is None:
                    continue
                color = CLASS_COLORS[class_name]
                bx = x + si * group_w + (ci + 0.5) * bar_w
                bh = max(0.0, min(BAR_Y_MAX, value)) / BAR_Y_MAX * h
                by = y + h - bh
                parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w*0.8:.1f}" height="{bh:.1f}" fill="{color}"/>')
                parts.append(text(bx + bar_w * 0.4, by - 5, f"{value:.3f}", 10, "400", "middle", "#5f5f5f"))
        for ci, class_name in enumerate(classes):
            lx = x + 10 + ci * 130
            parts.append(f'<rect x="{lx}" y="{y+h+86}" width="14" height="14" fill="{CLASS_COLORS[class_name]}"/>')
            parts.append(text(lx + 20, y + h + 98, display_label(class_name), 11))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    tn = fp = fn = tp = 0
    for label, score in zip(labels, scores):
        pred = int(score >= threshold)
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def blues(value: float) -> str:
    value = max(0.0, min(1.0, value))
    lo = (247, 251, 255)
    hi = (8, 48, 107)
    rgb = tuple(int(lo[i] + (hi[i] - lo[i]) * value) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def write_confusion_svg(sample: str, matrix: Dict[str, float], threshold: float, output: Path) -> None:
    width = 620
    height = 470
    x, y, cell = 180, 125, 135
    unmod_total = int(matrix["tn"]) + int(matrix["fp"])
    mod_total = int(matrix["fn"]) + int(matrix["tp"])
    cells = [
        ("TN", int(matrix["tn"]), unmod_total, x, y),
        ("FP", int(matrix["fp"]), unmod_total, x + cell, y),
        ("FN", int(matrix["fn"]), mod_total, x, y + cell),
        ("TP", int(matrix["tp"]), mod_total, x + cell, y + cell),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, f"{display_label(sample)}: Confusion Matrix", 18, "700", "middle"),
        text(width / 2, 60, f"threshold = {threshold:.4f}", 12, "400", "middle"),
        text(x + cell, y - 34, "Predicted label", 12, "400", "middle"),
        f'<text x="{x - 70:.1f}" y="{y + cell:.1f}" font-size="12" transform="rotate(-90 {x - 70:.1f},{y + cell:.1f})" text-anchor="middle">True label</text>',
        text(x + cell / 2, y - 10, "Unmodified", 11, "400", "middle"),
        text(x + cell * 1.5, y - 10, "Modified", 11, "400", "middle"),
        text(x - 10, y + cell / 2 + 4, "Unmodified", 11, "400", "end"),
        text(x - 10, y + cell * 1.5 + 4, "Modified", 11, "400", "end"),
    ]
    for _name, count, row_total, cx, cy in cells:
        prop = count / row_total if row_total else 0.0
        color = blues(prop)
        text_color = "white" if prop > 0.5 else "black"
        parts.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" fill="{color}" stroke="white" stroke-width="2"/>')
        parts.append(text(cx + cell / 2, cy + 58, f"{count:,}", 18, "700", "middle", text_color))
        parts.append(text(cx + cell / 2, cy + 84, f"({prop * 100:.1f}%)", 11, "400", "middle", text_color))
    cbar_x, cbar_y, cbar_w, cbar_h = x + 2 * cell + 45, y, 18, 2 * cell
    steps = 20
    for i in range(steps):
        frac0 = i / steps
        yy = cbar_y + cbar_h - (i + 1) * cbar_h / steps
        parts.append(f'<rect x="{cbar_x:.1f}" y="{yy:.1f}" width="{cbar_w:.1f}" height="{cbar_h/steps + 0.5:.1f}" fill="{blues(frac0)}"/>')
    parts.append(f'<rect x="{cbar_x:.1f}" y="{cbar_y:.1f}" width="{cbar_w:.1f}" height="{cbar_h:.1f}" fill="none" stroke="black" stroke-width="0.8"/>')
    parts.append(text(cbar_x + cbar_w + 8, cbar_y + 5, "1.0", 10, "400", "start"))
    parts.append(text(cbar_x + cbar_w + 8, cbar_y + cbar_h + 4, "0.0", 10, "400", "start"))
    parts.append(f'<text x="{cbar_x + cbar_w + 42:.1f}" y="{cbar_y + cbar_h / 2:.1f}" font-size="10" transform="rotate(-90 {cbar_x + cbar_w + 42:.1f},{cbar_y + cbar_h / 2:.1f})" text-anchor="middle">Proportion of true class</text>')
    footer = (
        f"accuracy={fmt(matrix['accuracy'])}  precision={fmt(matrix['precision'])}  "
        f"recall={fmt(matrix['recall'])}  F1={fmt(matrix['f1'])}"
    )
    parts.append(text(width / 2, 425, footer, 12, "400", "middle"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = parse_float(row.get("mean_prob", "NA"))
            if score is None:
                continue
            label = int(float(row["gt_label"]))
            rows.append({"sample": row["sample"], "label": label, "score": score})
    return rows


def write_metrics(curves: Sequence[Curve], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sample",
                "class",
                "scored_sites",
                "positive_sites_for_class",
                "negative_sites_for_class",
                "auprc",
                "auroc",
                "best_threshold_in_class_score",
                "best_precision",
                "best_recall",
                "best_f1",
            ]
        )
        for c in curves:
            writer.writerow(
                [
                    c.sample,
                    c.class_name,
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


def write_confusions(rows: Sequence[Dict[str, object]], threshold: float, output: Path, plot_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = ["all", "can", "mod"]
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample", "threshold", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "specificity", "f1"])
        for sample in samples:
            subset = rows if sample == "all" else [r for r in rows if r["sample"] == sample]
            labels = [int(r["label"]) for r in subset]
            scores = [float(r["score"]) for r in subset]
            matrix = confusion(labels, scores, threshold)
            writer.writerow(
                [
                    sample,
                    fmt(threshold),
                    matrix["tn"],
                    matrix["fp"],
                    matrix["fn"],
                    matrix["tp"],
                    fmt(matrix["accuracy"]),
                    fmt(matrix["precision"]),
                    fmt(matrix["recall"]),
                    fmt(matrix["specificity"]),
                    fmt(matrix["f1"]),
                ]
            )
            write_confusion_svg(sample, matrix, threshold, plot_dir / f"{sample}.confusion.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method-name", default="DeepMod")
    parser.add_argument(
        "--require-both-classes",
        action="store_true",
        help="Set PR/F1 to NA for single-class sample splits instead of reporting trivial perfect curves.",
    )
    parser.add_argument(
        "--confusion-threshold",
        default="auto",
        help="Modified-class probability threshold for confusion matrices. Use 'auto' for all-sample best modified F1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    samples = ["all", "can", "mod"]
    curves: List[Curve] = []
    for sample in samples:
        subset = rows if sample == "all" else [r for r in rows if r["sample"] == sample]
        labels_mod = [int(r["label"]) for r in subset]
        scores_mod = [float(r["score"]) for r in subset]
        labels_unmod = [1 - label for label in labels_mod]
        scores_unmod = [1.0 - score for score in scores_mod]
        curves.append(compute_curve(sample, "modified", labels_mod, scores_mod, args.require_both_classes))
        curves.append(compute_curve(sample, "unmodified", labels_unmod, scores_unmod, args.require_both_classes))

    out_dir = args.out_dir
    plot_dir = out_dir / "plots"
    write_metrics(curves, out_dir / "class_metrics.tsv")

    for sample in samples:
        write_pr_f1_svg(sample, [c for c in curves if c.sample == sample], plot_dir / f"{sample}.class_pr_f1.svg", args.method_name)
    write_summary_svg(curves, samples, plot_dir / "class_metric_summary.svg", args.method_name)

    all_modified = next(c for c in curves if c.sample == "all" and c.class_name == "modified")
    if args.confusion_threshold == "auto":
        if all_modified.best_threshold is None:
            raise SystemExit("Cannot use auto threshold; all-sample modified class has no best threshold")
        threshold = all_modified.best_threshold
    else:
        threshold = float(args.confusion_threshold)
    write_confusions(rows, threshold, out_dir / "confusion_matrices.tsv", plot_dir)

    print(f"Wrote {args.method_name} class breakdown to {out_dir}")
    print(f"Confusion matrices use modified threshold={threshold:.8g}")


if __name__ == "__main__":
    main()
