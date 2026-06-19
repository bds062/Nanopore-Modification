#!/usr/bin/env python3
"""Compare DeepMod scores against Dorado-derived reference labels."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


RED = "#d62728"
SEAGREEN = "seagreen"
DEEPMOD_BLUE = "steelblue"
BAR_Y_MAX = 1.1
MAX_POLYLINE_POINTS = 1200
SVG_STYLE = '<style>text { font-family: DejaVu Sans, Arial, Helvetica, sans-serif; }</style>'


@dataclass
class Curve:
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
    if value in ("", "NA", "nan", "NaN", None):
        return None
    return float(value)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.8g}"


def svg_escape(text_value: str) -> str:
    return text_value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, body: str, size: int = 12, weight: str = "400", anchor: str = "start", fill: str = "#111111") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{svg_escape(body)}</text>'


def maybe_write_png(svg_path: Path) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return
    subprocess.run([renderer, "-o", str(svg_path.with_suffix(".png")), str(svg_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


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


def compute_curve(labels: Sequence[int], scores: Sequence[float]) -> Curve:
    labels = list(labels)
    scores = list(scores)
    n_sites = len(labels)
    n_pos = int(sum(labels))
    n_neg = n_sites - n_pos
    if n_sites == 0 or n_pos == 0:
        return Curve(labels, scores, [], [], n_sites, n_pos, n_neg, None, None, None, None, None, None)

    order = sorted(range(n_sites), key=lambda i: scores[i], reverse=True)
    pr_points: List[Tuple[float, float]] = [(0.0, 1.0)]
    f1_points: List[Tuple[float, float]] = []
    tp = fp = 0
    best_threshold: Optional[float] = None
    best_precision: Optional[float] = None
    best_recall: Optional[float] = None
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

    return Curve(labels, scores, pr_points, f1_points, n_sites, n_pos, n_neg, average_precision(labels, scores), roc_auc(labels, scores), best_threshold, best_precision, best_recall, best_f1)


def confusion(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, Optional[float]]:
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
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    accuracy = (tp + tn) / total if total else None
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def thin_points(points: Sequence[Tuple[float, float]], max_points: int = MAX_POLYLINE_POINTS) -> List[Tuple[float, float]]:
    if len(points) <= max_points:
        return list(points)
    thinned: List[Tuple[float, float]] = []
    last_idx = -1
    for i in range(max_points):
        idx = round(i * (len(points) - 1) / (max_points - 1))
        if idx != last_idx:
            thinned.append(points[idx])
            last_idx = idx
    return thinned


def polyline(points: Sequence[Tuple[float, float]], color: str, width: float = 2.2) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in thin_points(points))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def transform(points: Sequence[Tuple[float, float]], x: float, y: float, w: float, h: float) -> List[Tuple[float, float]]:
    return [(x + max(0.0, min(1.0, px)) * w, y + h - max(0.0, min(1.0, py)) * h) for px, py in points]


def axes(x: float, y: float, w: float, h: float, xlabel: str, ylabel: str, title: str) -> List[str]:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="white" stroke="black" stroke-width="1"/>',
        text(x + w / 2, y - 14, title, 14, "600", "middle"),
        text(x + w / 2, y + h + 42, xlabel, 12, "400", "middle"),
        f'<text x="{x - 42:.1f}" y="{y + h / 2:.1f}" font-size="12" transform="rotate(-90 {x - 42:.1f},{y + h / 2:.1f})" text-anchor="middle">{svg_escape(ylabel)}</text>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        px = x + tick * w
        py = y + h - tick * h
        parts.append(f'<line x1="{px:.1f}" y1="{y+h:.1f}" x2="{px:.1f}" y2="{y+h+5:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(f'<line x1="{x-5:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{py:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(text(px, y + h + 18, f"{tick:g}", 10, "400", "middle"))
        parts.append(text(x - 8, py + 4, f"{tick:g}", 10, "400", "end"))
    return parts


def circle(x: float, y: float, r: float = 4.5, fill: str = RED) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="white" stroke-width="1"/>'


def write_pr_f1_svg(curve: Curve, dataset: str, mod_type: str, model_name: str, output: Path) -> None:
    width = 1200
    height = 600
    pr_x, pr_y, pr_w, panel_h = 80, 98, 650, 300
    f1_x, f1_y, f1_w = 845, 98, 290
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, f"{dataset}: DeepMod vs Dorado {mod_type}", 18, "700", "middle"),
        text(width / 2, 58, model_name, 11, "400", "middle", "#444444"),
    ]
    parts.extend(axes(pr_x, pr_y, pr_w, panel_h, "Recall", "Precision", "Precision-Recall Curve"))
    parts.extend(axes(f1_x, f1_y, f1_w, panel_h, "DeepMod threshold", "F1 Score", "F1 vs. Threshold"))
    if curve.pr_points:
        parts.append(polyline(transform(curve.pr_points, pr_x, pr_y, pr_w, panel_h), DEEPMOD_BLUE))
        if curve.best_precision is not None and curve.best_recall is not None:
            bx, by = transform([(curve.best_recall, curve.best_precision)], pr_x, pr_y, pr_w, panel_h)[0]
            parts.append(circle(bx, by))
            parts.append(f'<line x1="{bx:.1f}" y1="{pr_y:.1f}" x2="{bx:.1f}" y2="{pr_y + panel_h:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
            parts.append(f'<line x1="{pr_x:.1f}" y1="{by:.1f}" x2="{pr_x + pr_w:.1f}" y2="{by:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
    if curve.f1_points:
        parts.append(polyline(transform(sorted(curve.f1_points), f1_x, f1_y, f1_w, panel_h), SEAGREEN))
        if curve.best_threshold is not None and curve.best_f1 is not None:
            bx, by = transform([(curve.best_threshold, curve.best_f1)], f1_x, f1_y, f1_w, panel_h)[0]
            parts.append(circle(bx, by))
            parts.append(f'<line x1="{bx:.1f}" y1="{f1_y:.1f}" x2="{bx:.1f}" y2="{f1_y + panel_h:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
            parts.append(f'<line x1="{f1_x:.1f}" y1="{by:.1f}" x2="{f1_x + f1_w:.1f}" y2="{by:.1f}" stroke="{RED}" stroke-width="0.8" stroke-dasharray="3,3"/>')
    else:
        parts.append(text(width / 2, 250, "No Dorado-positive labels with scored DeepMod sites; PR/F1 undefined.", 14, "600", "middle"))

    detail = (
        f"AUPRC={fmt(curve.auprc)}  AUROC={fmt(curve.auroc)}  best F1={fmt(curve.best_f1)}  "
        f"threshold={fmt(curve.best_threshold)}  N={curve.n_sites:,}  Dorado+={curve.n_pos:,}  Dorado-={curve.n_neg:,}"
    )
    parts.append(f'<line x1="100" y1="470" x2="128" y2="470" stroke="{DEEPMOD_BLUE}" stroke-width="3"/>')
    parts.append(text(136, 474, detail, 11))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def blues(value: float) -> str:
    value = max(0.0, min(1.0, value))
    lo = (247, 251, 255)
    hi = (8, 48, 107)
    rgb = tuple(int(lo[i] + (hi[i] - lo[i]) * value) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def write_confusion_svg(matrix: Dict[str, Optional[float]], dataset: str, mod_type: str, threshold: float, output: Path) -> None:
    width = 620
    height = 470
    x, y, cell = 180, 125, 135
    unmod_total = int(matrix["tn"] or 0) + int(matrix["fp"] or 0)
    mod_total = int(matrix["fn"] or 0) + int(matrix["tp"] or 0)
    cells = [
        ("TN", int(matrix["tn"] or 0), unmod_total, x, y),
        ("FP", int(matrix["fp"] or 0), unmod_total, x + cell, y),
        ("FN", int(matrix["fn"] or 0), mod_total, x, y + cell),
        ("TP", int(matrix["tp"] or 0), mod_total, x + cell, y + cell),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 34, f"{dataset}: DeepMod vs Dorado {mod_type}", 18, "700", "middle"),
        text(width / 2, 60, f"DeepMod threshold = {threshold:.4f}", 12, "400", "middle"),
        text(x + cell, y - 34, "DeepMod label", 12, "400", "middle"),
        f'<text x="{x - 70:.1f}" y="{y + cell:.1f}" font-size="12" transform="rotate(-90 {x - 70:.1f},{y + cell:.1f})" text-anchor="middle">Dorado label</text>',
        text(x + cell / 2, y - 10, "Unmodified", 11, "400", "middle"),
        text(x + cell * 1.5, y - 10, "Modified", 11, "400", "middle"),
        text(x - 10, y + cell / 2 + 4, "Unmodified", 11, "400", "end"),
        text(x - 10, y + cell * 1.5 + 4, "Modified", 11, "400", "end"),
    ]
    for _name, count, row_total, cx, cy in cells:
        prop = count / row_total if row_total else 0.0
        fill = blues(prop)
        text_color = "white" if prop > 0.5 else "black"
        parts.append(f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="2"/>')
        parts.append(text(cx + cell / 2, cy + 58, f"{count:,}", 18, "700", "middle", text_color))
        parts.append(text(cx + cell / 2, cy + 84, f"({prop * 100:.1f}%)", 11, "400", "middle", text_color))
    footer = (
        f"accuracy={fmt(matrix['accuracy'])}  precision={fmt(matrix['precision'])}  "
        f"recall={fmt(matrix['recall'])}  F1={fmt(matrix['f1'])}"
    )
    parts.append(text(width / 2, 425, footer, 12, "400", "middle"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def load_deepmod(path: Path, score_col: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    rows: Dict[Tuple[str, int], Dict[str, str]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = parse_float(row.get(score_col, "NA"))
            if score is None:
                continue
            rows[(row["ref_name"], int(row["ref_pos"]))] = row
    return rows


def joined_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    deepmod = load_deepmod(args.deepmod, args.deepmod_score_col)
    rows: List[Dict[str, str]] = []
    with open(args.dorado, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for dorado_row in reader:
            label = int(float(dorado_row[args.dorado_label_col]))
            dorado_score = parse_float(dorado_row.get(args.dorado_score_col, "NA"))
            key = (dorado_row["ref_name"], int(dorado_row["ref_pos"]))
            deepmod_row = deepmod.get(key)
            if deepmod_row is None:
                continue
            deepmod_score = parse_float(deepmod_row.get(args.deepmod_score_col, "NA"))
            if deepmod_score is None:
                continue
            rows.append(
                {
                    "dataset": args.dataset or dorado_row.get("dataset", ""),
                    "mod_model": args.mod_model_name or dorado_row.get("mod_model", ""),
                    "mod_type": args.mod_type or dorado_row.get("mod_type", ""),
                    "ref_name": key[0],
                    "ref_pos": str(key[1]),
                    "ref_base": dorado_row.get("ref_base", deepmod_row.get("ref_base", "N")),
                    "dorado_label": str(label),
                    "dorado_score": fmt(dorado_score),
                    "dorado_n_calls": dorado_row.get("n_calls", "NA"),
                    "deepmod_score": fmt(deepmod_score),
                    "deepmod_label": str(int(deepmod_score >= args.deepmod_threshold)),
                }
            )
    return rows


def write_joined(rows: Sequence[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "mod_model", "mod_type", "ref_name", "ref_pos", "ref_base", "dorado_label", "dorado_score", "dorado_n_calls", "deepmod_score", "deepmod_label"]
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(args: argparse.Namespace, rows: Sequence[Dict[str, str]], curve: Curve, matrix: Dict[str, Optional[float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset or (rows[0]["dataset"] if rows else "")
    mod_model = args.mod_model_name or (rows[0]["mod_model"] if rows else "")
    mod_type = args.mod_type or (rows[0]["mod_type"] if rows else "")
    fields = [
        "dataset",
        "mod_model",
        "mod_type",
        "scored_sites",
        "dorado_positive_sites",
        "dorado_negative_sites",
        "deepmod_threshold",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auprc",
        "auroc",
        "best_threshold",
        "best_precision",
        "best_recall",
        "best_f1",
    ]
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerow(
            [
                dataset,
                mod_model,
                mod_type,
                curve.n_sites,
                curve.n_pos,
                curve.n_neg,
                fmt(args.deepmod_threshold),
                fmt(matrix["accuracy"]),
                fmt(matrix["precision"]),
                fmt(matrix["recall"]),
                fmt(matrix["specificity"]),
                fmt(matrix["f1"]),
                fmt(curve.auprc),
                fmt(curve.auroc),
                fmt(curve.best_threshold),
                fmt(curve.best_precision),
                fmt(curve.best_recall),
                fmt(curve.best_f1),
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepmod", type=Path, required=True)
    parser.add_argument("--dorado", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--mod-model-name", default="")
    parser.add_argument("--mod-type", default="")
    parser.add_argument("--deepmod-score-col", default="mean_prob")
    parser.add_argument("--dorado-score-col", default="mean_prob")
    parser.add_argument("--dorado-label-col", default="gt_label")
    parser.add_argument("--deepmod-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = joined_rows(args)
    labels = [int(row["dorado_label"]) for row in rows]
    scores = [float(row["deepmod_score"]) for row in rows]
    curve = compute_curve(labels, scores)
    matrix = confusion(labels, scores, args.deepmod_threshold)

    dataset = args.dataset or (rows[0]["dataset"] if rows else "dataset")
    mod_model = args.mod_model_name or (rows[0]["mod_model"] if rows else "dorado_model")
    mod_type = args.mod_type or (rows[0]["mod_type"] if rows else "mod")
    safe_mod_type = mod_type.replace("/", "_")

    write_joined(rows, args.out_dir / f"{safe_mod_type}.deepmod_vs_dorado.tsv")
    write_metrics(args, rows, curve, matrix, args.out_dir / f"{safe_mod_type}.metrics.tsv")
    write_pr_f1_svg(curve, dataset, mod_type, mod_model, args.out_dir / "plots" / f"{safe_mod_type}.pr_f1.svg")
    write_confusion_svg(matrix, dataset, mod_type, args.deepmod_threshold, args.out_dir / "plots" / f"{safe_mod_type}.confusion.svg")
    print(f"Wrote DeepMod-vs-Dorado comparison for {mod_model} {mod_type}: {args.out_dir}")


if __name__ == "__main__":
    main()
