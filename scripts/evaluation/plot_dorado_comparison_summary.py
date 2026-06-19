#!/usr/bin/env python3
"""Plot summary metrics for DeepMod against Dorado mod-call labels."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence


COLORS = {
    "accuracy": "steelblue",
    "f1": "seagreen",
    "auprc": "darkorange",
}
BAR_Y_MAX = 1.1
SVG_STYLE = '<style>text { font-family: DejaVu Sans, Arial, Helvetica, sans-serif; }</style>'


def parse_float(value: str) -> Optional[float]:
    if value in ("", "NA", "nan", "NaN", None):
        return None
    return float(value)


def svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, body: str, size: int = 12, weight: str = "400", anchor: str = "start", fill: str = "#111111") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{svg_escape(body)}</text>'


def multiline_text(x: float, y: float, lines: Sequence[str], size: int = 10, weight: str = "400", anchor: str = "middle", fill: str = "#111111") -> str:
    spans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else f"{size + 2}"
        spans.append(f'<tspan x="{x:.1f}" dy="{dy}">{svg_escape(line)}</tspan>')
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">' + "".join(spans) + "</text>"


def maybe_write_png(svg_path: Path) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return
    subprocess.run([renderer, "-o", str(svg_path.with_suffix(".png")), str(svg_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def label_lines(row: Dict[str, str], max_chars: int = 24) -> List[str]:
    model = row["mod_model"]
    suffix = model
    if "_" in model:
        suffix = model.split("400bps_")[-1]
    label = f"{row['mod_type']} {suffix}"
    words = label.replace("_", " ").split()
    lines: List[str] = []
    for word in words:
        if not lines or len(lines[-1]) + len(word) + 1 > max_chars:
            lines.append(word)
        else:
            lines[-1] += f" {word}"
    return lines[:4]


def bar_axes(x: float, y: float, w: float, h: float, ylabel: str, title: str) -> List[str]:
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="white" stroke="black" stroke-width="1"/>',
        text(x + w / 2, y - 14, title, 14, "600", "middle"),
        f'<text x="{x - 48:.1f}" y="{y + h / 2:.1f}" font-size="12" transform="rotate(-90 {x - 48:.1f},{y + h / 2:.1f})" text-anchor="middle">{svg_escape(ylabel)}</text>',
    ]
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        py = y + h - (tick / BAR_Y_MAX) * h
        if tick not in (0.0, 1.0):
            parts.append(f'<line x1="{x:.1f}" y1="{py:.1f}" x2="{x + w:.1f}" y2="{py:.1f}" stroke="#d9d9d9" stroke-width="1" stroke-dasharray="1,2"/>')
        parts.append(f'<line x1="{x-5:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{py:.1f}" stroke="black" stroke-width="1"/>')
        parts.append(text(x - 8, py + 4, f"{tick:.1f}", 10, "400", "end"))
    return parts


def load_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_svg(rows: Sequence[Dict[str, str]], output: Path, title: str) -> None:
    width = 1600
    height = 760
    panels = [
        ("accuracy", "Accuracy", "Accuracy at DeepMod threshold", 85, 100),
        ("f1", "F1", "F1 at DeepMod threshold", 600, 100),
        ("auprc", "AUPRC", "Area under PR curve", 1115, 100),
    ]
    panel_w = 410
    panel_h = 330
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        SVG_STYLE,
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        text(width / 2, 36, title, 20, "700", "middle"),
    ]
    n = max(len(rows), 1)
    for metric, ylabel, panel_title, x, y in panels:
        parts.extend(bar_axes(x, y, panel_w, panel_h, ylabel, panel_title))
        group_w = panel_w / n
        bar_w = max(8.0, group_w * 0.58)
        for i, row in enumerate(rows):
            value = parse_float(row.get(metric, "NA"))
            cx = x + i * group_w + group_w / 2
            parts.append(multiline_text(cx, y + panel_h + 26, label_lines(row), 9, "400", "middle"))
            if value is None:
                parts.append(text(cx, y + panel_h / 2, "NA", 10, "600", "middle", "#777777"))
                continue
            bh = max(0.0, min(BAR_Y_MAX, value)) / BAR_Y_MAX * panel_h
            by = y + panel_h - bh
            bx = cx - bar_w / 2
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{COLORS[metric]}"/>')
            parts.append(text(cx, by - 5, f"{value:.3f}", 9, "400", "middle", "#5f5f5f"))
    parts.append(text(width / 2, 710, "Dorado MM/ML reference labels are treated as ground truth; DeepMod score is binary modified probability.", 11, "400", "middle", "#444444"))
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n")
    maybe_write_png(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="DeepMod vs Dorado Modified-Base Calls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    write_svg(rows, args.output, args.title)
    print(f"Wrote Dorado comparison summary plot: {args.output}")


if __name__ == "__main__":
    main()
