#!/usr/bin/env python3
"""Create a publication-style DeepMod signal-pileup figure from an HDF5 file.

Each row in the figure is the reference track or one aligned read.  The plotted
curve is channel 0 of the DeepMod tensor: expected current for the reference row
and resampled normalized current for read rows.
"""

import argparse
import html
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


DEFAULT_CHANNEL_NAMES = [
    "raw_signal",
    "dwell_log1p",
    "is_A",
    "is_C",
    "is_G",
    "is_T",
    "strand",
    "mapq_norm",
    "matches_ref",
]

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {
    "A": "#2ca25f",
    "C": "#3182bd",
    "G": "#f59e0b",
    "T": "#de2d26",
    "N": "#e5e7eb",
}


def esc(value: object) -> str:
    return html.escape(str(value))


def decode_ref_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def scalar_attr(attrs, name: str, default=None):
    value = attrs.get(name, default)
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def channel_names_from_attrs(attrs, n_channels: int) -> List[str]:
    raw = attrs.get("channel_names")
    if raw is None:
        return DEFAULT_CHANNEL_NAMES[:n_channels]
    names = []
    for item in raw:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8", errors="replace"))
        else:
            names.append(str(item))
    if len(names) < n_channels:
        names.extend(DEFAULT_CHANNEL_NAMES[len(names):n_channels])
    return names[:n_channels]


def choose_index(
    h5,
    *,
    index: Optional[int],
    label: str,
    ref_name: Optional[str],
    ref_pos: Optional[int],
    image_idx: Optional[int],
) -> int:
    n = int(h5["tensors"].shape[0])
    if index is not None:
        if index < 0 or index >= n:
            raise SystemExit(f"--index {index} is out of range for {n} images")
        return int(index)

    candidates = np.arange(n)
    if label != "auto":
        labels = h5["labels"][:].astype(int)
        desired = 1 if label == "modified" else 0
        candidates = candidates[labels == desired]
    elif "labels" in h5:
        labels = h5["labels"][:].astype(int)
        positives = candidates[labels == 1]
        if len(positives) > 0:
            candidates = positives

    if ref_name is not None:
        ref_names = np.array([decode_ref_name(x) for x in h5["ref_names"][:]])
        candidates = candidates[ref_names[candidates] == ref_name]

    if ref_pos is not None:
        ref_pos_values = h5["ref_pos"][:]
        candidates = candidates[ref_pos_values[candidates] == ref_pos]

    if image_idx is not None and "image_idx" in h5:
        image_idx_values = h5["image_idx"][:]
        candidates = candidates[image_idx_values[candidates] == image_idx]

    if len(candidates) == 0:
        raise SystemExit("No H5 image matched the requested filters")

    if "n_reads" in h5:
        n_reads = h5["n_reads"][:]
        max_reads = np.max(n_reads[candidates])
        candidates = candidates[n_reads[candidates] == max_reads]

    if "image_idx" in h5:
        image_idx_values = h5["image_idx"][:]
        min_image_idx = np.min(image_idx_values[candidates])
        candidates = candidates[image_idx_values[candidates] == min_image_idx]

    return int(candidates[0])


def robust_signal_clip(raw: np.ndarray) -> float:
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return 1.0
    lo, hi = np.percentile(finite, [2, 98])
    return max(abs(float(lo)), abs(float(hi)), 1.0)


def infer_w_l(attrs, total_columns: int) -> Tuple[int, int]:
    w = int(scalar_attr(attrs, "W", 0) or 0)
    l = int(scalar_attr(attrs, "L", 0) or 0)
    if w > 0 and l > 0 and w * l == total_columns:
        return w, l
    if w > 0 and total_columns % w == 0:
        return w, total_columns // w
    if l > 0 and total_columns % l == 0:
        return total_columns // l, l
    return total_columns, 1


def base_from_block(block: np.ndarray) -> str:
    if block.size == 0 or block.shape[-1] < 4:
        return "N"
    scores = np.nanmean(block, axis=0)
    if not np.isfinite(scores).any() or float(np.nanmax(scores)) <= 0.1:
        return "N"
    return BASES[int(np.nanargmax(scores))]


def rect(x, y, w, h, fill, stroke="none", stroke_width=0, extra="") -> str:
    stroke_part = "" if stroke == "none" else f' stroke="{stroke}" stroke-width="{stroke_width}"'
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"{stroke_part} {extra}/>'


def text(x, y, value, size=12, weight=400, fill="#111827", anchor="start", extra="") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Inter, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" {extra}>{esc(value)}</text>'
    )


def line(x1, y1, x2, y2, stroke="#111827", stroke_width=1, dash: Optional[str] = None) -> str:
    dash_part = "" if dash is None else f' stroke-dasharray="{dash}"'
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"{dash_part}/>'
    )


def polyline(points: List[Tuple[float, float]], stroke: str, stroke_width: float, opacity: float = 1.0) -> str:
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" opacity="{opacity:.3f}"/>'
    )


def draw_signal_traces(
    raw: np.ndarray,
    x0: float,
    y0: float,
    col_px: float,
    row_px: float,
    clip: float,
) -> List[str]:
    rows, cols = raw.shape
    amp = row_px * 0.38
    out: List[str] = []
    for r in range(rows):
        band_top = y0 + r * row_px
        baseline = band_top + row_px * 0.52
        if r == 0:
            out.append(rect(x0, band_top + 1, cols * col_px, row_px - 2, "#f8fafc"))
            stroke = "#111827"
            width = 1.65
            opacity = 1.0
        else:
            stroke = "#2563eb"
            width = 0.82
            opacity = 0.78
        out.append(line(x0, baseline, x0 + cols * col_px, baseline, stroke="#e5e7eb", stroke_width=0.55))
        points: List[Tuple[float, float]] = []
        for c, value in enumerate(raw[r]):
            if not np.isfinite(value):
                continue
            clipped = max(-clip, min(clip, float(value)))
            x = x0 + (c + 0.5) * col_px
            y = baseline - (clipped / clip) * amp
            points.append((x, y))
        if len(points) >= 2:
            out.append(polyline(points, stroke=stroke, stroke_width=width, opacity=opacity))
    return out


def draw_reference_bases(
    base_channels: np.ndarray,
    x0: float,
    y0: float,
    window_positions: int,
    samples_per_base: int,
    col_px: float,
    height: float,
) -> List[str]:
    out: List[str] = []
    for b in range(window_positions):
        start = b * samples_per_base
        end = start + samples_per_base
        base = base_from_block(base_channels[0, start:end, :])
        x = x0 + start * col_px
        w = samples_per_base * col_px
        fill = BASE_COLORS[base]
        text_fill = "#ffffff" if base != "N" else "#6b7280"
        out.append(rect(x, y0, w, height, fill, stroke="#ffffff", stroke_width=0.7))
        out.append(text(x + w / 2, y0 + height * 0.68, base, size=11, weight=750, fill=text_fill, anchor="middle"))
    return out


def make_figure(args: argparse.Namespace) -> Path:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "h5py is required to read DeepMod feature files. Activate the "
            "DeepMod/rockfish environment or install h5py in this Python."
        ) from exc

    h5_path = args.h5
    out_png = args.output
    if out_png is None:
        out_png = h5_path.with_suffix(".signal_pileup.png")
    elif out_png.suffix.lower() != ".png":
        out_png = out_png.with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        idx = choose_index(
            h5,
            index=args.index,
            label=args.label,
            ref_name=args.ref_name,
            ref_pos=args.ref_pos,
            image_idx=args.image_idx,
        )
        tensor = h5["tensors"][idx].astype(np.float32)
        label = int(h5["labels"][idx]) if "labels" in h5 else -1
        ref_name = decode_ref_name(h5["ref_names"][idx]) if "ref_names" in h5 else "unknown"
        ref_pos = int(h5["ref_pos"][idx]) if "ref_pos" in h5 else -1
        n_reads = int(h5["n_reads"][idx]) if "n_reads" in h5 else tensor.shape[0] - 1
        image_idx = int(h5["image_idx"][idx]) if "image_idx" in h5 else 0
        attrs = dict(h5.attrs)
        channel_names = channel_names_from_attrs(h5.attrs, tensor.shape[-1])

    total_rows, total_cols, n_channels = tensor.shape
    window_positions, samples_per_base = infer_w_l(attrs, total_cols)
    display_rows = total_rows if args.show_padded else min(total_rows, max(1, n_reads + 1))
    tensor = tensor[:display_rows]
    raw = tensor[:, :, 0]
    base_channels = tensor[:, :, 2:6] if n_channels >= 6 else np.zeros((*tensor.shape[:2], 4), dtype=np.float32)

    col_px = float(args.col_px)
    trace_row_px = float(args.row_px)
    trace_w = total_cols * col_px
    trace_h = display_rows * trace_row_px
    left = 150.0
    base_track_y = 102.0
    base_track_h = 24.0
    trace_y = 152.0
    legend_x = left + trace_w + 54.0
    width = int(legend_x + 316)
    height = int(trace_y + trace_h + 112)
    center_base = window_positions // 2
    center_x = left + center_base * samples_per_base * col_px
    center_w = samples_per_base * col_px
    clip = robust_signal_clip(raw)
    label_text = "modified" if label == 1 else "unmodified" if label == 0 else "unknown"

    svg = [  # type: List[str]
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(width / 2, 36, "DeepMod Resampled Current Pileup", size=24, weight=750, anchor="middle"),
        text(width / 2, 62, f"{h5_path.name} | image {idx} | {ref_name}:{ref_pos} | label: {label_text} | reads: {n_reads}", size=13, fill="#4b5563", anchor="middle"),
        text(left, base_track_y - 10, "Reference bases", size=12, weight=750, fill="#374151"),
        # text(left, trace_y - 14, "Channel 0: expected current reference row + resampled read signal rows", size=14, weight=750),
    ]

    svg.extend(draw_reference_bases(base_channels, left, base_track_y, window_positions, samples_per_base, col_px, base_track_h))
    svg.append(rect(center_x, base_track_y - 1, center_w, base_track_h + 2, "none", stroke="#111827", stroke_width=1.4))
    svg.append(text(center_x + center_w / 2, base_track_y - 8, "candidate site", size=11, fill="#111827", anchor="middle"))

    svg.append(rect(left, trace_y, trace_w, trace_h, "#ffffff", stroke="#111827", stroke_width=1))
    svg.append(rect(center_x, trace_y, center_w, trace_h, "#fef3c7", stroke="none", stroke_width=0, extra='opacity="0.55"'))
    svg.extend(draw_signal_traces(raw, left, trace_y, col_px, trace_row_px, clip))
    svg.append(rect(center_x, trace_y, center_w, trace_h, "none", stroke="#111827", stroke_width=1.4))
    # svg.append(text(center_x + center_w / 2, trace_y - 6, "candidate base", size=11, fill="#111827", anchor="middle"))

    svg.append(text(left - 18, trace_y + trace_row_px * 0.70, "Reference", size=11, fill="#374151", anchor="end"))
    svg.append(text(left - 18, trace_y + trace_row_px * (display_rows + 1) / 2, "Reads", size=11, fill="#374151", anchor="end"))
    svg.append(line(left - 8, trace_y + trace_row_px * 1.05, left - 8, trace_y + trace_h - 2, stroke="#9ca3af", stroke_width=1))

    for b in range(window_positions + 1):
        x = left + b * samples_per_base * col_px
        stroke = "#64748b" if b == center_base or b == center_base + 1 else "#cbd5e1"
        stroke_width = 1.0 if b == center_base or b == center_base + 1 else 0.8
        svg.append(line(x, base_track_y, x, base_track_y + base_track_h, stroke="#ffffff", stroke_width=0.7))
        svg.append(line(x, trace_y, x, trace_y + trace_h, stroke=stroke, stroke_width=stroke_width))

    tick_y = trace_y + trace_h + 24
    axis_y = trace_y + trace_h + 8
    svg.append(line(left, axis_y, left + trace_w, axis_y, stroke="#374151", stroke_width=0.8))
    for rel in [-10, -5, 0, 5, 10]:
        b = center_base + rel
        if 0 <= b < window_positions:
            x = left + (b + 0.5) * samples_per_base * col_px
            svg.append(line(x, axis_y, x, axis_y + 5, stroke="#374151", stroke_width=0.8))
            label = "0" if rel == 0 else f"{rel:+d}"
            svg.append(text(x, tick_y, label, size=10, fill="#4b5563", anchor="middle"))
    svg.append(text(left + trace_w / 2, tick_y + 21, "Reference window", size=12, fill="#4b5563", anchor="middle"))

    # Trace legend.
    legend_y = trace_y + 6
    svg.append(text(legend_x, legend_y, "Signal traces", size=12, weight=700, fill="#374151"))
    svg.append(line(legend_x, legend_y + 22, legend_x + 54, legend_y + 22, stroke="#111827", stroke_width=1.8))
    svg.append(text(legend_x + 66, legend_y + 26, "Reference expected current", size=11, fill="#374151"))
    svg.append(line(legend_x, legend_y + 46, legend_x + 54, legend_y + 46, stroke="#2563eb", stroke_width=1.2))
    svg.append(text(legend_x + 66, legend_y + 50, "Read resampled current", size=11, fill="#374151"))
    svg.append(rect(legend_x, legend_y + 64, 54, 16, "#fef3c7", stroke="#111827", stroke_width=0.8, extra='opacity="0.7"'))
    svg.append(text(legend_x + 66, legend_y + 77, "Candidate base window", size=11, fill="#374151"))
    # svg.append(text(legend_x, legend_y + 108, f"Trace scale: +/-{clip:.1f}", size=11, fill="#6b7280"))

    # Channel legend.
    chan_y = legend_y + 148
    svg.append(text(legend_x, chan_y, "Tensor channels", size=12, weight=700, fill="#374151"))
    descriptions = {
        "raw_signal": "current / expected level",
        "dwell_log1p": "event dwell time",
        "is_A": "base one-hot",
        "is_C": "base one-hot",
        "is_G": "base one-hot",
        "is_T": "base one-hot",
        "strand": "read strand",
        "mapq_norm": "mapping quality",
        "matches_ref": "base matches reference",
    }
    for i, name in enumerate(channel_names):
        y = chan_y + 22 + i * 18
        svg.append(text(legend_x, y, f"{i}. {name}", size=11, weight=650 if i in (0, 1, 6, 7, 8) else 500, fill="#111827"))
        svg.append(text(legend_x + 118, y, descriptions.get(name, ""), size=10, fill="#6b7280"))

    note_y = height - 34
    svg.append(text(left, note_y, f"Rows: reference + {display_rows - 1} displayed reads. Columns: {window_positions} reference bases x {samples_per_base} signal samples/base.", size=12, fill="#4b5563"))
    if display_rows < total_rows:
        svg.append(text(left, note_y + 18, f"Padded rows omitted from view ({total_rows - display_rows} hidden).", size=11, fill="#6b7280"))

    svg.append("</svg>")

    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise SystemExit("rsvg-convert is required to write PNG output, but it was not found on PATH.")

    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False, dir=str(out_png.parent)) as handle:
        tmp_svg = Path(handle.name)
        handle.write("\n".join(svg) + "\n")
    try:
        subprocess.run([converter, "-o", str(out_png), str(tmp_svg)], check=True)
    finally:
        tmp_svg.unlink(missing_ok=True)
    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True, help="DeepMod featurized HDF5 file.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--index", type=int, default=None, help="Exact image index to visualize.")
    parser.add_argument("--label", choices=["auto", "modified", "unmodified"], default="auto", help="Image label to select when --index is not set.")
    parser.add_argument("--ref-name", default=None, help="Reference name filter.")
    parser.add_argument("--ref-pos", type=int, default=None, help="Reference position filter.")
    parser.add_argument("--image-idx", type=int, default=None, help="Image chunk index within a reference position.")
    parser.add_argument("--show-padded", action="store_true", help="Show zero-padded read rows instead of omitting them.")
    parser.add_argument("--col-px", type=float, default=3.0, help="SVG pixels per signal sample.")
    parser.add_argument("--row-px", type=float, default=14.0, help="SVG pixels per signal/read trace row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path = make_figure(args)
    print(f"Wrote PNG: {png_path}")


if __name__ == "__main__":
    main()
