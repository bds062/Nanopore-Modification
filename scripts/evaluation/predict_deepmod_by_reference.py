#!/usr/bin/env python3
"""Run a Deep Modification checkpoint on an HDF5 feature file.

The output is a per-reference-base TSV with the same `gt_label` + `mean_prob`
shape used by the Rockfish reference-level tables, so both methods can be
plotted by the same comparison script.
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
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.001),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InceptionA(nn.Module):
    def __init__(self, in_ch: int, pool_proj: int):
        super().__init__()
        self.b1 = ConvBnRelu(in_ch, 16, 1)
        self.b2 = nn.Sequential(ConvBnRelu(in_ch, 12, 1), ConvBnRelu(12, 16, 5, padding=2))
        self.b3 = nn.Sequential(
            ConvBnRelu(in_ch, 16, 1),
            ConvBnRelu(16, 24, 3, padding=1),
            ConvBnRelu(24, 24, 3, padding=1),
        )
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), ConvBnRelu(in_ch, pool_proj, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1 = ConvBnRelu(in_ch, 96, 3, stride=2, padding=1)
        self.b2 = nn.Sequential(
            ConvBnRelu(in_ch, 16, 1),
            ConvBnRelu(16, 24, 3, padding=1),
            ConvBnRelu(24, 24, 3, stride=2, padding=1),
        )
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_ch: int, channels_7x7: int):
        super().__init__()
        c7 = channels_7x7
        self.b1 = ConvBnRelu(in_ch, 48, 1)
        self.b2 = nn.Sequential(
            ConvBnRelu(in_ch, c7, 1),
            ConvBnRelu(c7, c7, (1, 7), padding=(0, 3)),
            ConvBnRelu(c7, 48, (7, 1), padding=(3, 0)),
        )
        self.b3 = nn.Sequential(
            ConvBnRelu(in_ch, c7, 1),
            ConvBnRelu(c7, c7, (7, 1), padding=(3, 0)),
            ConvBnRelu(c7, c7, (1, 7), padding=(0, 3)),
            ConvBnRelu(c7, c7, (7, 1), padding=(3, 0)),
            ConvBnRelu(c7, 48, (1, 7), padding=(0, 3)),
        )
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), ConvBnRelu(in_ch, 48, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionD(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1 = nn.Sequential(ConvBnRelu(in_ch, 48, 1), ConvBnRelu(48, 80, 3, stride=2, padding=1))
        self.b2 = nn.Sequential(
            ConvBnRelu(in_ch, 48, 1),
            ConvBnRelu(48, 48, (1, 7), padding=(0, 3)),
            ConvBnRelu(48, 48, (7, 1), padding=(3, 0)),
            ConvBnRelu(48, 48, 3, stride=2, padding=1),
        )
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class InceptionE(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1 = ConvBnRelu(in_ch, 80, 1)
        self.b2_0 = ConvBnRelu(in_ch, 96, 1)
        self.b2_1a = ConvBnRelu(96, 96, (1, 3), padding=(0, 1))
        self.b2_1b = ConvBnRelu(96, 96, (3, 1), padding=(1, 0))
        self.b3_0 = nn.Sequential(ConvBnRelu(in_ch, 112, 1), ConvBnRelu(112, 96, 3, padding=1))
        self.b3_1a = ConvBnRelu(96, 96, (1, 3), padding=(0, 1))
        self.b3_1b = ConvBnRelu(96, 96, (3, 1), padding=(1, 0))
        self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), ConvBnRelu(in_ch, 48, 1))

    def forward(self, x):
        b1 = self.b1(x)
        b2_ = self.b2_0(x)
        b2 = torch.cat([self.b2_1a(b2_), self.b2_1b(b2_)], dim=1)
        b3_ = self.b3_0(x)
        b3 = torch.cat([self.b3_1a(b3_), self.b3_1b(b3_)], dim=1)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class PileupInceptionV3(nn.Module):
    def __init__(self, in_channels: int = 9, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnRelu(in_channels, 32, 3, stride=2, padding=1),
            ConvBnRelu(32, 32, 3, padding=1),
            ConvBnRelu(32, 64, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBnRelu(64, 48, 1),
            ConvBnRelu(48, 64, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.inceptionA1 = InceptionA(64, pool_proj=8)
        self.inceptionA2 = InceptionA(64, pool_proj=16)
        self.inceptionA3 = InceptionA(72, pool_proj=16)
        self.inceptionB = InceptionB(72)
        self.inceptionC1 = InceptionC(192, channels_7x7=32)
        self.inceptionC2 = InceptionC(192, channels_7x7=40)
        self.inceptionC3 = InceptionC(192, channels_7x7=40)
        self.inceptionC4 = InceptionC(192, channels_7x7=48)
        self.inceptionD = InceptionD(192)
        self.inceptionE1 = InceptionE(320)
        self.inceptionE2 = InceptionE(512)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.inceptionA1(x)
        x = self.inceptionA2(x)
        x = self.inceptionA3(x)
        x = self.inceptionB(x)
        x = self.inceptionC1(x)
        x = self.inceptionC2(x)
        x = self.inceptionC3(x)
        x = self.inceptionC4(x)
        x = self.inceptionD(x)
        x = self.inceptionE1(x)
        x = self.inceptionE2(x)
        return self.head(x)


@dataclass
class BaseAgg:
    label: int = 0
    n_images: int = 0
    sum_prob: float = 0.0
    n_mod_images: int = 0
    min_prob: float = math.inf
    max_prob: float = -math.inf

    def add(self, label: int, prob: float, threshold: float) -> None:
        self.label = max(self.label, int(label > 0))
        self.n_images += 1
        self.sum_prob += prob
        self.n_mod_images += int(prob > threshold)
        self.min_prob = min(self.min_prob, prob)
        self.max_prob = max(self.max_prob, prob)

    @property
    def mean_prob(self) -> float:
        return self.sum_prob / max(self.n_images, 1)

    @property
    def frac_mod(self) -> float:
        return self.n_mod_images / max(self.n_images, 1)


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
    return {name: "".join(parts) for name, parts in seqs.items()}


def decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def fmt(value: float) -> str:
    return f"{value:.8g}"


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def load_model(path: Path, device: torch.device, dropout: float) -> PileupInceptionV3:
    ckpt = torch.load(str(path), map_location="cpu")
    in_channels = int(ckpt.get("in_channels", 9))
    model = PileupInceptionV3(in_channels=in_channels, dropout=dropout)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_h5(
    h5_path: Path,
    model: nn.Module,
    device: torch.device,
    dataset: str,
    output: Path,
    reference: Optional[Path],
    batch_size: int,
    threshold: float,
    limit_images: Optional[int],
    progress_every: int,
) -> None:
    fasta = load_fasta(reference)
    agg: Dict[Tuple[str, int], BaseAgg] = defaultdict(BaseAgg)

    with h5py.File(str(h5_path), "r") as hf:
        n_total = int(hf["tensors"].shape[0])
        n = min(n_total, limit_images) if limit_images is not None else n_total
        print(f"[{dataset}] H5 images={n_total:,}; evaluating={n:,}", file=sys.stderr)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = hf["tensors"][start:end].astype(np.float32)
            x = np.transpose(x, (0, 3, 1, 2))
            xb = torch.from_numpy(x).to(device, non_blocking=True)
            probs = torch.sigmoid(model(xb).squeeze(1)).cpu().numpy()

            labels = hf["labels"][start:end]
            ref_names = hf["ref_names"][start:end]
            ref_pos = hf["ref_pos"][start:end]
            for label, prob, ref_name, pos in zip(labels, probs, ref_names, ref_pos):
                key = (decode_name(ref_name), int(pos))
                agg[key].add(int(label), float(prob), threshold)

            if progress_every > 0 and (end == n or end % progress_every == 0):
                print(f"[{dataset}] evaluated {end:,}/{n:,} images", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "ref_name",
                "ref_pos",
                "start",
                "end",
                "ref_base",
                "gt_label",
                "n_images",
                "mean_prob",
                "frac_mod",
                "n_mod_images",
                "n_unmod_images",
                "min_prob",
                "max_prob",
            ]
        )
        for ref_name, pos in sorted(agg, key=lambda k: (k[0], k[1])):
            item = agg[(ref_name, pos)]
            seq = fasta.get(ref_name, "")
            ref_base = seq[pos] if pos < len(seq) else "N"
            writer.writerow(
                [
                    dataset,
                    ref_name,
                    pos,
                    pos,
                    pos + 1,
                    ref_base,
                    item.label,
                    item.n_images,
                    fmt(item.mean_prob),
                    fmt(item.frac_mod),
                    item.n_mod_images,
                    item.n_images - item.n_mod_images,
                    fmt(item.min_prob),
                    fmt(item.max_prob),
                ]
            )

    print(f"[{dataset}] wrote {len(agg):,} reference-base predictions: {output}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"[{args.dataset}] loading model on {device}: {args.model}", file=sys.stderr)
    model = load_model(args.model, device, dropout=args.dropout)
    predict_h5(
        h5_path=args.h5,
        model=model,
        device=device,
        dataset=args.dataset,
        output=args.output,
        reference=args.reference,
        batch_size=args.batch_size,
        threshold=args.threshold,
        limit_images=args.limit_images,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
