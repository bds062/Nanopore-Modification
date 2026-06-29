#!/usr/bin/env python3
"""
deep_mod_model.py
=================
Lightweight InceptionV3-style CNN for nanopore base modification detection.

Architecture
------------
Scaled-down InceptionV3 (~1.4M parameters).  Channel widths are reduced to
~25% of the original torchvision InceptionV3 throughout, while the block
topology is preserved so the model still captures multi-scale spatial patterns
in the pileup image.

  Input: (B, 9, H, W)
    H = max_reads + 1   (row 0 = reference track, rows 1..H = reads)
    W = window_positions * L_samples

  Stem → InceptionA ×3 → InceptionB → InceptionC ×4
       → InceptionD → InceptionE ×2 → GAP → Dropout → Linear(1)

  Loss      : BCE with pos_weight (default) or focal loss (--focal).
              BCE + pos_weight is preferred for large datasets (>10K positives)
              where focal's hard-example focusing is less critical.
              Focal is useful for very small positive counts (<1K) where
              easy negatives would otherwise dominate the gradient.
  Scheduler : ReduceLROnPlateau(mode='max', patience=7) on val AUPRC
  Metric    : AUPRC (primary), AUROC (secondary)

HDF5 worker safety
------------------
h5py file handles are NOT safe to share across multiprocessing workers.
Each DataLoader worker opens its own per-process file handle cache via a
worker_init_fn, avoiding deadlocks that occur with num_workers > 0.

Checkpoint resumption
---------------------
By default (resume is ON unless --no-resume is passed) the script will skip any
stage whose sentinel file already exists in --out-dir, and continue unfinished
training loops from their last completed epoch when a state checkpoint exists:

  Stage                  Sentinel file
  ─────────────────────  ──────────────────────────────────
  Main training + eval   test_predictions.npz
  Channel importance     channel_importance.npz
  LODO fold <stem>       lodo_<stem>_result.npz

  In-progress training   training_state.pt
  In-progress LODO fold  lodo_<stem>_state.pt

This lets you re-run the exact same command after a crash and only the
unfinished stages will execute.  Use --no-resume to force everything to
rerun from scratch.

Usage
-----
  python deep_mod_model.py \\
      --input sample_a.h5 [sample_b.h5 ...] \\
      --out-dir results/

  # After a crash, just re-run the same command — finished stages are skipped:
  python deep_mod_model.py \\
      --input sample_a.h5 [sample_b.h5 ...] \\
      --out-dir results/

  # Force everything to rerun:
  python deep_mod_model.py \\
      --input sample_a.h5 [sample_b.h5 ...] \\
      --out-dir results/ --no-resume
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score,
)

try:
    from . import lodo as loo_mod
    from .visualization import (
        CHANNEL_NAMES,
        evaluate,
        plot_channel_importance,
        plot_confusion_matrix,
        plot_loo_results,
        plot_precision_recall,
        plot_training_curves,
        save_loo_metrics_tsv,
        save_training_history,
    )
except ImportError:  # support direct execution from this directory
    import lodo as loo_mod
    from visualization import (
        CHANNEL_NAMES,
        evaluate,
        plot_channel_importance,
        plot_confusion_matrix,
        plot_loo_results,
        plot_precision_recall,
        plot_training_curves,
        save_loo_metrics_tsv,
        save_training_history,
    )


# ── reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── focal loss ────────────────────────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    """
    Sigmoid focal loss for binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Use when positive count is very small (<1K) and easy true-negatives
    dominate the gradient.  For larger datasets, plain BCE + pos_weight is
    simpler and equally effective.

    pos_weight is applied as alpha scaling on positive terms only.
    Label smoothing is intentionally omitted — it interacts badly with
    focal loss because soft targets break the p_t computation.
    """
    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma      = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')

        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)

        # Alpha: scale positive terms by pos_weight, negative terms by 1,
        # then normalise so the mean weight stays ~1 (prevents loss scale shift
        # when pos_weight is large).
        alpha_t = self.pos_weight * targets + 1.0 * (1 - targets)
        alpha_t = alpha_t / alpha_t.mean()

        focal_w = (1 - p_t) ** self.gamma
        return (alpha_t * focal_w * bce_loss).mean()


# ── HDF5 worker-safe file handle cache ───────────────────────────────────────
#
# h5py file objects are not safe to share across multiprocessing workers.
# We keep a per-process dict of open handles; worker_init_fn resets the
# dict in each worker so every worker opens its own handles on first access.

_H5_HANDLES: dict = {}


def _worker_init_fn(worker_id: int):
    """Called once per DataLoader worker at startup — resets the handle cache."""
    global _H5_HANDLES
    _H5_HANDLES = {}


def _get_h5(path: str) -> h5py.File:
    """Return a cached open h5py.File for this process, opening if necessary."""
    if path not in _H5_HANDLES:
        _H5_HANDLES[path] = h5py.File(path, 'r')
    return _H5_HANDLES[path]


# ── dataset ───────────────────────────────────────────────────────────────────

class PileupDataset(Dataset):
    """
    Loads pileup tensors from one or more HDF5 files produced by
    featurize_pileup.py.

    Tensor on disk : (H, W, C)  where H = max_reads+1, C = 9
    Returned       : (C, H, W)  channels-first for PyTorch Conv2d

    When delta_channels=True (default), two extra channels are appended at
    load time without any re-featurization:
      Ch 9  read_supports_modification : per-read center-position signal
                deviation from expected kmer level, broadcast across the
                full row (analogous to DeepVariant's read_supports_variant).
      Ch 10 window_delta               : per-sample full-window deviation
                (observed signal minus expected kmer level) for all positions.

    HDF5 file handles are cached per-process (see _get_h5) so that
    multiprocessing DataLoader workers do not share handles and deadlock.
    """
    def __init__(self, h5_paths: list, indices: np.ndarray,
                 file_sizes: np.ndarray,
                 augment: bool = False, seed: int = 42,
                 rc_augment: bool = False,
                 signal_noise_std: float = 0.05,
                 delta_channels: bool = True):
        self.h5_paths   = h5_paths
        self.indices    = indices
        self.file_sizes = file_sizes
        self.augment    = augment
        self.rc_augment = rc_augment
        self.signal_noise_std = signal_noise_std
        self.delta_channels   = delta_channels
        self.rng        = np.random.default_rng(seed)
        self.offsets    = np.concatenate([[0], np.cumsum(file_sizes)])

        with h5py.File(h5_paths[0], 'r') as hf:
            self.n_channels       = int(hf.attrs.get('n_channels', 9))
            self.window_positions = int(hf.attrs.get('W', 0))
            self.samples_per_base = int(hf.attrs.get('L', 0))
            # center_idx: which window position is the candidate base (0-based)
            self.center_idx = int(hf.attrs.get(
                'center_idx', self.window_positions // 2))

    def _resolve(self, global_idx: int) -> tuple:
        file_idx  = int(np.searchsorted(self.offsets[1:], global_idx, side='right'))
        local_idx = global_idx - int(self.offsets[file_idx])
        return file_idx, local_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        global_idx          = int(self.indices[item])
        file_idx, local_idx = self._resolve(global_idx)

        # Use per-process cached handle — safe for multiprocessing workers
        hf = _get_h5(self.h5_paths[file_idx])
        x  = hf['tensors'][local_idx].astype(np.float32)   # (H, W, C)
        y  = float(hf['labels'][local_idx])

        x = np.transpose(x, (2, 0, 1))   # → (C, H, W)

        if self.augment:
            H = x.shape[1]
            # Zero out up to 30% of read rows (row 0 = reference, never masked)
            n_mask = int((H - 1) * 0.30 * self.rng.random())
            if n_mask > 0:
                rows = self.rng.choice(np.arange(1, H), size=n_mask, replace=False)
                x[:, rows, :] = 0.0
            # Additive noise only on raw signal.  Noisy one-hot/metadata
            # channels are not meaningful features.
            if self.signal_noise_std > 0:
                x[0] += (self.rng.standard_normal(x[0].shape).astype(np.float32)
                         * self.signal_noise_std)
            if self.rc_augment and self.rng.random() < 0.5:
                x = reverse_complement_tensor(
                    x, self.window_positions, self.samples_per_base)

        # Append delta channels after augmentation so they are consistent with
        # the (potentially noised) raw signal channel the model sees.
        if self.delta_channels and self.samples_per_base > 0:
            L  = self.samples_per_base
            ci = self.center_idx
            cs, ce = ci * L, ci * L + L          # center base column slice

            # Ch 9: per-read center-position delta broadcast across the full
            # row — the deepmod analogue of DeepVariant's read_supports_variant.
            # x[0, 0, cs:ce] = expected kmer level at center (reference row).
            # x[0, 1:, cs:ce] = observed signal at center for each read.
            ref_ctr  = float(x[0, 0, cs:ce].mean())
            read_ctr = x[0, 1:, cs:ce].mean(axis=1)     # (H-1,)
            ch9 = np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float32)
            ch9[0, 1:, :] = (read_ctr - ref_ctr)[:, np.newaxis]

            # Ch 10: full-window per-sample delta (observed minus expected).
            ch10 = np.zeros((1, x.shape[1], x.shape[2]), dtype=np.float32)
            ch10[0, 1:] = x[0, 1:] - x[0, 0]

            x = np.concatenate([x, ch9, ch10], axis=0)  # (C+2, H, W)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def reverse_complement_tensor(x: np.ndarray,
                              window_positions: int,
                              samples_per_base: int) -> np.ndarray:
    """
    Reverse-complement a channels-first pileup tensor.

    This reverses position/sample order, swaps A/T and C/G one-hot channels,
    and flips strand sign.  The reference-row expected-level channel is simply
    reversed; if the pore model is strand-specific, prefer leaving
    --rc-augment disabled.
    """
    c, h, total_w = x.shape
    y = x
    if window_positions > 0 and samples_per_base > 0:
        expected_w = window_positions * samples_per_base
        if expected_w == total_w:
            y = x.reshape(c, h, window_positions, samples_per_base)
            y = y[:, :, ::-1, ::-1].reshape(c, h, total_w).copy()
        else:
            y = x[:, :, ::-1].copy()
    else:
        y = x[:, :, ::-1].copy()

    if c >= 6:
        bases = y[2:6].copy()
        y[2] = bases[3]  # A <- T
        y[3] = bases[2]  # C <- G
        y[4] = bases[1]  # G <- C
        y[5] = bases[0]  # T <- A
    if c > 6:
        y[6] *= -1.0
    return y


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Mixup augmentation: interpolate pairs of samples within a batch.
    alpha controls the Beta distribution shape; 0 disables mixup.
    Note: do not use mixup with focal loss — soft targets break p_t.
    """
    lam        = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx        = torch.randperm(x.size(0), device=x.device)
    x_mix      = lam * x + (1 - lam) * x[idx]
    y_mix      = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix


# ── Inception building blocks (scaled-down channel widths) ───────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.001),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class InceptionA(nn.Module):
    """
    4-branch Inception module (InceptionV3 Figure 4), ~25% channel width.
    Output: 16 + 16 + 24 + pool_proj
    """
    def __init__(self, in_ch: int, pool_proj: int):
        super().__init__()
        self.b1 = ConvBnRelu(in_ch, 16, 1)
        self.b2 = nn.Sequential(
            ConvBnRelu(in_ch, 12, 1),
            ConvBnRelu(12, 16, 5, padding=2),
        )
        self.b3 = nn.Sequential(
            ConvBnRelu(in_ch, 16, 1),
            ConvBnRelu(16, 24, 3, padding=1),
            ConvBnRelu(24, 24, 3, padding=1),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBnRelu(in_ch, pool_proj, 1),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionB(nn.Module):
    """
    Grid-reduction (InceptionV3 Figure 10), ~25% channel width.
    Output: 96 + 24 + in_ch
    """
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
    """
    Factorised convolution (InceptionV3 Figure 6), ~25% channel width.
    Output: 48 × 4 = 192
    """
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
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBnRelu(in_ch, 48, 1),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class InceptionD(nn.Module):
    """
    Second grid-reduction (InceptionV3), ~25% channel width.
    Output: 80 + 48 + in_ch
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1 = nn.Sequential(
            ConvBnRelu(in_ch, 48, 1),
            ConvBnRelu(48, 80, 3, stride=2, padding=1),
        )
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
    """
    Expanded parallel branches (InceptionV3 Figure 7), ~25% channel width.
    Output: 80 + 192 + 192 + 48 = 512
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1    = ConvBnRelu(in_ch, 80, 1)
        self.b2_0  = ConvBnRelu(in_ch, 96, 1)
        self.b2_1a = ConvBnRelu(96, 96, (1, 3), padding=(0, 1))
        self.b2_1b = ConvBnRelu(96, 96, (3, 1), padding=(1, 0))
        self.b3_0  = nn.Sequential(
            ConvBnRelu(in_ch, 112, 1),
            ConvBnRelu(112, 96, 3, padding=1),
        )
        self.b3_1a = ConvBnRelu(96, 96, (1, 3), padding=(0, 1))
        self.b3_1b = ConvBnRelu(96, 96, (3, 1), padding=(1, 0))
        self.b4    = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBnRelu(in_ch, 48, 1),
        )

    def forward(self, x):
        b1  = self.b1(x)
        b2_ = self.b2_0(x)
        b2  = torch.cat([self.b2_1a(b2_), self.b2_1b(b2_)], dim=1)
        b3_ = self.b3_0(x)
        b3  = torch.cat([self.b3_1a(b3_), self.b3_1b(b3_)], dim=1)
        b4  = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# ── full model ────────────────────────────────────────────────────────────────

class PileupInceptionV3(nn.Module):
    """
    Scaled-down InceptionV3 for pileup classification (~1.4M parameters).

    Input : (B, 9, H, W)
    Output: (B, 1) logit
    """
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

        self.inceptionA1 = InceptionA(64,  pool_proj=8)
        self.inceptionA2 = InceptionA(64,  pool_proj=16)
        self.inceptionA3 = InceptionA(72,  pool_proj=16)
        self.inceptionB  = InceptionB(72)
        self.inceptionC1 = InceptionC(192, channels_7x7=32)
        self.inceptionC2 = InceptionC(192, channels_7x7=40)
        self.inceptionC3 = InceptionC(192, channels_7x7=40)
        self.inceptionC4 = InceptionC(192, channels_7x7=48)
        self.inceptionD  = InceptionD(192)
        self.inceptionE1 = InceptionE(320)
        self.inceptionE2 = InceptionE(512)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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


PileupCNN = PileupInceptionV3   # alias for loo.py compatibility


# ── training helpers ──────────────────────────────────────────────────────────

def get_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)


def make_sampler(labels: np.ndarray,
                 num_samples: int | None = None) -> WeightedRandomSampler:
    n_pos   = int(labels.sum())
    n_neg   = len(labels) - n_pos
    w_pos   = 1.0 / max(n_pos, 1)
    w_neg   = 1.0 / max(n_neg, 1)
    weights = np.where(labels == 1, w_pos, w_neg)
    if num_samples is None:
        num_samples = len(weights)
    return WeightedRandomSampler(
        torch.from_numpy(weights).float(),
        num_samples=int(num_samples),
        replacement=True,
    )


def make_loader_kwargs(batch_size: int,
                       num_workers: int,
                       device: torch.device,
                       worker_init_fn=None) -> dict:
    """
    Build DataLoader kwargs that are conservative on shared/HPC systems.

    Multiprocessing DataLoader workers require PyTorch shared-memory/resource
    sharing support.  Some batch/sandbox environments block that path, which
    makes training appear to hang before the first batch.  The default
    num_workers=0 avoids that failure mode; users can opt into workers when
    their runtime supports it.
    """
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    if num_workers > 0:
        kwargs['worker_init_fn'] = worker_init_fn
        kwargs['persistent_workers'] = True
    return kwargs


def binary_labels(labels: np.ndarray) -> np.ndarray:
    """Coerce any non-zero modification label to the binary modified class."""
    return (labels.astype(np.int64) > 0).astype(np.int64)


def decode_ref_names(values: np.ndarray) -> list[str]:
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode('utf-8'))
        else:
            names.append(str(value))
    return names


def make_position_keys(ref_names: np.ndarray, ref_pos: np.ndarray) -> list[tuple]:
    names = decode_ref_names(ref_names)
    return list(zip(names, ref_pos.astype(np.int64).tolist()))


def _split_group_keys(group_to_label: dict,
                      val_frac: float,
                      test_frac: float,
                      seed: int):
    """Stratify a list of leakage-safe groups into train/val/test keys."""
    by_label = {
        0: [k for k, v in group_to_label.items() if v == 0],
        1: [k for k, v in group_to_label.items() if v == 1],
    }
    rng = np.random.default_rng(seed)

    def split_keys(keys: list):
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        if n == 0:
            return [], [], []
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if test_frac > 0 and n_test == 0 and n >= 3:
            n_test = 1
        if val_frac > 0 and n_val == 0 and n - n_test >= 2:
            n_val = 1
        while n_test + n_val >= n and n_test + n_val > 0:
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            else:
                n_test -= 1
        test_keys = keys[:n_test]
        val_keys = keys[n_test:n_test + n_val]
        train_keys = keys[n_test + n_val:]
        return train_keys, val_keys, test_keys

    train_keys, val_keys, test_keys = [], [], []
    for label in (0, 1):
        tr, va, te = split_keys(by_label[label])
        train_keys.extend(tr)
        val_keys.extend(va)
        test_keys.extend(te)

    return train_keys, val_keys, test_keys


def split_position_groups(labels: np.ndarray,
                          position_keys: list[tuple],
                          val_frac: float,
                          test_frac: float,
                          seed: int):
    """
    Stratify train/val/test by unique genomic position, then expand to images.

    All images with the same (ref_name, ref_pos) stay in one split.  If the
    same coordinate appears in multiple input files, those images stay together
    too, which prevents sequence/context leakage across splits.
    """
    group_to_indices: dict[tuple, list[int]] = {}
    group_to_label: dict[tuple, int] = {}
    for i, key in enumerate(position_keys):
        group_to_indices.setdefault(key, []).append(i)
        group_to_label[key] = max(group_to_label.get(key, 0), int(labels[i] > 0))

    train_keys, val_keys, test_keys = _split_group_keys(
        group_to_label=group_to_label,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    def expand(keys: list[tuple]) -> np.ndarray:
        idx = []
        for key in keys:
            idx.extend(group_to_indices[key])
        return np.array(idx, dtype=np.int64)

    stats = {
        'train_positions': len(train_keys),
        'val_positions':   len(val_keys),
        'test_positions':  len(test_keys),
    }
    return expand(train_keys), expand(val_keys), expand(test_keys), stats


def split_contig_groups(labels: np.ndarray,
                        position_keys: list[tuple],
                        val_frac: float,
                        test_frac: float,
                        seed: int):
    """
    Stratify train/val/test by reference contig, then expand to images.

    This is stricter than the position-level split: every image from a contig
    is assigned to exactly one split across all input HDF5 files.  It is useful
    when the desired evaluation is generalization to unseen contigs rather than
    unseen individual reference bases.
    """
    group_to_indices: dict[str, list[int]] = {}
    group_to_label: dict[str, int] = {}
    for i, key in enumerate(position_keys):
        contig = str(key[0])
        group_to_indices.setdefault(contig, []).append(i)
        group_to_label[contig] = max(
            group_to_label.get(contig, 0),
            int(labels[i] > 0),
        )

    train_keys, val_keys, test_keys = _split_group_keys(
        group_to_label=group_to_label,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )

    def expand(keys: list[str]) -> np.ndarray:
        idx = []
        for key in keys:
            idx.extend(group_to_indices[key])
        return np.array(idx, dtype=np.int64)

    train_idx = expand(train_keys)
    val_idx = expand(val_keys)
    test_idx = expand(test_keys)

    def n_positions(indices: np.ndarray) -> int:
        return len({position_keys[int(i)] for i in indices})

    stats = {
        'train_contigs': len(train_keys),
        'val_contigs':   len(val_keys),
        'test_contigs':  len(test_keys),
        'train_positions': n_positions(train_idx),
        'val_positions':   n_positions(val_idx),
        'test_positions':  n_positions(test_idx),
    }
    return train_idx, val_idx, test_idx, stats


def source_position_keys(global_indices: np.ndarray,
                         position_keys: list[tuple],
                         file_sizes: np.ndarray):
    """
    Return source-aware base keys for image indices.

    Splitting uses bare (ref_name, ref_pos) keys so the same reference base is
    never shared across train/val/test.  Scoring uses (file_idx, ref_name,
    ref_pos), because the same reference coordinate can appear as unmodified in
    control and modified in treatment files.
    """
    offsets = np.concatenate([[0], np.cumsum(file_sizes)])
    file_idx = np.searchsorted(offsets[1:], global_indices, side='right')
    keys = [
        (int(fi), position_keys[int(i)][0], int(position_keys[int(i)][1]))
        for i, fi in zip(global_indices, file_idx)
    ]
    return keys, file_idx.astype(np.int64)


def aggregate_by_position(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          group_keys: list[tuple]):
    """Average image probabilities into one probability per supplied group."""
    grouped: dict[tuple, list] = {}
    for yt, yp, key in zip(y_true, y_prob, group_keys):
        if key not in grouped:
            grouped[key] = [0, []]
        grouped[key][0] = max(grouped[key][0], int(yt > 0))
        grouped[key][1].append(float(yp))

    keys, labels, probs = [], [], []
    for key, (label, values) in grouped.items():
        keys.append(key)
        labels.append(label)
        probs.append(float(np.mean(values)))
    return (np.array(labels, dtype=np.int64),
            np.array(probs, dtype=np.float32),
            keys)


@torch.no_grad()
def run_inference(model: nn.Module, loader: DataLoader,
                  device: torch.device):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x      = x.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_y.append(y.numpy())
        all_p.append(probs)
    return np.concatenate(all_y), np.concatenate(all_p)


@torch.no_grad()
def run_inference_and_loss(model: nn.Module, loader: DataLoader,
                           criterion: nn.Module, device: torch.device):
    model.eval()
    loss_sum = 0.0
    n_seen = 0
    all_y, all_p = [], []
    for x, y in loader:
        y_cpu = y.numpy()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x).squeeze(1)
        loss_sum += criterion(logits, y).item() * len(y)
        n_seen += len(y)
        all_y.append(y_cpu)
        all_p.append(torch.sigmoid(logits).cpu().numpy())
    return (loss_sum / max(n_seen, 1),
            np.concatenate(all_y),
            np.concatenate(all_p))


# ── multi-file helpers ────────────────────────────────────────────────────────

def load_and_validate_files(h5_paths: list):
    if not h5_paths:
        raise ValueError("No input files provided.")

    all_labels = []
    all_position_keys = []
    file_sizes = []
    ref_attrs  = None
    ref_shape  = None

    for i, path in enumerate(h5_paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input file not found: {path}")

        with h5py.File(path, 'r') as hf:
            shape  = hf['tensors'].shape
            labels = binary_labels(hf['labels'][:])
            position_keys = make_position_keys(hf['ref_names'][:],
                                               hf['ref_pos'][:])
            attrs  = dict(hf.attrs)

        tensor_shape = shape[1:]

        if i == 0:
            ref_shape = tensor_shape
            ref_attrs = attrs
        else:
            if tensor_shape != ref_shape:
                raise ValueError(
                    f"Tensor shape mismatch:\n"
                    f"  {h5_paths[0]} : {ref_shape}\n"
                    f"  {path}        : {tensor_shape}\n"
                    f"All files must be featurized with identical settings."
                )

        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        print(f"  {os.path.basename(path):40s}  "
              f"N={shape[0]:>7,}  mod={n_pos:>6,}  unmod={n_neg:>6,}",
              file=sys.stderr)

        all_labels.append(labels)
        all_position_keys.extend(position_keys)
        file_sizes.append(shape[0])

    return (np.concatenate(all_labels),
            np.array(file_sizes, dtype=np.int64),
            ref_attrs,
            all_position_keys)


# ── resume helpers ────────────────────────────────────────────────────────────

def lodo_stem(h5_path: str) -> str:
    """Canonical fold identifier: stem of the HDF5 filename without extension."""
    return Path(h5_path).stem


def lodo_sentinel(out_dir: Path, h5_path: str) -> Path:
    """Per-fold sentinel file written after a LODO fold completes."""
    return out_dir / f'lodo_{lodo_stem(h5_path)}_result.npz'


def save_lodo_result(out_dir: Path, h5_path: str, result: dict):
    """
    Persist a LODO fold result dict to an npz so we can skip it on resume.

    All values are stored as scalars (0-d arrays) or strings so they can be
    round-tripped through numpy savez / load without type loss.
    """
    path = lodo_sentinel(out_dir, h5_path)
    np.savez(
        path,
        held_out   = str(result['held_out']),
        precision  = float(result['precision']),
        recall     = float(result['recall']),
        f1         = float(result['f1']),
        auprc      = float(result['auprc']),
        auroc      = float(result['auroc']),
        threshold  = float(result['threshold']),
        n_train    = int(result['n_train']),
        n_test     = int(result['n_test']),
        all_unmod  = bool(result['all_unmod']),
        train_labels = np.asarray(result.get('train_labels', []), dtype=np.int64),
        test_labels  = np.asarray(result.get('test_labels', []), dtype=np.int64),
        y_prob       = np.asarray(result.get('y_prob', []), dtype=np.float32),
    )
    print(f"  LODO fold sentinel → {path}")


def load_lodo_result(out_dir: Path, h5_path: str) -> dict:
    """Reload a previously saved LODO fold result dict."""
    path = lodo_sentinel(out_dir, h5_path)
    d = np.load(path, allow_pickle=True)
    result = {
        'held_out' : str(d['held_out']),
        'precision': float(d['precision']),
        'recall'   : float(d['recall']),
        'f1'       : float(d['f1']),
        'auprc'    : float(d['auprc']),
        'auroc'    : float(d['auroc']),
        'threshold': float(d['threshold']),
        'n_train'  : int(d['n_train']),
        'n_test'   : int(d['n_test']),
        'all_unmod': bool(d['all_unmod']),
    }
    if 'train_labels' in d.files and len(d['train_labels']) > 0:
        result['train_labels'] = d['train_labels'].astype(np.int64)
    if 'test_labels' in d.files and len(d['test_labels']) > 0:
        result['test_labels'] = d['test_labels'].astype(np.int64)
    if 'y_prob' in d.files and len(d['y_prob']) > 0:
        result['y_prob'] = d['y_prob'].astype(np.float32)
    return result


def main_training_state_path(out_dir: Path) -> Path:
    """Checkpoint for an unfinished main training stage."""
    return out_dir / 'training_state.pt'


def save_main_training_state(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    best_auprc: float,
    best_epoch: int,
    patience_count: int,
    train_losses: list[float],
    val_losses: list[float],
    val_auprcs: list[float],
    args,
    in_ch: int,
    global_step: int = 0,
    warmup_sched=None,
) -> None:
    state = {
        'epoch': int(epoch),
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_auprc': float(best_auprc),
        'best_epoch': int(best_epoch),
        'patience_count': int(patience_count),
        'train_losses': list(train_losses),
        'val_losses': list(val_losses),
        'val_auprcs': list(val_auprcs),
        'args': vars(args),
        'in_channels': int(in_ch),
        'global_step': int(global_step),
    }
    if warmup_sched is not None:
        state['warmup_sched_state'] = warmup_sched.state_dict()
    torch.save(state, path)


def load_main_metrics(pred_path: Path, ckpt_path: Path) -> dict | None:
    """
    Reconstruct main_metrics from test_predictions.npz + best_model.pt.

    Returns None if either file is missing or malformed (triggers retraining).
    The returned dict matches the shape built in main() so downstream code
    (ZeroR, channel importance, LODO summary) runs unchanged.
    """
    if not pred_path.exists() or not ckpt_path.exists():
        return None
    try:
        d    = np.load(pred_path, allow_pickle=True)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        y_true = d['base_y_true']
        y_prob = d['base_y_prob']
        y_true_img = d['image_y_true']
        y_prob_img = d['image_y_prob']

        # Recompute threshold from saved predictions (deterministic)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1s = (2 * precisions[:-1] * recalls[:-1]
               / (precisions[:-1] + recalls[:-1] + 1e-8))
        best_thresh = float(thresholds[int(np.argmax(f1s))])

        y_pred = (y_prob >= best_thresh).astype(int)

        auprc = (average_precision_score(y_true, y_prob)
                 if int(y_true.sum()) > 0 else float(np.mean(1.0 - y_prob)))
        auroc = (roc_auc_score(y_true, y_prob)
                 if len(np.unique(y_true)) == 2 else float('nan'))
        img_auprc = (average_precision_score(y_true_img, y_prob_img)
                     if int(y_true_img.sum()) > 0
                     else float(np.mean(1.0 - y_prob_img)))
        img_auroc = (roc_auc_score(y_true_img, y_prob_img)
                     if len(np.unique(y_true_img)) == 2 else float('nan'))

        return {
            'precision'     : float(precision_score(y_true, y_pred, zero_division=0)),
            'recall'        : float(recall_score(y_true, y_pred, zero_division=0)),
            'f1'            : float(f1_score(y_true, y_pred, zero_division=0)),
            'auprc'         : auprc,
            'auroc'         : auroc,
            'image_auprc'   : img_auprc,
            'image_auroc'   : img_auroc,
            'threshold'     : best_thresh,
            'n_train'       : int(d.get('n_train', 0)) if 'n_train' in d.files else 0,
            'n_test'        : int(len(y_true)),
            'n_train_images': int(d.get('n_train_images', 0)) if 'n_train_images' in d.files else 0,
            'n_test_images' : int(len(y_true_img)),
            'all_unmod'     : bool(int(y_true.sum()) == 0),
            'y_true'        : y_true,
            'y_prob'        : y_prob,
            # carry through epoch info from checkpoint
            'best_epoch'    : int(ckpt.get('epoch', 0)),
            'val_auprc'     : float(ckpt.get('val_auprc', float('nan'))),
        }
    except Exception as exc:
        print(f"WARNING: could not reload main metrics from {pred_path}: {exc}",
              file=sys.stderr)
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train scaled InceptionV3 CNN on nanopore pileup tensors.')
    parser.add_argument('--input', required=True, nargs='+', metavar='H5',
                        help='One or more HDF5 files from featurize_pileup.py.')
    parser.add_argument('--out-dir',     required=True)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch',       type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--val-frac',    type=float, default=0.15)
    parser.add_argument('--test-frac',   type=float, default=0.15)
    parser.add_argument('--split-mode', choices=('position', 'contig'),
                        default='position',
                        help='Leakage-safe split unit. "position" keeps all '
                             'images from a reference base together; "contig" '
                             'keeps all images from a reference contig together.')
    parser.add_argument('--patience',    type=int,   default=15)
    parser.add_argument('--dropout',     type=float, default=0.4)
    parser.add_argument('--focal',       action='store_true',
                        help='Use focal loss instead of BCE + pos_weight. '
                             'Recommended only for very small positive counts '
                             '(<1K). For larger datasets BCE is more stable.')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma (only used with --focal, default: 2.0)')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='Mixup alpha; 0 disables. Do not use with --focal.')
    parser.add_argument('--signal-noise-std', type=float, default=0.05,
                        help='Stddev of additive Gaussian noise on raw signal '
                             'channel during training augmentation.')
    parser.add_argument('--rc-augment', action='store_true',
                        help='Use reverse-complement augmentation. This swaps '
                             'base channels and strand after reversing width; '
                             'leave disabled for strand-specific level tables.')
    parser.add_argument('--balanced-sampler', action='store_true',
                        help='Use replacement sampling with balanced modified/'
                             'unmodified class probability for each epoch. '
                             'This is useful for fast deadline runs on strongly '
                             'imbalanced datasets.')
    parser.add_argument('--epoch-samples', type=int, default=0,
                        help='Number of examples drawn per sampled epoch when '
                             '--balanced-sampler is enabled. 0 means use the '
                             'full training-set size.')
    parser.add_argument('--num-workers', type=int,   default=0,
                        help='DataLoader workers. Default 0 is safest for HDF5 '
                             'and restricted HPC runtimes; use >0 only when '
                             'multiprocessing/shared-memory transfer works.')
    parser.add_argument('--log-every',   type=int,   default=200,
                        help='Print training progress every N batches '
                             '(0 disables; default: 200).')
    parser.add_argument('--skip-channel-importance', action='store_true',
                        help='Skip expensive permutation channel-importance '
                             'analysis after test-set evaluation.')
    parser.add_argument('--channel-importance-repeats', type=int, default=5,
                        help='Permutation repeats per channel when channel '
                             'importance is enabled.')
    parser.add_argument('--skip-lodo', action='store_true',
                        help='Skip leave-one-dataset-out retraining after the '
                             'main model finishes.')
    parser.add_argument('--lodo-held-out', default=None, metavar='STEM',
                        help='Run only the LODO fold for this dataset stem '
                             '(stem = HDF5 filename without extension). '
                             'Requires best_model.pt + test_predictions.npz '
                             'to exist in --out-dir. Skips final summary '
                             'plots (run collect_lodo.py to regenerate them).')
    parser.add_argument('--seed',        type=int,   default=42)
    # ── resume control ────────────────────────────────────────────────────────
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore existing checkpoint files and rerun all '
                             'stages from scratch.  By default the script '
                             'resumes from wherever it left off.')
    parser.add_argument('--no-delta-channels', action='store_true',
                        help='Disable the two computed delta channels (9 and '
                             '10). By default, ch9 (per-read center-position '
                             'signal deviation broadcast across the row) and '
                             'ch10 (full-window signal deviation) are appended '
                             'at load time from existing HDF5 data without '
                             're-featurization. Pass this flag to ablate.')
    parser.add_argument('--lr-warmup-steps', type=int, default=0,
                        help='Number of optimizer steps for linear LR warmup. '
                             'LR ramps from ~0 to --lr over this many steps '
                             'before ReduceLROnPlateau takes over. 0 disables.')
    args = parser.parse_args()

    # Warn about bad combination
    if args.focal and args.mixup_alpha > 0:
        print("WARNING: --focal + --mixup-alpha > 0 is not recommended. "
              "Soft mixup targets break focal loss p_t computation. "
              "Consider --mixup-alpha 0 when using --focal.", file=sys.stderr)

    resume = not args.no_resume

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", file=sys.stderr)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("CUDA: TF32 enabled; cuDNN benchmark enabled", file=sys.stderr)

    h5_paths = args.input
    print(f"Input files: {len(h5_paths)}", file=sys.stderr)
    labels, file_sizes, attrs, position_keys = load_and_validate_files(h5_paths)

    N = len(labels)
    print(f"\nCombined dataset: {N:,} images  "
          f"(modified={int(labels.sum()):,}  "
          f"unmodified={N-int(labels.sum()):,})", file=sys.stderr)

    delta_channels = not args.no_delta_channels
    in_ch  = int(attrs.get('n_channels', 9)) + (2 if delta_channels else 0)
    height = int(attrs.get('height', attrs.get('max_reads', 30) + 1))
    W      = int(attrs.get('W', 21))
    L      = int(attrs.get('L', 10))
    print(f"Tensor shape per sample: ({height} × {W*L} × {in_ch})  [H × W × C]",
          file=sys.stderr)
    print(f"  (row 0 = reference track, rows 1-{height-1} = reads)", file=sys.stderr)
    print(f"  delta channels: {'enabled (ch9+ch10)' if delta_channels else 'disabled'}",
          file=sys.stderr)

    # ── leakage-safe split ────────────────────────────────────────────────────
    split_fn = (split_contig_groups
                if args.split_mode == 'contig'
                else split_position_groups)
    train_idx, val_idx, test_idx, split_stats = split_fn(
        labels=labels,
        position_keys=position_keys,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    train_base_keys, train_file_idx = source_position_keys(
        train_idx, position_keys, file_sizes)
    val_base_keys, val_file_idx = source_position_keys(
        val_idx, position_keys, file_sizes)
    test_base_keys, test_file_idx = source_position_keys(
        test_idx, position_keys, file_sizes)

    print(f"Split mode: {args.split_mode}", file=sys.stderr)
    if args.split_mode == 'contig':
        print(f"Split contigs — train: {split_stats['train_contigs']:,}  "
              f"val: {split_stats['val_contigs']:,}  "
              f"test: {split_stats['test_contigs']:,}", file=sys.stderr)
    print(f"Split images — train: {len(train_idx):,}  "
          f"val: {len(val_idx):,}  test: {len(test_idx):,}", file=sys.stderr)
    print(f"Split reference bases — train: {split_stats['train_positions']:,}  "
          f"val: {split_stats['val_positions']:,}  "
          f"test: {split_stats['test_positions']:,}", file=sys.stderr)
    print(f"Split source/base calls — train: {len(set(train_base_keys)):,}  "
          f"val: {len(set(val_base_keys)):,}  "
          f"test: {len(set(test_base_keys)):,}", file=sys.stderr)

    # ── datasets & loaders ────────────────────────────────────────────────────
    train_ds = PileupDataset(h5_paths, train_idx, file_sizes,
                              augment=True,  seed=args.seed,
                              rc_augment=args.rc_augment,
                              signal_noise_std=args.signal_noise_std,
                              delta_channels=delta_channels)
    val_ds   = PileupDataset(h5_paths, val_idx,   file_sizes,
                              augment=False, seed=args.seed,
                              delta_channels=delta_channels)
    test_ds  = PileupDataset(h5_paths, test_idx,  file_sizes,
                              augment=False, seed=args.seed,
                              delta_channels=delta_channels)

    train_labels = labels[train_idx]

    loader_kwargs = make_loader_kwargs(
        batch_size=args.batch,
        num_workers=args.num_workers,
        device=device,
        worker_init_fn=_worker_init_fn,
    )
    if args.num_workers == 0:
        print("DataLoader: num_workers=0 (single-process HDF5 reads)",
              file=sys.stderr)
    else:
        print(f"DataLoader: num_workers={args.num_workers}  "
              f"pin_memory={loader_kwargs['pin_memory']}",
              file=sys.stderr)
    sampler = None
    if args.balanced_sampler:
        epoch_samples = (args.epoch_samples
                         if args.epoch_samples and args.epoch_samples > 0
                         else len(train_labels))
        sampler = make_sampler(train_labels, num_samples=epoch_samples)
        print(f"Training sampler: balanced replacement draws "
              f"{epoch_samples:,} examples/epoch "
              f"({int(train_labels.sum()):,} modified, "
              f"{len(train_labels)-int(train_labels.sum()):,} unmodified "
              f"available)", file=sys.stderr)
    elif args.epoch_samples and args.epoch_samples > 0:
        print("WARNING: --epoch-samples is ignored unless --balanced-sampler "
              "is enabled.", file=sys.stderr)

    train_loader = DataLoader(train_ds, sampler=sampler,
                              shuffle=(sampler is None), **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── model (always instantiated; weights loaded from ckpt on resume) ───────
    model = PileupInceptionV3(in_channels=in_ch, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: PileupInceptionV3 (scaled)  ({n_params:,} parameters)",
          file=sys.stderr)

    ckpt_path = out_dir / 'best_model.pt'
    pred_path = out_dir / 'test_predictions.npz'

    if args.lodo_held_out and not (ckpt_path.exists() and pred_path.exists()):
        print(f"ERROR: --lodo-held-out '{args.lodo_held_out}' requires "
              f"best_model.pt and test_predictions.npz to exist in {out_dir}.\n"
              f"Run main training first without --lodo-held-out "
              f"(pass --skip-lodo to skip the sequential LODO).",
              file=sys.stderr)
        sys.exit(1)
    if args.lodo_held_out:
        args.skip_channel_importance = True

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Main training + test evaluation
    # Sentinel: test_predictions.npz
    # ══════════════════════════════════════════════════════════════════════════
    resumed_main = False
    main_metrics  = None

    if resume and pred_path.exists() and ckpt_path.exists():
        main_metrics = load_main_metrics(pred_path, ckpt_path)

    if main_metrics is not None:
        # ── resumed: reload checkpoint weights and cached predictions ─────────
        print(f"\n[RESUME] Skipping main training — "
              f"found {pred_path.name} + {ckpt_path.name}", file=sys.stderr)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        best_epoch  = main_metrics['best_epoch']
        best_auprc  = main_metrics['val_auprc']
        best_thresh = main_metrics['threshold']

        d = np.load(pred_path, allow_pickle=True)
        y_true_test_img = d['image_y_true']
        y_prob_test_img = d['image_y_prob']
        y_true_test     = d['base_y_true']
        y_prob_test     = d['base_y_prob']
        test_position_keys_agg = list(zip(
            d['base_file_idx'].tolist(),
            d['base_ref_names'].tolist(),
            d['base_ref_pos'].tolist(),
        ))

        auroc     = main_metrics['auroc']
        auprc     = main_metrics['auprc']
        img_auroc = main_metrics['image_auroc']
        img_auprc = main_metrics['image_auprc']
        resumed_main = True

        print(f"  Restored best epoch {best_epoch}  "
              f"val AUPRC={best_auprc:.4f}  "
              f"test AUPRC={auprc:.4f}  thresh={best_thresh:.4f}",
              file=sys.stderr)
    else:
        # ── fresh run: full training loop ──────────────────────────────────────
        actual_pw = float(get_pos_weight(train_labels, device).item())
        if args.balanced_sampler:
            pw = 1.0
            print(f"Loss pos_weight reset to 1.00 because --balanced-sampler "
                  f"already balances training batches "
                  f"(full-data pos_weight would be {actual_pw:.2f})",
                  file=sys.stderr)
        else:
            pw = actual_pw

        if args.focal:
            print(f"Loss: Focal  pos_weight={pw:.2f}  gamma={args.focal_gamma}",
                  file=sys.stderr)
            criterion = FocalBCELoss(pos_weight=pw, gamma=args.focal_gamma)
        else:
            print(f"Loss: BCE  pos_weight={pw:.2f}", file=sys.stderr)
            pw_tensor = torch.tensor(pw, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)

        # Linear LR warmup: ramp from ~0 to args.lr over lr_warmup_steps steps,
        # then hand off to ReduceLROnPlateau for epoch-level decay.
        warmup_sched = None
        if args.lr_warmup_steps > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(args.lr_warmup_steps, 1),
                end_factor=1.0,
                total_iters=args.lr_warmup_steps,
            )
            print(f"LR warmup: {args.lr_warmup_steps} steps  "
                  f"({args.lr_warmup_steps / max(len(train_loader), 1):.1f} epochs)",
                  file=sys.stderr)

        best_auprc     = -1.0
        best_epoch     = 1
        patience_count = 0
        global_step    = 0
        train_losses, val_losses, val_auprcs = [], [], []
        start_epoch = 1
        state_path = main_training_state_path(out_dir)

        if resume and state_path.exists():
            try:
                state = torch.load(state_path, map_location=device,
                                   weights_only=False)
                model.load_state_dict(state['model_state'])
                optimizer.load_state_dict(state['optimizer_state'])
                scheduler.load_state_dict(state['scheduler_state'])
                best_auprc = float(state.get('best_auprc', best_auprc))
                best_epoch = int(state.get('best_epoch', best_epoch))
                patience_count = int(state.get('patience_count', patience_count))
                train_losses = list(state.get('train_losses', []))
                val_losses = list(state.get('val_losses', []))
                val_auprcs = list(state.get('val_auprcs', []))
                global_step = int(state.get('global_step', 0))
                start_epoch = int(state['epoch']) + 1
                if warmup_sched is not None and 'warmup_sched_state' in state:
                    warmup_sched.load_state_dict(state['warmup_sched_state'])
                print(f"\n[RESUME] Continuing main training from "
                      f"epoch {start_epoch}/{args.epochs} "
                      f"(best epoch {best_epoch}, AUPRC={best_auprc:.4f})",
                      file=sys.stderr)
            except Exception as exc:
                print(f"WARNING: could not load main training state "
                      f"{state_path}: {exc}. Restarting main training.",
                      file=sys.stderr)
                best_auprc = -1.0
                best_epoch = 1
                patience_count = 0
                global_step = 0
                train_losses, val_losses, val_auprcs = [], [], []
                start_epoch = 1

        with torch.no_grad():
            dummy = torch.zeros(2, in_ch, height, W * L, device=device)
            out   = model(dummy)
            print(f"Forward pass OK: input {tuple(dummy.shape)} → "
                  f"output {tuple(out.shape)}", file=sys.stderr)

        if patience_count >= args.patience:
            print("[RESUME] Main training had already reached early-stopping "
                  "patience; evaluating best checkpoint.", file=sys.stderr)
            start_epoch = args.epochs + 1
        elif start_epoch > args.epochs:
            print(f"[RESUME] Main training already completed {args.epochs} "
                  f"epochs; evaluating best checkpoint.", file=sys.stderr)

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_seen = 0
            n_train_batches = len(train_loader)
            for batch_idx, (x, y) in enumerate(train_loader, start=1):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                if args.mixup_alpha > 0 and not args.focal:
                    x, y = mixup_batch(x, y, alpha=args.mixup_alpha)

                optimizer.zero_grad()
                logits = model(x).squeeze(1)
                loss   = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1
                # Step warmup scheduler per batch until warmup is complete.
                if warmup_sched is not None and global_step <= args.lr_warmup_steps:
                    warmup_sched.step()
                epoch_loss += loss.item() * len(y)
                epoch_seen += len(y)
                if args.log_every > 0 and (
                        batch_idx == 1 or batch_idx % args.log_every == 0):
                    print(f"  epoch {epoch:3d} batch "
                          f"{batch_idx:>6}/{n_train_batches:<6}  "
                          f"batch_loss={loss.item():.4f}",
                          flush=True)
            train_loss = epoch_loss / max(epoch_seen, 1)

            val_loss, y_true_val_img, y_prob_val_img = run_inference_and_loss(
                model, val_loader, criterion, device)
            y_true_val, y_prob_val, _ = aggregate_by_position(
                y_true_val_img, y_prob_val_img, val_base_keys)
            if int(y_true_val.sum()) > 0:
                val_auprc = average_precision_score(y_true_val, y_prob_val)
            else:
                val_auprc = float(np.mean(1.0 - y_prob_val))
            if len(np.unique(y_true_val)) == 2:
                val_auroc = roc_auc_score(y_true_val, y_prob_val)
            else:
                val_auroc = float('nan')

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_auprcs.append(val_auprc)

            # ReduceLROnPlateau only takes over once warmup is complete.
            if warmup_sched is None or global_step > args.lr_warmup_steps:
                scheduler.step(val_auprc)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_AUPRC={val_auprc:.4f}  val_AUROC={val_auroc:.4f}  "
                  f"lr={current_lr:.2e}")

            if val_auprc > best_auprc:
                best_auprc     = val_auprc
                best_epoch     = epoch
                patience_count = 0
                torch.save({
                    'epoch':       epoch,
                    'model_state': model.state_dict(),
                    'val_auprc':   val_auprc,
                    'args':        vars(args),
                    'in_channels': in_ch,
                }, ckpt_path)
                print(f"  ✓ New best  AUPRC={best_auprc:.4f}  checkpoint saved")
            else:
                patience_count += 1

            save_main_training_state(
                state_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_auprc=best_auprc,
                best_epoch=best_epoch,
                patience_count=patience_count,
                train_losses=train_losses,
                val_losses=val_losses,
                val_auprcs=val_auprcs,
                args=args,
                in_ch=in_ch,
                global_step=global_step,
                warmup_sched=warmup_sched,
            )

            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # Restore best & evaluate
        if not ckpt_path.exists():
            raise RuntimeError(
                f"No best checkpoint available at {ckpt_path}; cannot evaluate "
                f"main model.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        print(f"\nRestored best model from epoch {best_epoch}  "
              f"(val AUPRC = {best_auprc:.4f})")

        print("\n=== Test Set Evaluation: Images ===")
        y_true_test_img, y_prob_test_img = run_inference(model, test_loader, device)
        y_true_test, y_prob_test, test_position_keys_agg = aggregate_by_position(
            y_true_test_img, y_prob_test_img, test_base_keys)
        img_auroc, img_auprc = evaluate(
            y_true_test_img, y_prob_test_img, unit_name='images')

        print("\n=== Test Set Evaluation: Source/Base Calls ===")
        auroc, auprc = evaluate(
            y_true_test, y_prob_test, unit_name='source/base calls')

        plot_training_curves(
            train_losses, val_losses, val_auprcs, best_epoch,
            out_path=str(out_dir / 'training_curves.png'),
        )
        save_training_history(
            out_dir / 'training_history.npz',
            train_losses, val_losses, val_auprcs, best_epoch,
        )
        best_thresh = plot_precision_recall(
            y_true_test, y_prob_test, auprc,
            out_path=str(out_dir / 'precision_recall.png'),
        )
        plot_confusion_matrix(
            y_true_test, y_prob_test, best_thresh,
            out_path=str(out_dir / 'confusion_matrix.png'),
        )

        base_file_idx  = np.array([k[0] for k in test_position_keys_agg], dtype=np.int64)
        base_ref_names = np.array([k[1] for k in test_position_keys_agg])
        base_ref_pos   = np.array([k[2] for k in test_position_keys_agg], dtype=np.int64)
        y_true_train_base, train_base_y_prob, train_position_keys_agg = (
            aggregate_by_position(
                labels[train_idx],
                labels[train_idx].astype(np.float32),
                train_base_keys,
            )
        )
        train_base_file_idx = np.array(
            [k[0] for k in train_position_keys_agg], dtype=np.int64)
        train_base_ref_names = np.array([k[1] for k in train_position_keys_agg])
        train_base_ref_pos = np.array(
            [k[2] for k in train_position_keys_agg], dtype=np.int64)

        np.savez(pred_path,
                 y_true=y_true_test,
                 y_prob=y_prob_test,
                 ref_names=base_ref_names,
                 ref_pos=base_ref_pos,
                 file_idx=base_file_idx,
                 base_y_true=y_true_test,
                 base_y_prob=y_prob_test,
                 base_ref_names=base_ref_names,
                 base_ref_pos=base_ref_pos,
                 base_file_idx=base_file_idx,
                 image_y_true=y_true_test_img,
                 image_y_prob=y_prob_test_img,
                 test_indices=test_idx,
                 test_file_idx=test_file_idx,
                 train_indices=train_idx,
                 val_indices=val_idx,
                 split_mode=np.array(args.split_mode),
                 split_stat_names=np.array(list(split_stats.keys())),
                 split_stat_values=np.array(list(split_stats.values()), dtype=np.int64),
                 train_base_y_true=y_true_train_base,
                 train_base_y_prob=train_base_y_prob,
                 train_base_ref_names=train_base_ref_names,
                 train_base_ref_pos=train_base_ref_pos,
                 train_base_file_idx=train_base_file_idx,
                 h5_paths=np.array(h5_paths))
        print(f"  Test predictions → {pred_path}")

        y_pred_test  = (y_prob_test >= best_thresh).astype(int)
        main_metrics = {
            'precision'     : float(precision_score(y_true_test, y_pred_test, zero_division=0)),
            'recall'        : float(recall_score(y_true_test, y_pred_test, zero_division=0)),
            'f1'            : float(f1_score(y_true_test, y_pred_test, zero_division=0)),
            'auprc'         : float(auprc),
            'auroc'         : float(auroc),
            'image_auprc'   : float(img_auprc),
            'image_auroc'   : float(img_auroc),
            'threshold'     : float(best_thresh),
            'n_train'       : int(len(set(train_base_keys))),
            'n_test'        : int(len(y_true_test)),
            'n_train_images': int(len(train_idx)),
            'n_test_images' : int(len(test_idx)),
            'all_unmod'     : bool(int(y_true_test.sum()) == 0),
            'y_true'         : y_true_test,
            'y_prob'         : y_prob_test,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — ZeroR baseline (always recomputed; cheap)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== ZeroR Baseline ===")
    y_true_train_base, _, _ = aggregate_by_position(
        labels[train_idx], labels[train_idx].astype(np.float32), train_base_keys)
    zeror_base = loo_mod.zeror_metrics(
        train_labels=y_true_train_base,
        test_labels=y_true_test,
    )
    print(f"  Majority class: {zeror_base['majority_class']}  "
          f"Prec={zeror_base['precision']:.4f}  "
          f"Rec={zeror_base['recall']:.4f}  "
          f"F1={zeror_base['f1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Channel importance
    # Sentinel: channel_importance.npz
    # ══════════════════════════════════════════════════════════════════════════
    ci_path = out_dir / 'channel_importance.npz'

    if args.skip_channel_importance:
        print("\nSkipping Channel Importance (--skip-channel-importance)")
        importance = None
    elif resume and ci_path.exists():
        print(f"\n[RESUME] Skipping channel importance — found {ci_path.name}",
              file=sys.stderr)
        importance = np.load(ci_path)['importance']
        print("\n=== Channel Importance (loaded from cache) ===")
        channel_names = CHANNEL_NAMES[:in_ch]
        for ch_name, imp in zip(channel_names, importance):
            print(f"  {ch_name:20s}  AUPRC drop = {imp:+.4f}")
        # Regenerate the plot in case it was lost
        plot_channel_importance(
            importance=importance,
            baseline_auprc=float(img_auprc),
            out_path=str(out_dir / 'channel_importance.png'),
            channel_names=channel_names,
        )
    else:
        print("\n=== Channel Importance (permutation) ===")
        importance = loo_mod.permutation_channel_importance(
            model=model,
            loader=test_loader,
            device=device,
            n_channels=in_ch,
            n_repeats=args.channel_importance_repeats,
            seed=args.seed,
        )
        channel_names = CHANNEL_NAMES[:in_ch]
        for ch_name, imp in zip(channel_names, importance):
            print(f"  {ch_name:20s}  AUPRC drop = {imp:+.4f}")

        plot_channel_importance(
            importance=importance,
            baseline_auprc=float(img_auprc),
            out_path=str(out_dir / 'channel_importance.png'),
            channel_names=channel_names,
        )
        # Save sentinel so we can skip this on resume
        np.savez(ci_path,
                 importance=importance,
                 baseline_auprc=float(img_auprc),
                 channel_names=np.array(channel_names))
        print(f"  Channel importance sentinel → {ci_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — LODO evaluation
    # Per-fold sentinel: lodo_{stem}_result.npz
    # ══════════════════════════════════════════════════════════════════════════
    if args.skip_lodo:
        print("\nSkipping LODO evaluation (--skip-lodo)")
        loo_results = []
    elif len(h5_paths) >= 2:
        print(f"\n{'═'*60}")
        print(f"  Starting LODO evaluation ({len(h5_paths)} folds)")
        print(f"{'═'*60}")

        loo_results = []
        for held_out_path in h5_paths:
            fold_sentinel = lodo_sentinel(out_dir, held_out_path)
            stem          = lodo_stem(held_out_path)

            # When running a single fold in parallel, skip all other folds.
            # Load their sentinels so the summary (if any) has all results.
            if args.lodo_held_out and stem != args.lodo_held_out:
                if resume and fold_sentinel.exists():
                    result = load_lodo_result(out_dir, held_out_path)
                    loo_results.append(result)
                continue

            if resume and fold_sentinel.exists():
                print(f"\n[RESUME] Skipping LODO fold '{stem}' — "
                      f"found {fold_sentinel.name}", file=sys.stderr)
                result = load_lodo_result(out_dir, held_out_path)
                loo_results.append(result)
                continue

            print(f"\n  ── LODO fold: held-out = {stem} ──")
            result = loo_mod.run_lodo(
                held_out_path=held_out_path,
                all_h5_paths=h5_paths,
                PileupDataset=PileupDataset,
                PileupInceptionV3=PileupInceptionV3,
                get_pos_weight=get_pos_weight,
                make_sampler=make_sampler,
                run_inference=run_inference,
                plot_training_curves=plot_training_curves,
                device=device,
                out_dir=out_dir,
                epochs=args.epochs,
                batch=args.batch,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                dropout=args.dropout,
                num_workers=args.num_workers,
                no_oversample=True,
                seed=args.seed,
                val_frac=args.val_frac,
                use_focal=args.focal,
                focal_gamma=args.focal_gamma,
                rc_augment=args.rc_augment,
                signal_noise_std=args.signal_noise_std,
                resume=resume,
                delta_channels=delta_channels,
                lr_warmup_steps=args.lr_warmup_steps,
            )
            # Write the per-fold sentinel immediately so a crash after this
            # fold does not force it to rerun.
            save_lodo_result(out_dir, held_out_path, result)
            loo_results.append(result)

        # Summary plots are written by collect_lodo.py when folds run in parallel.
        if not args.lodo_held_out:
            plot_loo_results(
                loo_results=loo_results,
                main_metrics=main_metrics,
                zeror_base=zeror_base,
                out_path=str(out_dir / 'lodo_comparison.png'),
            )
            save_loo_metrics_tsv(
                loo_results=loo_results,
                main_metrics=main_metrics,
                zeror_base=zeror_base,
                out_path=str(out_dir / 'lodo_metrics.tsv'),
            )
    else:
        print("\nSkipping LODO evaluation: need >= 2 input files.", file=sys.stderr)
        loo_results = []

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    loss_desc = (f"Focal(gamma={args.focal_gamma}, pos_weight=<see log>)"
                 if args.focal else f"BCE(pos_weight=<see log>)")
    print(f"\n{'═'*60}")
    print(f"  Final Summary")
    print(f"{'═'*60}")
    if resumed_main:
        print(f"  (main training stage was resumed from checkpoint)")
    print(f"  Model           : PileupInceptionV3 scaled ({n_params:,} params)")
    print(f"  Loss            : {loss_desc}")
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Val AUPRC       : {best_auprc:.4f}")
    print(f"  Test AUROC      : {auroc:.4f}  (source/base calls)")
    print(f"  Test AUPRC      : {auprc:.4f}  (source/base calls)")
    print(f"  Image AUROC     : {img_auroc:.4f}")
    print(f"  Image AUPRC     : {img_auprc:.4f}")
    print(f"  Optimal thresh  : {best_thresh:.4f}")
    print(f"  ZeroR F1        : {zeror_base['f1']:.4f}  "
          f"(majority class = {zeror_base['majority_class']})")
    if loo_results:
        print(f"\n  LODO results:")
        print(f"  {'Held-out':12s}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUPRC':>6}")
        for r in loo_results:
            note = '*' if r['all_unmod'] else ''
            print(f"  {r['held_out']:12s}  "
                  f"{r['precision']:>6.4f}  {r['recall']:>6.4f}  "
                  f"{r['f1']:>6.4f}  {r['auprc']:>6.4f}{note}")
    print(f"\n  Outputs in      : {out_dir}/")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
