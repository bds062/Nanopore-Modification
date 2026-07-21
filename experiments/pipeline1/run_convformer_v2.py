#!/usr/bin/env python3
"""
results6 — ConvFormer-v2: multi-scale (mini-Inception-style) per-read 1D conv
trunk + the SAME cross-read Transformer as ConvFormer-v1 (results5).

This is a controlled ablation, not a bigger model for its own sake. Comparing
results1 (Inception) vs results5 (ConvFormer-v1) showed a clean split:

  - ConvFormer-v1 achieves genuine POSITIVE cross-dataset transfer on native
    UMCES (umces_only mod_f1 0.392 -> both mod_f1 0.509 when ONT is added to
    training) — Inception instead shows NEGATIVE transfer on both datasets when
    combined (umces_only 0.440 -> both 0.369; ont_only 0.926 -> both 0.811).
    ConvFormer-v1's lower train/val gap (train_loss 0.29 vs Inception's 0.03,
    val_AUPRC 0.769 vs 0.689) indicates a genuinely better-regularized fit, not
    just a different failure mode.
  - But ConvFormer-v1 is intrinsically worse than Inception at the clean
    synthetic ONT task, even in complete isolation (ont_only ONT mod_f1: 0.636
    vs Inception's 0.926) — evidence the flat 3-layer conv trunk (32->64->96,
    single kernel size per layer) lacks the multi-scale receptive-field
    diversity of Inception's stem + 9 Inception blocks (1x1/3x3/5x5/7x1/1x7
    branches) needed to capture ONT's fine local signal-deviation shape.

ConvFormer-v2 tests that hypothesis directly: it swaps ONLY the conv trunk for
a small multi-scale block (parallel kernel sizes, mirroring InceptionA) plus a
grid-reduction block (mirroring InceptionB), while leaving the cross-read
Transformer (the piece responsible for the positive-transfer effect) completely
unchanged. If ONT recovers without losing the UMCES transfer gain, the trunk
was the bottleneck, not the attention mechanism.

  (B, 11, 31, 210)  -- per read, along the W (signal) axis --
    stem:    Conv1d(11->32,k3) -> Conv1d(32->32,k3) -> Conv1d(32->48,k3,s2)  (W:210->105)
    blockA:  4-branch multi-scale (1x1 / 1x1->5 / 1x1->3->3 / pool->1x1)     (48->64ch, W:105)
    reduce:  3-branch grid reduction (3,s2 / 1x1->3->3,s2 / maxpool,s2)     (64->184ch, W:105->53)
    proj:    Conv1d(184->96,k1) -> AdaptiveAvgPool1d(1)                     (-> 96-dim per-read embedding)
    → (B, 31, 96) + row-position embed
    → TransformerEncoder (2 layers, 4 heads, dim_ff=192)  [same as ConvFormer-v1, masked]
    → masked mean-pool → LayerNorm → Dropout → Linear(1)

Reuses the entire evaluation framework from run_pipeline.py via model_factory.

Usage:
  python run_convformer_v2.py --model {ont_only|umces_only|both} [--out-dir results6] [--epochs N]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import run_pipeline as R

IN_CH, H, W = 11, 31, 210
D_MODEL = 96   # identical to ConvFormer-v1 — keeps the transformer an exact match


def conv_bn_relu_1d(in_ch, out_ch, k, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
    )


class InceptionA1d(nn.Module):
    """4-branch multi-scale block (1D analogue of InceptionV3's InceptionA)."""
    def __init__(self, in_ch, pool_proj=8):
        super().__init__()
        self.b1 = conv_bn_relu_1d(in_ch, 16, 1)
        self.b2 = nn.Sequential(conv_bn_relu_1d(in_ch, 12, 1), conv_bn_relu_1d(12, 16, 5, padding=2))
        self.b3 = nn.Sequential(conv_bn_relu_1d(in_ch, 16, 1),
                                conv_bn_relu_1d(16, 24, 3, padding=1),
                                conv_bn_relu_1d(24, 24, 3, padding=1))
        self.b4 = nn.Sequential(nn.AvgPool1d(3, stride=1, padding=1), conv_bn_relu_1d(in_ch, pool_proj, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class ReductionB1d(nn.Module):
    """3-branch grid reduction, halves W (1D analogue of InceptionV3's InceptionB)."""
    def __init__(self, in_ch):
        super().__init__()
        self.b1 = conv_bn_relu_1d(in_ch, 96, 3, stride=2, padding=1)
        self.b2 = nn.Sequential(conv_bn_relu_1d(in_ch, 16, 1),
                                conv_bn_relu_1d(16, 24, 3, padding=1),
                                conv_bn_relu_1d(24, 24, 3, stride=2, padding=1))
        self.b3 = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class ReadConvEncoderV2(nn.Module):
    """Multi-scale per-read conv trunk: stem -> InceptionA1d -> ReductionB1d -> project."""
    def __init__(self, in_ch=IN_CH, d_model=D_MODEL):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_relu_1d(in_ch, 32, 3, padding=1),
            conv_bn_relu_1d(32, 32, 3, padding=1),
            conv_bn_relu_1d(32, 48, 3, stride=2, padding=1),   # W: 210 -> 105
        )
        self.blockA = InceptionA1d(48, pool_proj=8)             # 48 -> 16+16+24+8=64 ch
        self.reduce = ReductionB1d(64)                          # 64 -> 96+24+64=184 ch, W: 105->53
        self.proj = conv_bn_relu_1d(184, d_model, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):                        # x: (B*H, C, W)
        x = self.stem(x)
        x = self.blockA(x)
        x = self.reduce(x)
        x = self.proj(x)
        return self.pool(x).squeeze(-1)           # (B*H, d_model)


class _GradReverse(torch.autograd.Function):
    """Identity forward; gradient multiplied by -lambda on the backward pass."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class ConvFormerV2(nn.Module):
    """
    With dann_lambda > 0, a sequence-context adversary is attached to the pooled
    representation via a gradient-reversal layer. It predicts the FLANKING
    reference bases (offsets ±1..±flank from the candidate; the candidate base
    itself is NOT predicted, since 6mA=A / 5mC=C is legitimate signal). Gradient
    reversal drives the shared representation to be UNINFORMATIVE about the motif
    context, so the model cannot fire on a recognition motif (e.g. GATC) from
    sequence alone and must rely on the modification signal. The adversary reads
    the flanking-base targets straight from the input tensor's reference row, so
    no extra labels are needed; forward() returns (logit, adv_loss) when active.
    """
    def __init__(self, in_ch=IN_CH, h=H, w=W, d_model=D_MODEL, nhead=4,
                 layers=2, dim_ff=192, dropout=0.4,
                 dann_lambda=0.0, dann_flank=4, window_positions=21):
        super().__init__()
        self.read_encoder = ReadConvEncoderV2(in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, h, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # --- sequence-context adversary ---
        self.dann_lambda = float(dann_lambda)
        self.dann_flank = int(dann_flank)
        self.Wp = int(window_positions)
        # candidate at window centre; flanking offsets exclude 0
        c = self.Wp // 2
        self.flank_cols = [c + d for d in range(-self.dann_flank, self.dann_flank + 1)
                           if d != 0 and 0 <= c + d < self.Wp]
        if self.dann_lambda > 0:
            self.adv = nn.Sequential(
                nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, len(self.flank_cols) * 4))

    def _flank_targets(self, x):
        """Per-flank-position reference base index (0-3) + validity, from row 0."""
        B = x.shape[0]
        L = x.shape[3] // self.Wp
        ref = x[:, 2:6, 0, :].view(B, 4, self.Wp, L).mean(dim=3)  # (B,4,Wp)
        cols = torch.tensor(self.flank_cols, device=x.device)
        oh = ref[:, :, cols]                        # (B,4,F)
        valid = oh.sum(dim=1) > 0.05                # (B,F) — position has a base
        tgt = oh.argmax(dim=1)                      # (B,F)
        return tgt, valid

    def forward(self, x):                        # (B, C, H, W)
        B, C, Hh, Ww = x.shape
        pad = x[:, 0].abs().sum(dim=2) < 1e-6     # (B, H); reference row never masked
        pad[:, 0] = False

        reads = x.permute(0, 2, 1, 3).reshape(B * Hh, C, Ww)
        emb = self.read_encoder(reads).view(B, Hh, -1) + self.pos

        enc = self.encoder(emb, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        pooled = (enc * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        rep = self.norm(pooled)
        logit = self.head(rep)

        if self.dann_lambda > 0 and self.training:
            tgt, valid = self._flank_targets(x)                  # (B,F)
            adv_logits = self.adv(grad_reverse(rep, self.dann_lambda))
            adv_logits = adv_logits.view(B, len(self.flank_cols), 4)
            ce = F.cross_entropy(adv_logits.reshape(-1, 4), tgt.reshape(-1),
                                 reduction='none').view(B, -1)
            adv_loss = (ce * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
            return logit, adv_loss
        return logit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=['ont_only', 'umces_only', 'both'])
    ap.add_argument('--out-dir', default=str(Path(__file__).resolve().parent / 'results6'))
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()

    n = sum(p.numel() for p in ConvFormerV2().parameters())
    print(f"ConvFormer-v2 — {n:,} params", flush=True)
    R.run_experiment(args.model, args.out_dir,
                     model_factory=lambda: ConvFormerV2(dropout=R.HP.dropout),
                     epochs_override=args.epochs)


if __name__ == '__main__':
    main()
