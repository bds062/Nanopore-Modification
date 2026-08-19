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
from model import ProjectionHead          # SupCon head reused from deepmod/model.py

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
    Two optional auxiliary training objectives hang off the pooled 96-d
    representation `rep` (both no-ops at eval — forward() returns the logit
    alone unless training AND an auxiliary is active):

    * dann_lambda > 0 — a sequence-context adversary via a gradient-reversal
      layer. It predicts the FLANKING reference bases (offsets ±1..±flank from
      the candidate; the candidate base itself is NOT predicted, since 6mA=A /
      5mC=C is legitimate signal). Gradient reversal drives `rep` to be
      UNINFORMATIVE about the motif context, so the model cannot fire on a
      recognition motif (e.g. GATC) from sequence alone. Targets are read
      straight from the input tensor's reference row, so no extra labels.

    * supcon_dim > 0 — a supervised-contrastive projection head (ProjectionHead
      from deepmod/model.py) maps `rep` onto an L2-normalised sphere. The
      SupCon loss itself is computed in the training loop (it needs the batch
      labels, which forward() does not receive), so forward() only RETURNS the
      projection embedding. SupCon pulls together every modified site and every
      unmodified site regardless of source dataset, forcing the encoder to find
      signal-level features shared across datasets rather than memorising a
      dataset/motif-specific context — the direct counter to motif memorisation.

    When training and any auxiliary is active, forward() returns
    (logit, aux) where aux is a dict with keys among {'adv_loss', 'proj'}.
    """
    def __init__(self, in_ch=IN_CH, h=H, w=W, d_model=D_MODEL, nhead=4,
                 layers=2, dim_ff=192, dropout=0.4,
                 dann_lambda=0.0, dann_flank=4, window_positions=21,
                 supcon_dim=0, sad_dim=0, org_adv_classes=0, org_adv_lambda=1.0):
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

        # --- supervised-contrastive projection head ---
        self.supcon_dim = int(supcon_dim)
        self.proj = (ProjectionHead(d_model, d_model, self.supcon_dim)
                     if self.supcon_dim > 0 else None)

        # --- Deep SAD one-class head (Ruff et al. 2019) ---
        # Maps rep -> a small space where UNMODIFIED sites are pulled to a fixed
        # centre and (seen) modifications are pushed away, so at test an UNSEEN
        # chemistry also lands far from the centre. bias=False (Deep SVDD hygiene:
        # a bias lets the net map everything to the centre, the trivial solution).
        # The anomaly score used at inference is ||sad_head(rep) - sad_center||.
        # sad_center is a buffer (fixed after init, saved in the checkpoint).
        self.sad_dim = int(sad_dim)
        self.sad_head = (nn.Linear(d_model, self.sad_dim, bias=False)
                         if self.sad_dim > 0 else None)
        if self.sad_head is not None:
            self.register_buffer('sad_center', torch.zeros(self.sad_dim))
        self._sad = None                       # last sad embedding (set in forward)

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

        # --- organism/dataset adversary (results13) ---
        # Gradient-reversed K-way organism classifier (ONT/SPO1/HP), same
        # grad_reverse primitive as the sequence-context adversary above but a
        # DIFFERENT target. Meant to erase organism substructure WITHIN the two
        # SupCon-formed presence balls, not replace SupCon's attractive pull --
        # see memory: strand-split-partial-fix (gradient reversal alone supplies
        # no pull, only erasure; SupCon supplies the pull). The loss itself
        # (needs external organism labels) is computed in the training loop,
        # same pattern as 'proj'/SupCon -- forward() only returns the logits.
        self.org_adv_classes = int(org_adv_classes)
        self.org_adv_lambda = float(org_adv_lambda)
        self.org_adv_head = (nn.Linear(d_model, self.org_adv_classes)
                             if self.org_adv_classes > 0 else None)

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

        # Always compute the SAD embedding when the head exists, so it is available
        # at eval (for centre init and anomaly scoring) as well as in training.
        if self.sad_head is not None:
            self._sad = self.sad_head(rep)

        if self.training and (self.dann_lambda > 0 or self.proj is not None
                              or self.sad_head is not None
                              or self.org_adv_head is not None):
            aux = {}
            if self.dann_lambda > 0:
                tgt, valid = self._flank_targets(x)              # (B,F)
                adv_logits = self.adv(grad_reverse(rep, self.dann_lambda))
                adv_logits = adv_logits.view(B, len(self.flank_cols), 4)
                ce = F.cross_entropy(adv_logits.reshape(-1, 4), tgt.reshape(-1),
                                     reduction='none').view(B, -1)
                aux['adv_loss'] = ((ce * valid.float()).sum()
                                   / valid.float().sum().clamp(min=1.0))
            if self.proj is not None:
                # L2-normalised contrastive embedding; the SupCon loss (which
                # needs the batch labels) is applied in the training loop.
                aux['proj'] = self.proj(rep)                     # (B, supcon_dim)
            if self.sad_head is not None:
                aux['sad'] = self._sad                           # (B, sad_dim)
            if self.org_adv_head is not None:
                # organism labels live outside x; loss computed in training loop
                aux['org_adv_logits'] = self.org_adv_head(
                    grad_reverse(rep, self.org_adv_lambda))       # (B, org_adv_classes)
            return logit, aux
        return logit


class ConvFormerV2DANN(nn.Module):
    """
    Two-head DANN variant (results7/results8): NO BCE head, NO SupCon, NO Deep
    SAD. Same backbone as ConvFormerV2 (read_encoder -> Transformer -> masked
    mean-pool -> LayerNorm = 96-d penultimate `rep`), but exactly two heads
    hang off `rep`, both classification-via-NLLLoss (raw logits returned; the
    training loop applies log_softmax + nll_loss, mirroring how ConvFormerV2's
    single BCE head returns a raw logit for BCEWithLogitsLoss):

      presence_head (2-way): predicts modification PRESENCE (mod vs unmod) --
          this IS the detector. Normal (non-reversed) gradient: the backbone
          is optimized to make `rep` MORE informative for this.

      adv_head (n_adv_classes-way): predicts either modification TYPE
          (results7, 5-way over the modified chemistries; unmodified images are
          excluded from this loss by the training loop, not by the model) or
          source DATASET/organism (results8, 3-way ONT/SPO1/HP; every image has
          one). Fed through the SAME gradient-reversal primitive already used
          by ConvFormerV2's sequence-context adversary (`grad_reverse` above):
          adv_head itself is trained normally to classify as well as it can,
          but the gradient reaching `rep` (and thus the shared backbone) is
          NEGATED and scaled by adv_lambda, so the backbone is adversarially
          pushed to make `rep` UNINFORMATIVE about type/dataset while it still
          must stay informative about presence. Purpose: test whether this
          directly attacks the organism-dominated embedding clustering found
          in `embedding-not-chemistry-space` / `results6-bce-supcon-tradeoff`.

    forward() returns (presence_logits, adv_logits) whenever training=True,
    else presence_logits alone (adv_head is training-only, exactly like the
    dann adversary in ConvFormerV2). The penultimate `rep` is retrievable at
    eval via a forward hook on presence_head (its INPUT), same technique
    score_genome.py already uses for ConvFormerV2.
    """
    def __init__(self, in_ch=IN_CH, h=H, w=W, d_model=D_MODEL, nhead=4,
                 layers=2, dim_ff=192, dropout=0.4,
                 n_adv_classes=5, adv_lambda=1.0):
        super().__init__()
        self.read_encoder = ReadConvEncoderV2(in_ch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, h, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.presence_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))
        self.n_adv_classes = int(n_adv_classes)
        self.adv_lambda = float(adv_lambda)
        self.adv_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, self.n_adv_classes))

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

        presence_logits = self.presence_head(rep)
        if self.training:
            adv_logits = self.adv_head(grad_reverse(rep, self.adv_lambda))
            return presence_logits, adv_logits
        return presence_logits


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
