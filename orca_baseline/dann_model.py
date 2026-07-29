#!/usr/bin/env python3
"""
ORCA-style domain-adversarial model on DeepMod pileup features.

Bhargav's ask: keep our per-position pileup-image features, put a CNN trunk in
front of a bi-LSTM, and add ORCA's gradient-reversal domain head (no
stoichiometry). The adversarial head forces the embedding to be invariant to
modification type, which is what drives cross-modification (leave-one-out)
generalization.

Input : (B, C, H, W)  pileup image  (C=channels, H=max_reads+1, W=window_pos*L)
Heads : binary modified/unmodified (BCE)  +  domain=modification-type (CE, via GRL)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.lambd, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.c = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.c(x)))


class CNNTrunkBiLSTM(nn.Module):
    """Pileup image -> CNN trunk -> per-position sequence -> bi-LSTM embedding."""
    def __init__(self, in_ch=9, window=21, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            ConvBNReLU(in_ch, 32, 3, 1, 1),
            ConvBNReLU(32, 64, 3, 2, 1),     # halve H and W
            ConvBNReLU(64, 128, 3, 1, 1),
        )
        # collapse the reads dim (H->1) and land on exactly `window` positions
        self.pool = nn.AdaptiveAvgPool2d((1, window))
        self.lstm = nn.LSTM(128, hidden, num_layers=2, bidirectional=True)
        self.out_dim = hidden * 2

    def forward(self, x):              # (B, C, H, W)
        h = self.trunk(x)             # (B, 128, H', W')
        h = self.pool(h).squeeze(2)   # (B, 128, window)
        h = h.permute(2, 0, 1)        # (window, B, 128)
        out, _ = self.lstm(h)         # (window, B, 2*hidden)
        return out.mean(0)            # (B, 2*hidden)  pooled over positions


class DANN(nn.Module):
    def __init__(self, in_ch=9, window=21, hidden=128, n_domains=4, dropout=0.3):
        super().__init__()
        self.backbone = CNNTrunkBiLSTM(in_ch, window, hidden)
        d = self.backbone.out_dim
        self.cls = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.dom = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_domains))

    def forward(self, x, grl_lambda=0.0):
        emb = self.backbone(x)
        logit = self.cls(emb)[:, 0]                       # binary mod/unmod
        dom = self.dom(grad_reverse(emb, grl_lambda))     # adversarial domain
        return logit, dom

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
