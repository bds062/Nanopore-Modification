#!/usr/bin/env python3
"""
ORCA model, reimplemented faithfully from the released architecture
(bioinfo-biols/ORCA, orca/scripts/Models_20.py) since the paper ships no
training code. Kept structurally identical so this is a fair baseline.

Architecture (domain-adversarial network):
  Extractor       : per-position features (window of W positions, C channels)
                    -> 2 stacked unidirectional LSTMs -> 128*W embedding
  ClassClassifier : embedding -> modified/unmodified (NLL) + stoichiometry (MSE)
  DomainClassifier: [embedding, stoichiometry] -> gradient reversal -> which
                    modification "domain" (NLL). The reversal forces the
                    embedding to be invariant to modification type, which is
                    how ORCA generalizes to unseen modifications.

The released model uses W=5 positions and C=56 features/position (Extractor
reshapes to 128*5). Both are constructor args here so we can match whatever
Bhargav's featurization produces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """Gradient reversal layer (identity forward, negated-and-scaled backward)."""
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None


def grad_reverse(x, constant):
    return GradReverse.apply(x, constant)


class Extractor(nn.Module):
    def __init__(self, in_ch: int = 56, window: int = 5):
        super().__init__()
        self.window = window
        self.rnns = nn.ModuleList([
            nn.LSTM(in_ch, in_ch, 1, bidirectional=False),
            nn.LSTM(in_ch, 128, 2, bidirectional=False),
        ])

    def forward(self, x):                       # x: (batch, window, in_ch)
        x = x.permute(1, 0, 2)                  # (window, batch, in_ch)
        for rnn in self.rnns:
            x, _ = rnn(x)
            x = x.flip([0])                     # ORCA flips sequence between layers
        x = x.permute(1, 2, 0)                  # (batch, 128, window)
        x = x.reshape(x.size(0), 128 * self.window)
        return x


class ClassClassifier(nn.Module):
    """Two heads off the shared embedding: presence (CE) + stoichiometry (MSE)."""
    def __init__(self, window: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(128 * window, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)            # modified vs unmodified
        self.fc4 = nn.Linear(128, 1)            # stoichiometry (0..1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(F.dropout(h))
        h = F.relu(h)
        ce = F.log_softmax(self.fc3(h), dim=1)  # for NLLLoss
        mse = self.fc4(h)[:, 0]                 # stoichiometry prediction
        return ce, mse


class DomainClassifier(nn.Module):
    """Adversarial head: predicts modification/domain from [embedding, stoich]."""
    def __init__(self, n_mod: int, window: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(128 * window + 1, 128)
        self.fc3 = nn.Linear(128, n_mod)

    def forward(self, x, constant):
        x = grad_reverse(x, constant)
        h = F.relu(self.fc1(x))
        return F.log_softmax(self.fc3(h), dim=1)


class ORCA(nn.Module):
    """Convenience wrapper tying the three heads together."""
    def __init__(self, in_ch: int = 56, window: int = 5, n_mod: int = 4):
        super().__init__()
        self.extractor = Extractor(in_ch, window)
        self.classifier = ClassClassifier(window)
        self.domain = DomainClassifier(n_mod, window)

    def forward(self, x, grl_lambda: float = 0.0):
        feat = self.extractor(x)                     # (batch, 128*window)
        ce, mse = self.classifier(feat)              # presence + stoichiometry
        dom_in = torch.cat([feat, mse.unsqueeze(1)], dim=1)  # append stoich (the +1)
        dom = self.domain(dom_in, grl_lambda)        # adversarial domain logits
        return ce, mse, dom
