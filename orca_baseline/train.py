#!/usr/bin/env python3
"""
Train the ORCA domain-adversarial model on DNA modification features, as a
baseline for the DeepMod paper. Faithful to the ORCA architecture; the paper's
undocumented training details are set to standard DANN defaults and flagged.

Hyperparameters
---------------
Documented in the ORCA paper / code (matched):
  - AdamW, learning rate 5e-4
  - losses: NLL (presence) + NLL (domain) + MSE (stoichiometry)
  - positive label = position with >10% modified reads
  - 4:1 train/test split; leave-one-modification-out for zero-shot eval

NOT documented in the paper (set to standard DANN defaults — flagged as ASSUMED):
  - GRL lambda schedule: 2/(1+exp(-gamma*p)) - 1, gamma=10  [ASSUMED]
  - loss weighting between the three heads: equal (1:1:1)     [ASSUMED]
  - batch size, epochs                                        [ASSUMED]

Data loader
-----------
`load_features()` is the ONE piece to wire to Bhargav's actual feature format
once he sends the path/layout. Expected per-sample content:
  X        : float32 (N, window, channels)   per-position features
  y_mod    : int64   (N,)  0=unmodified, 1=modified   (>10% reads => 1)
  y_stoich : float32 (N,)  modification rate in [0,1]
  y_domain : int64   (N,)  modification-type index (control=0, 5mC, 5hmC, 6mA ...)
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import ORCA

MODS = ["control", "5mC", "5hmC", "6mA"]   # domain order; adjust to Bhargav's


# ── data loader (WIRE TO BHARGAV'S FORMAT) ────────────────────────────────────
def load_features(path):
    """Return X, y_mod, y_stoich, y_domain as numpy arrays.

    Placeholder: supports a single .npz with keys X, y_mod, y_stoich, y_domain,
    or a directory of per-modification .npz files. Replace/extend once Bhargav
    confirms the actual layout.
    """
    if path.endswith(".npz"):
        d = np.load(path)
        return d["X"], d["y_mod"], d["y_stoich"], d["y_domain"]
    if os.path.isdir(path):
        Xs, ym, ys, yd = [], [], [], []
        for i, mod in enumerate(MODS):
            f = os.path.join(path, f"{mod}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            Xs.append(d["X"]); ym.append(d["y_mod"])
            ys.append(d["y_stoich"]); yd.append(np.full(len(d["X"]), i))
        return (np.concatenate(Xs), np.concatenate(ym),
                np.concatenate(ys), np.concatenate(yd))
    raise NotImplementedError(
        f"Wire load_features() to Bhargav's format for: {path}")


def grl_lambda(step, total, gamma=10.0):
    p = step / max(1, total)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0   # [ASSUMED] standard DANN


def split(n, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * frac)
    return idx[:k], idx[k:]


def evaluate(model, loader, device):
    model.eval()
    correct = tot = 0
    probs, labels = [], []
    with torch.no_grad():
        for X, ym, ys, yd in loader:
            X = X.to(device)
            ce, mse, _ = model(X, 0.0)
            pred = ce.argmax(1).cpu()
            correct += (pred == ym).sum().item(); tot += len(ym)
            probs.append(ce[:, 1].exp().cpu()); labels.append(ym)
    probs = torch.cat(probs).numpy(); labels = torch.cat(labels).numpy()
    try:
        from sklearn.metrics import average_precision_score, f1_score
        auprc = average_precision_score(labels, probs) if labels.sum() else float("nan")
        f1 = f1_score(labels, (probs > 0.5).astype(int)) if labels.sum() else float("nan")
    except Exception:
        auprc = f1 = float("nan")
    return correct / max(1, tot), auprc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="path to Bhargav's features")
    ap.add_argument("--out-dir", default="orca_dna_out")
    ap.add_argument("--epochs", type=int, default=50)          # [ASSUMED]
    ap.add_argument("--batch", type=int, default=256)          # [ASSUMED]
    ap.add_argument("--lr", type=float, default=5e-4)          # documented
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--channels", type=int, default=56)
    ap.add_argument("--leave-out", default=None,
                    help="modification to hold out for zero-shot eval, e.g. 6mA")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    X, y_mod, y_stoich, y_domain = load_features(args.features)
    print(f"Loaded X{X.shape}  mod+={int(y_mod.sum())}/{len(y_mod)}  "
          f"domains={sorted(set(y_domain.tolist()))}")

    # optional leave-one-modification-out (zero-shot) split
    test_mask = None
    if args.leave_out is not None:
        d = MODS.index(args.leave_out)
        test_mask = (y_domain == d)
        print(f"Zero-shot: holding out {args.leave_out} (domain {d}), "
              f"{int(test_mask.sum())} samples")

    Xt = torch.tensor(X, dtype=torch.float32)
    tm = torch.tensor(y_mod, dtype=torch.long)
    ts = torch.tensor(y_stoich, dtype=torch.float32)
    td = torch.tensor(y_domain, dtype=torch.long)

    if test_mask is not None:
        tr = ~test_mask
        tr_idx = np.where(tr)[0]; te_idx = np.where(test_mask)[0]
        va_idx = tr_idx[:len(tr_idx) // 5]; tr_idx = tr_idx[len(tr_idx) // 5:]
    else:
        idx_tr, idx_te = split(len(X), 0.8, args.seed)     # 4:1 documented
        va_idx = idx_tr[:len(idx_tr) // 5]; tr_idx = idx_tr[len(idx_tr) // 5:]
        te_idx = idx_te

    def make(idx, shuffle):
        return DataLoader(TensorDataset(Xt[idx], tm[idx], ts[idx], td[idx]),
                          batch_size=args.batch, shuffle=shuffle)
    dl_tr, dl_va, dl_te = make(tr_idx, True), make(va_idx, False), make(te_idx, False)

    n_mod = len(set(y_domain.tolist()))
    model = ORCA(in_ch=args.channels, window=args.window, n_mod=n_mod).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)      # documented
    nll = nn.NLLLoss(); mse = nn.MSELoss()

    total_steps = args.epochs * max(1, len(dl_tr))
    step = 0; best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X_b, ym_b, ys_b, yd_b in dl_tr:
            X_b, ym_b = X_b.to(device), ym_b.to(device)
            ys_b, yd_b = ys_b.to(device), yd_b.to(device)
            lam = grl_lambda(step, total_steps)
            ce, pred_s, dom = model(X_b, lam)
            loss = nll(ce, ym_b) + mse(pred_s, ys_b) + nll(dom, yd_b)  # [ASSUMED 1:1:1]
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
        acc, auprc, f1 = evaluate(model, dl_va, device)
        print(f"epoch {epoch:3d}  val acc={acc:.3f}  AUPRC={auprc:.3f}  F1={f1:.3f}  "
              f"grl_lambda={lam:.2f}")
        if auprc == auprc and auprc > best:      # not-NaN and improved
            best = auprc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pt")))
    acc, auprc, f1 = evaluate(model, dl_te, device)
    tag = f"zero-shot({args.leave_out})" if args.leave_out else "test"
    print(f"\n=== {tag}: acc={acc:.3f}  AUPRC={auprc:.3f}  F1={f1:.3f} ===")


if __name__ == "__main__":
    main()
