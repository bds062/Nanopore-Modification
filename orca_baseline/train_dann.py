#!/usr/bin/env python3
"""
Leave-one-modification-out training of the ORCA-style domain-adversarial model
(CNN trunk -> bi-LSTM -> gradient-reversal domain head) on DeepMod pileup
features. Reproduces ORCA Fig 3b (validation 1): hold out one modification,
train on the rest, measure precision on the held-out modification.

Inputs: one HDF5 per condition (control.h5, 5mC.h5, 5hmC.h5, 6mA.h5), each with
  /tensors  (N, H, W, C)  pileup images
  /labels   (N,)          0/1 modified label
  attrs: W (window positions), n_channels
"""
import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dann_model import DANN


class H5Pileup(Dataset):
    """Lazy per-item HDF5 reader. domain = index of the modification/file."""
    def __init__(self, files, domain_ids, cap=None, seed=0):
        import h5py
        self.files = files
        self.domain_ids = domain_ids
        self._h = [None] * len(files)
        self.entries = []
        rng = np.random.default_rng(seed)
        for fi, f in enumerate(files):
            with h5py.File(f, "r") as hf:
                n = hf["labels"].shape[0]
            idx = np.arange(n)
            if cap and n > cap:
                idx = rng.choice(n, cap, replace=False)
            self.entries += [(fi, int(i)) for i in idx]

    def handle(self, fi):
        import h5py
        if self._h[fi] is None:
            self._h[fi] = h5py.File(self.files[fi], "r")
        return self._h[fi]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        fi, li = self.entries[i]
        hf = self.handle(fi)
        x = np.asarray(hf["tensors"][li], dtype=np.float32)   # (H, W, C)
        y = np.float32(hf["labels"][li])
        x = np.transpose(x, (2, 0, 1))                        # (C, H, W)
        return torch.from_numpy(x), torch.tensor(y), torch.tensor(self.domain_ids[fi])


def h5_attrs(path):
    import h5py
    with h5py.File(path, "r") as hf:
        W = int(hf.attrs.get("W", 21))
        C = int(hf.attrs.get("n_channels", 9))
        n = hf["labels"].shape[0]
        pos = int(np.asarray(hf["labels"]).sum())
    return W, C, n, pos


def grl_lambda(step, total, gamma=10.0):
    p = step / max(1, total)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    probs, ys = [], []
    for x, y, _ in loader:
        logit, _ = model(x.to(device), 0.0)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.numpy())
    p = np.concatenate(probs); y = np.concatenate(ys)
    pred = (p > thr).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
    out = {}
    if y.sum() > 0:
        out["precision"] = precision_score(y, pred, zero_division=0)
        out["recall"] = recall_score(y, pred, zero_division=0)
        out["f1"] = f1_score(y, pred, zero_division=0)
        out["auprc"] = average_precision_score(y, p)
    else:
        out = {k: float("nan") for k in ["precision", "recall", "f1", "auprc"]}
    return out


def train_one(held, files, mods, args, device):
    """Train holding out `held`; test on it. Returns held-out metrics."""
    train_files = [f for f, m in zip(files, mods) if m != held]
    train_mods = [m for m in mods if m != held]
    test_file = files[mods.index(held)]
    dom_ids = {m: i for i, m in enumerate(train_mods)}
    n_dom = len(train_mods)

    W, C, _, _ = h5_attrs(files[0])
    ds_tr = H5Pileup(train_files, [dom_ids[m] for m in train_mods], cap=args.cap)
    ds_te = H5Pileup([test_file], [0], cap=args.cap)

    # pos_weight from the training label balance
    ntr = len(ds_tr); npos = sum(1 for e in ds_tr.entries
                                 if float(ds_tr.handle(e[0])["labels"][e[1]]) > 0)
    pos_w = torch.tensor([(ntr - npos) / max(1, npos)], dtype=torch.float32).to(device)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = DANN(in_ch=C, window=W, n_domains=n_dom).to(device)
    print(f"[hold={held}] train {ntr} ({npos} pos), domains={train_mods}, "
          f"model params={model.n_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ce = nn.CrossEntropyLoss()

    total = args.epochs * max(1, len(dl_tr)); step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y, d in dl_tr:
            x, y, d = x.to(device), y.to(device), d.to(device)
            lam = grl_lambda(step, total)
            logit, dom = model(x, lam)
            loss = bce(logit, y) + ce(dom, d)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
        m = evaluate(model, dl_te, device)
        print(f"  [{held}] epoch {ep:2d}  P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} AUPRC={m['auprc']:.3f}  grl={lam:.2f}")
    return evaluate(model, dl_te, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir with {control,5mC,5hmC,6mA}.h5")
    ap.add_argument("--out-dir", default="orca_dann_out")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--cap", type=int, default=40000, help="max images per file")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--holdouts", default="5mC,5hmC,6mA",
                    help="modifications to leave out, comma-separated")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    mods = ["control", "5mC", "5hmC", "6mA"]
    files = [os.path.join(args.features, f"{m}.h5") for m in mods]
    for f in files:
        assert os.path.exists(f), f"missing {f}"
        W, C, n, pos = h5_attrs(f)
        print(f"  {os.path.basename(f)}: {n} images, {pos} positive, W={W}, C={C}")

    results = {}
    for held in args.holdouts.split(","):
        results[held] = train_one(held, files, mods, args, device)
        print(f"=== leave-out {held}: {results[held]} ===\n")

    # Fig 3b-style summary (precision per held-out modification)
    with open(os.path.join(args.out_dir, "lodo_metrics.tsv"), "w") as fh:
        fh.write("held_out\tprecision\trecall\tf1\tauprc\n")
        for h, m in results.items():
            fh.write(f"{h}\t{m['precision']:.4f}\t{m['recall']:.4f}\t"
                     f"{m['f1']:.4f}\t{m['auprc']:.4f}\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        hs = list(results.keys()); precs = [results[h]["precision"] for h in hs]
        plt.figure(figsize=(6, 4))
        plt.bar(hs, precs, color="#4C78A8")
        plt.ylim(0, 1); plt.ylabel("Precision (held-out)")
        plt.title("Leave-one-modification-out (ORCA Fig 3b analog)")
        for i, p in enumerate(precs):
            plt.text(i, p + 0.02, f"{p:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "lodo_precision.png"), dpi=150)
        print("Saved lodo_precision.png + lodo_metrics.tsv")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
