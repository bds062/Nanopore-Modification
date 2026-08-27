#!/usr/bin/env python3
"""
Leave-one-modification-out training of ORCA's actual model (LSTM extractor +
gradient-reversal domain head, no stoichiometry) on ORCA's EXACT features
(the merged.feature.per.site tables from orca-pred_feature_merge).

This is the "true baseline": ORCA architecture on ORCA features, so it can be
compared head-to-head against our DeepMod model.

Each merged CSV row = one site: id, position, kmer, depth, then a 5-position
window (-2..+2) x 56 features (3 error ratios + 3 qual stats + 50 signal shape).
Labels: modified (1) if (id, position) is in the modification's GT BED, else 0.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

from model import Extractor, grad_reverse   # ORCA architecture

# avoid a oneDNN "could not create a primitive" error on some torch builds (LSTM CPU)
torch.backends.mkldnn.enabled = False

MODS = ["control", "5mC", "5hmC", "6mA"]
IDX = {"id", "position", "kmer", "depth"}


class ORCAcls(nn.Module):
    """ORCA extractor + binary class head + gradient-reversal domain head."""
    def __init__(self, in_ch=56, window=5, n_domains=4, dropout=0.3):
        super().__init__()
        self.extractor = Extractor(in_ch, window)
        d = 128 * window
        self.cls = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(128, 2))
        self.dom = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                                 nn.Linear(128, n_domains))

    def forward(self, x, lam=0.0):
        f = self.extractor(x)
        ce = F.log_softmax(self.cls(f), dim=1)
        dm = F.log_softmax(self.dom(grad_reverse(f, lam)), dim=1)
        return ce, dm

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def load_condition(feat_dir, cond, mod_name, gt_dir):
    """Return X (N,5,56), y (N,), and the site ids for one condition."""
    csv = os.path.join(feat_dir, cond, f"{cond}.merged.feature.per.site")
    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c not in IDX]
    assert len(feat_cols) % 5 == 0, f"{len(feat_cols)} feat cols not divisible by 5"
    C = len(feat_cols) // 5
    X = df[feat_cols].to_numpy(np.float32).reshape(-1, 5, C)
    X = np.nan_to_num(X)

    gt = set()
    if mod_name != "control":
        bed = os.path.join(gt_dir, f"all_5mers_{mod_name}_sites.bed")
        with open(bed) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    gt.add((p[0], int(p[1])))
    y = np.array([1 if (r.id, int(r.position)) in gt else 0
                  for r in df.itertuples()], dtype=np.int64)
    return X, y, C


def grl_lambda(step, total, gamma=10.0):
    p = step / max(1, total)
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


METRICS = ["precision", "recall", "f1", "auprc", "pos_rate", "lift", "prec_at_r30",
           "rank99", "eff_rank"]


def prec_at_recall(y, p, target=0.30):
    """Precision at the lowest threshold that still reaches `target` recall.

    ORCA reports precision in the main paper at a recall of roughly 0.25-0.40
    (their supplementary recall panels), while our class-weighted loss puts us
    at recall 0.7+. Comparing precision at a fixed 0.5 threshold therefore
    compares two different operating points; this reads precision off the PR
    curve at ORCA's recall instead.
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, p)
    ok = rec >= target
    return float(prec[ok].max()) if ok.any() else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5, return_probs=False):
    model.eval(); probs, ys = [], []
    for x, y, _ in loader:
        ce, _ = model(x.to(device), 0.0)
        probs.append(ce[:, 1].exp().cpu().numpy()); ys.append(y.numpy())
    p = np.concatenate(probs); y = np.concatenate(ys)
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 average_precision_score)
    if y.sum() == 0:
        out = {k: float("nan") for k in METRICS}
        return (out, p, y) if return_probs else out
    pred = (p > thr).astype(int)
    # a random ranker scores AUPRC == the positive rate, so lift says how much
    # of the apparent "low AUPRC" is just class imbalance
    pos_rate = float(y.mean())
    auprc = average_precision_score(y, p)
    out = {"precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0),
           "f1": f1_score(y, pred, zero_division=0),
           "auprc": auprc,
           "pos_rate": pos_rate,
           "lift": auprc / pos_rate if pos_rate > 0 else float("nan"),
           "prec_at_r30": prec_at_recall(y, p, 0.30)}
    return (out, p, y) if return_probs else out


@torch.no_grad()
def embedding_rank(model, loader, device, var=0.99, cap=4000):
    """Diagnose adversarial collapse of the shared embedding.

    A gradient-reversal head that is too strong drives the extractor output
    onto a low-dimensional subspace, which looks like good domain confusion but
    destroys the class signal. Returns how many principal directions carry
    `var` of the variance, plus the entropy-based effective rank, out of 128*5
    possible dimensions.
    """
    model.eval(); feats = []
    n = 0
    for x, _, _ in loader:
        feats.append(model.extractor(x.to(device)).cpu().numpy())
        n += len(feats[-1])
        if n >= cap:
            break
    f = np.concatenate(feats)[:cap]
    f = f - f.mean(0, keepdims=True)
    s = np.linalg.svd(f, compute_uv=False)
    ev = s ** 2
    if ev.sum() <= 0:
        return {"rank99": 0, "eff_rank": 0.0, "dim": f.shape[1]}
    ratio = ev / ev.sum()
    p = ratio[ratio > 0]
    return {"rank99": int(np.searchsorted(np.cumsum(ratio), var) + 1),
            "eff_rank": float(np.exp(-(p * np.log(p)).sum())),
            "dim": int(f.shape[1])}


def train_one(held, data, args, device, seed=0):
    import copy
    train_mods = [m for m in MODS if m != held]
    dom_ids = {m: i for i, m in enumerate(train_mods)}
    C = data[MODS[0]][2]

    Xtr = np.concatenate([data[m][0] for m in train_mods])
    ytr = np.concatenate([data[m][1] for m in train_mods])
    dtr = np.concatenate([np.full(len(data[m][1]), dom_ids[m]) for m in train_mods])
    Xte, yte = data[held][0], data[held][1]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    nval = max(args.batch, len(Xtr) // 10)
    vi, ti = perm[:nval], perm[nval:]

    def dl(X, y, d, idx=None, shuffle=False):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        d = torch.tensor(d, dtype=torch.long)
        ds = TensorDataset(X, y, d)
        if idx is not None:
            ds = Subset(ds, idx)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, drop_last=shuffle)

    dl_tr = dl(Xtr, ytr, dtr, ti, shuffle=True)
    dl_val = dl(Xtr, ytr, dtr, vi, shuffle=False)
    dl_te = dl(Xte, yte, np.zeros(len(yte)), shuffle=False)

    npos = int(ytr[ti].sum())
    pos_w = torch.tensor([(len(ti) - npos) / max(1, npos)], dtype=torch.float32).to(device)
    torch.manual_seed(seed)
    model = ORCAcls(in_ch=C, window=5, n_domains=len(train_mods)).to(device)
    print(f"[hold={held} seed={seed}] train {len(ti)} ({npos} pos), val {len(vi)}, "
          f"test {len(yte)} ({int(yte.sum())} pos), params={model.n_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    nll = nn.NLLLoss(weight=torch.tensor([1.0, float(pos_w)]).to(device))
    ce_dom = nn.NLLLoss()

    total = args.epochs * max(1, len(dl_tr)); step = 0
    best_val = -1.0; best_state = None
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y, d in dl_tr:
            x, y, d = x.to(device), y.to(device), d.to(device)
            lam = args.grl_max * grl_lambda(step, total)
            ce, dm = model(x, lam)
            loss = nll(ce, y) + args.dom_weight * ce_dom(dm, d)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        vm = evaluate(model, dl_val, device)
        if vm["auprc"] == vm["auprc"] and vm["auprc"] > best_val:
            best_val = vm["auprc"]; best_state = copy.deepcopy(model.state_dict())
        print(f"  [{held}] ep {ep:2d} val_AUPRC={vm['auprc']:.3f} (best {best_val:.3f})")
    if best_state:
        model.load_state_dict(best_state)
    er = embedding_rank(model, dl_val, device)
    print(f"  [{held}] embedding rank99={er['rank99']}/{er['dim']} "
          f"eff_rank={er['eff_rank']:.1f} (low values = GRL collapse)")
    m, p, y = evaluate(model, dl_te, device, return_probs=True)
    m["rank99"], m["eff_rank"] = er["rank99"], er["eff_rank"]
    # keep the raw scores so precision/recall can be re-read at any operating
    # point later without retraining
    np.savez_compressed(os.path.join(args.out_dir, f"probs_{held}_seed{seed}.npz"),
                        probs=p, labels=y)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="/fs/nexus-scratch/vgandhi/orca_feat")
    ap.add_argument("--gt-dir",
                    default="/fs/nexus-scratch/bds062/data/ont-os/references")
    ap.add_argument("--suffix", default="_rep1", help="condition suffix, e.g. _rep1")
    ap.add_argument("--out-dir", default="/fs/nexus-scratch/vgandhi/orca_true_baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grl-max", type=float, default=1.0,
                    help="cap on the gradient-reversal lambda. The v3 run at 1.0 "
                         "collapsed the embedding to rank99=10-25 of 640, so lower "
                         "values trade domain invariance for a usable representation")
    ap.add_argument("--dom-weight", type=float, default=1.0,
                    help="weight on the domain (adversarial) loss term")
    ap.add_argument("--holdouts", default="5mC,5hmC,6mA")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data = {}
    for m in MODS:
        cond = m + args.suffix
        X, y, C = load_condition(args.feat_dir, cond, m, args.gt_dir)
        data[m] = (X, y, C)
        print(f"  {cond}: {len(y)} sites, {int(y.sum())} positive, feat {X.shape}")

    seeds = [int(s) for s in args.seeds.split(",")]
    agg = {}
    for held in args.holdouts.split(","):
        runs = [train_one(held, data, args, device, seed=s) for s in seeds]
        agg[held] = {k: (float(np.nanmean([r[k] for r in runs])),
                         float(np.nanstd([r[k] for r in runs])))
                     for k in METRICS}
        print(f"### {held}: P={agg[held]['precision'][0]:.3f} "
              f"R={agg[held]['recall'][0]:.3f} "
              f"AUPRC={agg[held]['auprc'][0]:.3f} "
              f"(pos_rate={agg[held]['pos_rate'][0]:.4f}, "
              f"lift={agg[held]['lift'][0]:.1f}x, "
              f"P@R30={agg[held]['prec_at_r30'][0]:.3f})\n")

    with open(os.path.join(args.out_dir, "lodo_metrics.tsv"), "w") as fh:
        fh.write("held_out\tprecision\tprecision_std\tauprc\tauprc_std\trecall\tf1"
                 "\tpos_rate\tlift\tprec_at_r30\trank99\teff_rank\n")
        for h, m in agg.items():
            fh.write(f"{h}\t{m['precision'][0]:.4f}\t{m['precision'][1]:.4f}\t"
                     f"{m['auprc'][0]:.4f}\t{m['auprc'][1]:.4f}\t"
                     f"{m['recall'][0]:.4f}\t{m['f1'][0]:.4f}\t"
                     f"{m['pos_rate'][0]:.4f}\t{m['lift'][0]:.2f}\t"
                     f"{m['prec_at_r30'][0]:.4f}\t{m['rank99'][0]:.1f}\t"
                     f"{m['eff_rank'][0]:.1f}\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        hs = list(agg.keys())
        precs = [agg[h]["precision"][0] for h in hs]
        errs = [agg[h]["precision"][1] for h in hs]
        plt.figure(figsize=(6, 4))
        plt.bar(hs, precs, yerr=errs, capsize=5, color="#E45756")
        plt.ylim(0, 1); plt.ylabel("Precision (held-out)")
        plt.title(f"ORCA true baseline: leave-one-out ({len(seeds)} seeds)")
        for i, p in enumerate(precs):
            plt.text(i, p + errs[i] + 0.02, f"{p:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "lodo_precision.png"), dpi=150)
        print("Saved lodo_precision.png + lodo_metrics.tsv")
    except Exception as e:
        print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
