#!/usr/bin/env python3
"""
PURE one-class Deep SVDD (Ruff et al. 2018) — the maximally chemistry-agnostic test.

Unlike Deep SAD (run_matched_loco.py with SAD_DIM>0), this trains on UNMODIFIED
data ONLY: the model never sees a single modification during training. It learns a
compact "normal" hypersphere around a fixed centre c; the anomaly score at test is
||f(x) - c||. If a held-out chemistry is detectable here, detection required NO
modification supervision at all — the strongest possible modification-agnostic claim.

Collapse guard (the known Deep SVDD failure = mapping everything to c, loss->0 but
useless): bias-free head, c fixed from an init pass, and MODEL SELECTION on a val
anomaly-AUROC (not the training loss) using held-out controls + a sample of SEEN-
chemistry positives (never the held-out chemistry). Embedding std is logged so a
collapse is visible at once.

Fold loco_<CHEM>:
  train      = unmodified controls only (position-split, held-out chemistry irrelevant
               since no positives are used in training)
  val (sel.) = held-out controls  +  sampled seen-chemistry positives (monitor only)
  test       = held-out-chemistry positives  +  ref-base-matched controls
  score      = ||sad_head(rep) - c||  (distance from normal)

Usage: python run_svdd_loco.py --fold loco_5hmU --out-dir <dir> [--epochs N]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / 'experiments' / 'pipeline1', REPO / 'deepmod'):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(HERE))

import run_pipeline as R
import run_matched_loco as rml                    # reuse pool assembly + split logic
from run_convformer_v2 import ConvFormerV2
from mod_types import build_umces_mod_map
from model import PileupDataset, make_loader_kwargs, _worker_init_fn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loader_for(pool, idx, batch=512, shuffle=False):
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=shuffle, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    return DataLoader(ds, shuffle=shuffle,
                      **make_loader_kwargs(batch, 6, device, _worker_init_fn))


def sad_embed(model, pool, idx):
    """Stream idx through the model; return the (N, sad_dim) SAD embeddings."""
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in loader_for(pool, idx):
            model(xb.to(device, non_blocking=True))
            out.append(model._sad.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)          # loco_<CHEM>
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    chem_x = a.fold[len('loco_'):]
    assert chem_x in rml.CHEMS, f"bad fold {a.fold}"

    hp = R.HP(); R.set_seed(hp.seed)
    sad_dim = int(os.environ.get('SAD_DIM', '32'))
    out = Path(a.out_dir); (out / 'metrics').mkdir(parents=True, exist_ok=True)
    (out / 'models' / a.fold).mkdir(parents=True, exist_ok=True)

    # ── assemble matched pool + typing (reused helpers) ─────────────────────────
    members = rml.build_members(); pool = R.Group(list(members), members)
    mod_map = build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = rml.ref_base_center(pool)
    chem = rml.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    neg = rml.subsample_negatives(pool, seed=hp.seed)
    neg_mask = np.zeros(pool.N, bool); neg_mask[neg] = True

    # controls: position-split; test negatives ref-base + organism matched
    ctrl_idx = np.nonzero(neg_mask)[0]
    tr_ctrl, te_ctrl = rml.pos_hash_split(pool, ctrl_idx, 0.15, hp.seed)
    bases = rml.CHEM_BASES[chem_x]; orgs = rml.CHEM_ORGS[chem_x]
    te_ctrl = np.array([i for i in te_ctrl
                        if rml.org_of(pool.names[int(pool.file_of[i])]) in orgs
                        and refbase[i] in bases], dtype=np.int64)
    pos_x = np.nonzero(is_pos & (chem == chem_x))[0].astype(np.int64)
    pos_other = np.nonzero(is_pos & (chem != chem_x) & (chem != '')
                           & (chem != 'untyped'))[0].astype(np.int64)

    # train = UNMODIFIED only; carve a val split of controls for early stopping,
    # plus a monitor sample of SEEN-chemistry positives (selection only, no grads)
    rng = np.random.default_rng(hp.seed)
    tr_ctrl = rng.permutation(tr_ctrl)
    n_val = max(2000, int(0.1 * len(tr_ctrl)))
    val_ctrl, train_ctrl = tr_ctrl[:n_val], tr_ctrl[n_val:]
    val_pos = rng.choice(pos_other, min(3000, len(pos_other)), replace=False)
    test_idx = np.sort(np.concatenate([pos_x, te_ctrl]))
    print(f"[SVDD {a.fold}] train_ctrl={len(train_ctrl):,} (UNMOD only)  "
          f"val_ctrl={len(val_ctrl):,} val_pos(seen)={len(val_pos):,}  "
          f"test: pos_{chem_x}={len(pos_x):,} neg={len(te_ctrl):,}  sad_dim={sad_dim}",
          flush=True)
    if len(pos_x) == 0 or len(te_ctrl) == 0:
        raise SystemExit("empty test")

    # ── model + fixed centre from init pass over training controls ──────────────
    model = ConvFormerV2(dropout=hp.dropout, sad_dim=sad_dim).to(device)
    e0 = sad_embed(model, pool, train_ctrl[:8000])
    c = torch.tensor(e0.mean(0), device=device)
    c[c.abs() < 1e-6] = 1e-6
    model.sad_center.copy_(c)
    print(f"  centre ||c||={float(c.norm()):.3f} from {min(8000,len(train_ctrl)):,} "
          f"normals; init embed std={e0.std():.3f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    tr_loader = loader_for(pool, train_ctrl, shuffle=True)

    def val_auroc():
        dv = sad_embed(model, pool, val_ctrl); dp = sad_embed(model, pool, val_pos)
        sc = np.concatenate([np.linalg.norm(dv - c.cpu().numpy(), axis=1),
                             np.linalg.norm(dp - c.cpu().numpy(), axis=1)])
        y = np.concatenate([np.zeros(len(dv)), np.ones(len(dp))])
        std = np.concatenate([dv, dp]).std()
        return roc_auc_score(y, sc), std

    best_auroc, best_state, patience = -1.0, None, 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = n = 0
        for xb, _ in tr_loader:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad()
            _, aux = model(xb)
            loss = ((aux['sad'] - model.sad_center) ** 2).sum(1).mean()  # pull to c
            loss.backward(); opt.step()
            tot += float(loss) * len(xb); n += len(xb)
        va, std = val_auroc()
        print(f"  ep {ep:3d}/{a.epochs}  svdd_loss={tot/max(n,1):.4f}  "
              f"val_anom_AUROC={va:.4f}  embed_std={std:.3f}  {time.time()-t0:.1f}s",
              flush=True)
        if va > best_auroc:
            best_auroc, patience = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                print(f"  early stop @ ep {ep}", flush=True); break

    model.load_state_dict(best_state)
    torch.save({'model_state': best_state, 'in_channels': 11,
                'val_auprc': best_auroc, 'epoch': ep, 'tag': 'svdd'},
               out / 'models' / a.fold / 'best_model.pt')

    # ── test: anomaly AUROC on the held-out chemistry ───────────────────────────
    et = sad_embed(model, pool, test_idx)
    score = np.linalg.norm(et - c.cpu().numpy(), axis=1)
    y = (pool.labels[test_idx] > 0).astype(int)
    auroc = roc_auc_score(y, score)
    base = int(y.sum()) / len(y)
    print(f"\n[SVDD {a.fold}] zero-shot anomaly AUROC={auroc:.4f} "
          f"(val_sel={best_auroc:.3f}, base={base:.3f}, n_pos={int(y.sum())})", flush=True)

    tsv = out / 'metrics' / f'{a.fold}.tsv'
    with open(tsv, 'w') as fh:
        fh.write("fold\ttest_set\theld_out\tauroc_svdd\tval_auroc\tbase_rate\tn_pos\tn_test\n")
        fh.write(f"{a.fold}\tzeroshot_{chem_x}\t{chem_x}\t{auroc:.6f}\t{best_auroc:.6f}\t"
                 f"{base:.6f}\t{int(y.sum())}\t{len(y)}\n")
    print(f"wrote {tsv}\nDONE [{a.fold}]", flush=True)


if __name__ == '__main__':
    main()
