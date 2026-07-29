#!/usr/bin/env python3
"""
DIAGNOSTIC (no retraining): does an ANOMALY decision rule — score = distance from
the UNMODIFIED cluster centre — beat the classifier logit at detecting a held-out
chemistry? Tests the one-class hypothesis on the already-trained results4 (SupCon +
curriculum) LOCO checkpoints before investing in a full Deep-SAD retrain.

For each loco_<CHEM> fold:
  centre c  = mean embedding over the fold's TRAINING controls (unmodified)
  score(x)  = distance(embedding(x), c)      [cosine and euclidean]
  compare AUROC(score) vs AUROC(logit) on the held-out chemistry test set,
  in both the 96-d penultimate space (rep) and the 128-d SupCon projection.

Reuses the pipeline4 pool assembly and pipeline3 checkpoint loader.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / 'experiments' / 'pipeline1', REPO / 'experiments' / 'pipeline3',
          REPO / 'deepmod'):
    sys.path.insert(0, str(p))

# load run_matched_loco as a module to reuse its pool-assembly helpers
spec = importlib.util.spec_from_file_location('rml', HERE / 'run_matched_loco.py')
rml = importlib.util.module_from_spec(spec); spec.loader.exec_module(rml)

import run_pipeline as R
from mod_types import build_umces_mod_map
from model import PileupDataset, make_loader_kwargs, _worker_init_fn
from score_genome import load_model
from torch.utils.data import DataLoader

MODELS = '/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results4/models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embed_and_logit(model, paths, sizes, idx):
    """Return (rep[N,96], proj[N,128], logit[N]) for image indices idx."""
    ds = PileupDataset(paths, np.asarray(idx, np.int64), sizes, augment=False,
                       seed=0, signal_noise_std=0.0, delta_channels=True,
                       preload=False)
    lk = make_loader_kwargs(512, 6, device, _worker_init_fn)
    loader = DataLoader(ds, shuffle=False, **lk)
    cap = {}
    h = model.head.register_forward_hook(lambda m, i, o: cap.__setitem__('rep', i[0].detach()))
    reps, projs, logits = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            lg = model(xb)
            rep = cap['rep']
            reps.append(rep.float().cpu().numpy())
            projs.append(F.normalize(model.proj(rep), dim=1).float().cpu().numpy())
            logits.append(lg.squeeze(-1).float().cpu().numpy())
    h.remove()
    return (np.concatenate(reps), np.concatenate(projs), np.concatenate(logits))


def auroc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float('nan')


def main():
    hp = R.HP()
    members = rml.build_members()
    pool = R.Group(list(members), members)
    mod_map = build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = rml.ref_base_center(pool)
    chem = rml.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    neg = rml.subsample_negatives(pool, seed=hp.seed)
    neg_mask = np.zeros(pool.N, bool); neg_mask[neg] = True
    ctrl_idx = np.nonzero(neg_mask)[0]
    tr_ctrl0, te_ctrl0 = rml.pos_hash_split(pool, ctrl_idx, 0.15, hp.seed)

    print(f"{'chem':5s} | {'logit':>6s} {'rep-cos':>7s} {'rep-euc':>7s} "
          f"{'proj-cos':>8s} {'proj-euc':>8s}  (AUROC)")
    print('-' * 60)
    for cx in rml.CHEMS:
        ckpt = f'{MODELS}/loco_{cx}/loco/best_model.pt'
        if not Path(ckpt).exists():
            print(f"{cx:5s}  missing checkpoint"); continue
        bases = rml.CHEM_BASES[cx]; orgs = rml.CHEM_ORGS[cx]
        te_ctrl = np.array([i for i in te_ctrl0
                            if rml.org_of(pool.names[int(pool.file_of[i])]) in orgs
                            and refbase[i] in bases], dtype=np.int64)
        pos_x = np.nonzero(is_pos & (chem == cx))[0].astype(np.int64)
        test_idx = np.sort(np.concatenate([pos_x, te_ctrl]))
        y = (pool.labels[test_idx] > 0).astype(int)

        model, _ = load_model(ckpt, device)
        # centre from TRAINING controls (unmodified)
        rep_c, proj_c, _ = embed_and_logit(model, pool.paths, pool.file_sizes, tr_ctrl0)
        c_rep = rep_c.mean(0); c_proj = proj_c.mean(0)
        c_proj = c_proj / (np.linalg.norm(c_proj) + 1e-8)
        # test embeddings
        rep_t, proj_t, logit_t = embed_and_logit(model, pool.paths, pool.file_sizes, test_idx)

        rep_cos = 1 - (rep_t @ c_rep) / (np.linalg.norm(rep_t, axis=1) * np.linalg.norm(c_rep) + 1e-8)
        rep_euc = np.linalg.norm(rep_t - c_rep, axis=1)
        proj_cos = 1 - proj_t @ c_proj
        proj_euc = np.linalg.norm(proj_t - c_proj, axis=1)
        print(f"{cx:5s} | {auroc(y, logit_t):6.3f} {auroc(y, rep_cos):7.3f} "
              f"{auroc(y, rep_euc):7.3f} {auroc(y, proj_cos):8.3f} {auroc(y, proj_euc):8.3f}")


if __name__ == '__main__':
    main()
