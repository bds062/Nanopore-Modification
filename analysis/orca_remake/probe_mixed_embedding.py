#!/usr/bin/env python3
"""
Post-hoc classification probe on the results6 'mixed' checkpoint's embedding.

results6 trains the SAME architecture/curriculum as results5 but with BCE_WEIGHT=0:
the classification head is never trained, so the ONLY signal shaping the
representation is SupCon (contrastive) + Deep SAD (semi-supervised anomaly),
both still label-driven, just not via a cross-entropy decision boundary. This
script asks the resulting question directly: given a frozen embedding trained
PURELY by metric-learning objectives, how well does mod/unmod separate if you fit
a simple downstream classifier on top? (Standard SimCLR/SupCon-style evaluation:
train representation, then linear-probe it.)

Three embedding spaces from the SAME forward pass are probed:
  - 96-d penultimate rep      (self.norm(pooled), input to the untrained head)
  - 32-d Deep-SAD space       (model._sad, cached every forward regardless of mode)
  - 128-d SupCon proj space   (model.proj(rep), L2-normalised; NOT cached at eval by
                               ConvFormerV2.forward, so computed here explicitly)

Train/test split is the mixed fold's OWN deterministic 85/15 split
(run_matched_loco.mixed_split), recomputed here (not re-derived by hand) so it is
guaranteed identical to what the checkpoint was actually trained/tested on.

Probes: LogisticRegression (linear) and k-NN (k=25, nonlinear) per space, fit on a
capped training subsample, scored by AUROC on the FULL held-out test split.
Also reports the raw (no-fit) Deep-SAD anomaly-distance AUROC for reference --
that number is also in metrics/mixed.tsv already (auroc_sad), computed independent
of any probe.

Usage: python probe_mixed_embedding.py [--ckpt-dir results6]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
for _p in (REPO / 'experiments' / 'pipeline4',
           REPO / 'experiments' / 'pipeline3',
           REPO / 'experiments' / 'pipeline1',
           REPO / 'deepmod'):
    sys.path.insert(0, str(_p))

import run_matched_loco as ML                                    # noqa: E402
from run_matched_loco import R                                   # noqa: E402
from score_genome import load_model                              # noqa: E402
from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.neighbors import KNeighborsClassifier                # noqa: E402
from sklearn.preprocessing import StandardScaler                  # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402

RESULTS_BASE = Path('/fs/cbcb-scratch/bds062/results/rawmod_matched_loco')
SEED = 0
MAX_TRAIN_PROBE = 60_000   # cap for probe-fitting speed; test split used in full


def extract_three_spaces(pool, idx, device, ckpt):
    """One forward pass: 96-d penultimate rep, 32-d Deep-SAD, 128-d SupCon proj."""
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(512, 6, device, _worker_init_fn))
    model, arch = load_model(ckpt, device)
    assert arch == 'convformer_v2', f"expected convformer_v2, got {arch}"
    has_proj = getattr(model, 'proj', None) is not None
    cap = {}
    hook = model.head.register_forward_hook(lambda _m, inp, _o: cap.__setitem__('e', inp[0].detach()))
    reps, sads, projs, labels = [], [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            model(xb.to(device, non_blocking=True))
            rep = cap['e']
            reps.append(rep.float().cpu().numpy())
            sads.append(model._sad.float().cpu().numpy())
            if has_proj:
                projs.append(model.proj(rep).float().cpu().numpy())
            labels.append(yb.numpy())
    hook.remove()
    out = {'rep': np.concatenate(reps, 0), 'sad': np.concatenate(sads, 0),
          'y': np.concatenate(labels, 0)}
    if has_proj:
        out['proj'] = np.concatenate(projs, 0)
    return out


def probe(Xtr, ytr, Xte, yte, seed=SEED):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    lr = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr_s, ytr)
    out['logreg'] = roc_auc_score(yte, lr.predict_proba(Xte_s)[:, 1])
    knn = KNeighborsClassifier(n_neighbors=25, n_jobs=6).fit(Xtr_s, ytr)
    out['knn'] = roc_auc_score(yte, knn.predict_proba(Xte_s)[:, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='results6')
    a = ap.parse_args()
    ckpt_root = RESULTS_BASE / a.ckpt_dir
    ckpt = ckpt_root / 'models' / 'mixed' / 'mixed' / 'best_model.pt'
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(SEED)

    members = ML.build_members()
    pool = R.Group(list(members), members)
    hp = R.HP()
    mod_map = ML.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = ML.ref_base_center(pool)
    is_pos = pool.labels > 0
    neg_mask = np.zeros(pool.N, dtype=bool)
    neg_mask[ML.subsample_negatives(pool, seed=hp.seed)] = True

    # EXACT same split the checkpoint was trained/tested on (deterministic, refactored
    # out of run_matched_loco.py's own 'mixed' branch -- not re-derived by hand).
    train_idx, test_idx, stats = ML.mixed_split(pool, is_pos, neg_mask, hp)
    print(f"mixed split: {stats}", flush=True)
    print(f"train={len(train_idx):,}  test={len(test_idx):,}", flush=True)

    train_sub = (rng.choice(train_idx, MAX_TRAIN_PROBE, replace=False)
                if len(train_idx) > MAX_TRAIN_PROBE else train_idx)
    print(f"probe-fitting on {len(train_sub):,} train images "
          f"(capped from {len(train_idx):,}); test on all {len(test_idx):,}", flush=True)

    print(f"\nEmbedding TRAIN subsample via {ckpt}", flush=True)
    tr = extract_three_spaces(pool, train_sub, device, ckpt)
    print(f"Embedding TEST split via {ckpt}", flush=True)
    te = extract_three_spaces(pool, test_idx, device, ckpt)

    spaces = ['rep', 'sad'] + (['proj'] if 'proj' in tr else [])
    dims = {'rep': 96, 'sad': 32, 'proj': 128}
    results = {}
    for sp in spaces:
        print(f"\n=== probing {sp} space ({dims[sp]}-d) ===", flush=True)
        r = probe(tr[sp], tr['y'], te[sp], te['y'])
        results[sp] = r
        print(f"  logreg AUROC={r['logreg']:.4f}   knn(k=25) AUROC={r['knn']:.4f}", flush=True)

    # raw (no-fit) Deep-SAD anomaly-distance AUROC, for reference (same metric
    # already written to metrics/mixed.tsv's auroc_sad column independently)
    sad_center = None
    from run_convformer_v2 import ConvFormerV2
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = raw['model_state']
    sad_center = sd.get('sad_center')
    if sad_center is not None:
        d = np.linalg.norm(te['sad'] - sad_center.cpu().numpy(), axis=1)
        raw_sad_auroc = roc_auc_score(te['y'], d)
        print(f"\nraw (no probe) Deep-SAD distance-from-centre AUROC = {raw_sad_auroc:.4f}",
              flush=True)
    else:
        raw_sad_auroc = float('nan')

    out_dir = ckpt_root / 'probe'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'probe_results.tsv', 'w') as fh:
        fh.write("space\tdim\tlogreg_auroc\tknn_auroc\n")
        for sp in spaces:
            fh.write(f"{sp}\t{dims[sp]}\t{results[sp]['logreg']:.4f}\t{results[sp]['knn']:.4f}\n")
        fh.write(f"sad_raw_distance\t{dims['sad']}\t{raw_sad_auroc:.4f}\tNA\n")
    print(f"\nwrote {out_dir/'probe_results.tsv'}")


if __name__ == '__main__':
    main()
