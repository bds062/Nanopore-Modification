#!/usr/bin/env python3
"""
orca_remake for the results13 architecture (ConvFormerV2 + BCE + SupCon-on-
presence + gradient-reversed organism adversary). Extracts the 96-d
penultimate rep (captured via a forward hook on `model.head`, its INPUT --
same technique used everywhere else in this repo) and computes:
  - 2-cluster GMM ARI vs the BINARY mod/unmod label (the actual target metric
    for this experiment -- "2 balls" of modified vs unmodified)
  - ARI vs 6-way type (5 chemistries + unmod) and vs organism, for comparison
    with every other architecture in this model family.

Usage: python supcon_orgadv_embedding_cluster.py --ckpt <path> --out-dir <dir>
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
for _p in (REPO / 'analysis' / 'orca_remake',
           REPO / 'experiments' / 'pipeline4',
           REPO / 'experiments' / 'pipeline1',
           REPO / 'deepmod'):
    sys.path.insert(0, str(_p))

import run_matched_loco as ML                                    # noqa: E402
from run_matched_loco import R                                   # noqa: E402
from run_convformer_v2 import ConvFormerV2                        # noqa: E402
from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402
from sad_embedding_cluster import (build_selection, scatter,     # noqa: E402
                                   TYPE_COLOR, TYPE_ORDER, ORG_COLOR, ORG_ORDER,
                                   MAX_PER_GROUP)
from sklearn.decomposition import PCA                             # noqa: E402
from sklearn.manifold import TSNE                                 # noqa: E402
from sklearn.mixture import GaussianMixture                       # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from sklearn.preprocessing import StandardScaler                  # noqa: E402

SEED = 0
MOD_COLOR = {'mod': '#D55E00', 'unmod': '#999999'}
MOD_ORDER = ['mod', 'unmod']


def load_model(ckpt, device):
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = raw['model_state']
    height = raw.get('height', int(sd['pos'].shape[1]))
    supcon_dim = raw.get('supcon_dim', 0)
    org_adv_classes = raw.get('org_adv_classes', 0)
    model = ConvFormerV2(h=height, supcon_dim=supcon_dim, org_adv_classes=org_adv_classes)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"  checkpoint: height={height} supcon_dim={supcon_dim} "
          f"org_adv_classes={org_adv_classes} epoch={raw.get('epoch')} "
          f"val_auprc={raw.get('val_auprc')}", flush=True)
    return model


def extract_rep(pool, idx, device, ckpt):
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(512, 6, device, _worker_init_fn))
    model = load_model(ckpt, device)
    cap = {}
    hook = model.head.register_forward_hook(
        lambda _m, inp, _o: cap.__setitem__('e', inp[0].detach()))
    reps = []
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device, non_blocking=True))
            reps.append(cap['e'].float().cpu().numpy())
    hook.remove()
    return np.concatenate(reps, 0)


def gmm_ari(X, labels, n_components):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    gmm = GaussianMixture(n_components=n_components, random_state=SEED, n_init=3).fit(Xs)
    pred = gmm.predict(Xs)
    return (adjusted_rand_score(labels, pred),
           normalized_mutual_info_score(labels, pred))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    ckpt, out = Path(a.ckpt), Path(a.out_dir)

    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    members = ML.build_members()
    pool = R.Group(list(members), members)
    print(f"Matched pool: {pool.N:,} images across {len(members)} files", flush=True)
    mod_map = ML.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = ML.ref_base_center(pool)
    chem = ML.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    print("  positives per chemistry:",
          {k: int(v) for k, v in Counter(chem[is_pos]).items()}, flush=True)

    idx, types, orgs = build_selection(pool, chem, is_pos, refbase, rng)
    print(f"\nSelected {len(idx):,} images: {dict(Counter(types))}", flush=True)
    is_mod = np.array(['unmod' if t == 'unmod' else 'mod' for t in types])

    print(f"\nEmbedding via {ckpt}", flush=True)
    rep = extract_rep(pool, idx, device, ckpt)
    print(f"  penultimate {rep.shape}", flush=True)
    np.savez_compressed(out / 'embeddings.npz', rep=rep.astype(np.float16),
                        types=types, orgs=orgs, is_mod=is_mod, idx=idx.astype(np.int64))

    # --- the actual target metric: 2-cluster GMM vs mod/unmod ---
    ari_mod, nmi_mod = gmm_ari(rep, (is_mod == 'mod').astype(int), 2)
    ari_type_all, nmi_type_all = gmm_ari(rep, types, 6)
    ari_org_all, nmi_org_all = gmm_ari(rep, orgs, 3)
    unconfounded = np.isin(types, ['5mC', '5hmC', '6mA'])
    ari_type_unc, nmi_type_unc = gmm_ari(rep[unconfounded], types[unconfounded], 3)
    ari_org_unc, nmi_org_unc = gmm_ari(rep[unconfounded], orgs[unconfounded], 3)

    with open(out / 'clustering_metrics.tsv', 'w') as fh:
        fh.write("space\tsubset\tari_modvunmod\tari_type\tari_organism\t"
                 "nmi_modvunmod\tnmi_type\tnmi_organism\n")
        fh.write(f"penultimate\tall\t{ari_mod:.4f}\t{ari_type_all:.4f}\t{ari_org_all:.4f}\t"
                 f"{nmi_mod:.4f}\t{nmi_type_all:.4f}\t{nmi_org_all:.4f}\n")
        fh.write(f"penultimate\tunconfounded\t\t{ari_type_unc:.4f}\t{ari_org_unc:.4f}\t"
                 f"\t{nmi_type_unc:.4f}\t{nmi_org_unc:.4f}\n")

    report = (f"2-cluster GMM ARI vs mod/unmod = {ari_mod:.4f}  (NMI={nmi_mod:.4f})\n"
             f"6-way GMM ARI vs type (incl unmod) = {ari_type_all:.4f}\n"
             f"3-way GMM ARI vs organism = {ari_org_all:.4f}\n"
             f"unconfounded (5mC,5hmC,6mA) ARI vs type = {ari_type_unc:.4f}\n"
             f"unconfounded (5mC,5hmC,6mA) ARI vs organism = {ari_org_unc:.4f}\n")
    print('\n' + report)
    (out / 'clustering_report.txt').write_text(
        f"checkpoint: {ckpt}\npooled: {len(idx):,} images\n\n" + report)

    Xs = (rep - rep.mean(0)) / (rep.std(0) + 1e-8)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    n_sub = min(8000, len(idx))
    sub = rng.choice(len(idx), n_sub, replace=False) if len(idx) > n_sub else np.arange(len(idx))
    tsne_in = PCA(n_components=32, random_state=SEED).fit_transform(Xs[sub]) \
        if Xs.shape[1] > 32 else Xs[sub]
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init='pca').fit_transform(tsne_in)

    info_box = (f"2-cluster GMM ARI(mod/unmod) = {ari_mod:.3f}\n"
               f"ARI(type,6-way) = {ari_type_all:.3f}\n"
               f"ARI(organism) = {ari_org_all:.3f}\n"
               f"unconfounded ARI(organism) = {ari_org_unc:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.5))
    scatter(axes[0, 0], pca, types, TYPE_ORDER, TYPE_COLOR,
           'PCA colored by TRUE modification type', info_box)
    scatter(axes[0, 1], pca, orgs, ORG_ORDER, ORG_COLOR,
           'PCA colored by SOURCE organism', info_box)
    scatter(axes[0, 2], pca, is_mod, MOD_ORDER, MOD_COLOR,
           'PCA colored by MOD vs UNMOD (target axis)', info_box)
    scatter(axes[1, 0], tsne, types[sub], TYPE_ORDER, TYPE_COLOR,
           't-SNE colored by TRUE modification type', info_box)
    scatter(axes[1, 1], tsne, orgs[sub], ORG_ORDER, ORG_COLOR,
           't-SNE colored by SOURCE organism', info_box)
    scatter(axes[1, 2], tsne, is_mod[sub], MOD_ORDER, MOD_COLOR,
           't-SNE colored by MOD vs UNMOD (target axis)', info_box)
    fig.suptitle(f"results13 embedding (BCE+SupCon-on-presence+organism-adversary) -- "
                f"does the embedding form 2 balls (mod vs unmod)?", fontsize=13, y=0.995)
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.98,
                        hspace=0.25, wspace=0.15)
    fig.savefig(out / 'figures' / 'fig_embedding_cluster.png', dpi=150)
    plt.close(fig)
    print(f"\nwrote {out/'figures'/'fig_embedding_cluster.png'}")
    print(f"wrote {out/'clustering_report.txt'}")
    print(f"wrote {out/'clustering_metrics.tsv'}")


if __name__ == '__main__':
    main()
