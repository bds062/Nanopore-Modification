#!/usr/bin/env python3
"""
orca_remake for the DANN two-head architecture (results7/results8) -- does
the 96-d penultimate rep separate modification chemistry, and did the
gradient-reversed adversary (type for results7, dataset/organism for
results8) actually reduce the organism-dominated clustering found in every
prior architecture (embedding-not-chemistry-space, results6-bce-supcon-
tradeoff)?

Unlike sad_embedding_cluster.py, ConvFormerV2DANN has only ONE embedding
space (no separate SupCon/Deep-SAD projections) -- the 96-d rep that both
presence_head and adv_head read from. Captured via a forward hook on
presence_head (its INPUT), same technique used everywhere else in this repo.
The checkpoint is loaded directly here (not via score_genome.load_model,
which does not know about this architecture).

Reuses build_selection/cluster_space/scatter/TYPE_COLOR/etc. from
sad_embedding_cluster.py.

Usage: python dann_embedding_cluster.py --ckpt <path> --out-dir <dir>
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
           REPO / 'scripts' / 'train',
           REPO / 'scripts' / 'train',
           REPO / 'rawmod'):
    sys.path.insert(0, str(_p))

import run_matched_loco as ML                                    # noqa: E402
from run_matched_loco import R                                   # noqa: E402
from run_convformer_v2 import ConvFormerV2DANN                    # noqa: E402
from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402
from sad_embedding_cluster import (build_selection, cluster_space, scatter,  # noqa: E402
                                   TYPE_COLOR, TYPE_ORDER, ORG_COLOR, ORG_ORDER,
                                   MAX_PER_GROUP)
from sklearn.decomposition import PCA                             # noqa: E402
from sklearn.manifold import TSNE                                 # noqa: E402

SEED = 0


def load_dann_model(ckpt, device):
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = raw['model_state']
    n_adv = raw.get('n_adv_classes', int(sd['adv_head.1.weight'].shape[0]))
    height = int(sd['pos'].shape[1])   # positional embedding shape (1, h, d_model)
    model = ConvFormerV2DANN(n_adv_classes=n_adv, h=height)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"  checkpoint: adv_target={raw.get('adv_target')} n_adv={n_adv} h={height} "
          f"epoch={raw.get('epoch')} val_auprc={raw.get('val_auprc')}", flush=True)
    return model


def extract_rep(pool, idx, device, ckpt):
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(512, 6, device, _worker_init_fn))
    model = load_dann_model(ckpt, device)
    cap = {}
    hook = model.presence_head.register_forward_hook(
        lambda _m, inp, _o: cap.__setitem__('e', inp[0].detach()))
    reps = []
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device, non_blocking=True))
            reps.append(cap['e'].float().cpu().numpy())
    hook.remove()
    return np.concatenate(reps, 0)


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

    print(f"\nEmbedding via {ckpt}", flush=True)
    rep = extract_rep(pool, idx, device, ckpt)
    print(f"  penultimate {rep.shape}", flush=True)
    np.savez_compressed(out / 'embeddings.npz', rep=rep.astype(np.float16),
                        types=types, orgs=orgs, idx=idx.astype(np.int64))

    report = [
        "Emergent modification-type structure in the DANN two-head architecture's\n",
        "embedding (no BCE/SupCon/Deep SAD -- presence head + gradient-reversed\n",
        "adversary only). Type labels used ONLY to score the unsupervised GMM\n",
        "clustering, never to fit it or the model.\n",
        f"\ncheckpoint: {ckpt}\n",
        f"pooled: {len(idx):,} images | types {dict(Counter(types))}\n",
        f"organisms {dict(Counter(orgs))}\n",
    ]
    rep_all, rep_sub = cluster_space(rep, types, orgs, "penultimate 96-d embedding", report)
    (out / 'clustering_report.txt').write_text(''.join(report))
    print('\n' + ''.join(report[7:]))

    with open(out / 'clustering_metrics.tsv', 'w') as fh:
        fh.write("space\tsubset\tari_type\tari_organism\tnmi_type\tnmi_organism\n")
        fh.write(f"penultimate\tall\t{rep_all['ari_type']:.4f}\t{rep_all['ari_dataset']:.4f}"
                 f"\t{rep_all['nmi_type']:.4f}\t{rep_all['nmi_dataset']:.4f}\n")
        if rep_sub:
            fh.write(f"penultimate\tunconfounded\t{rep_sub['ari_type']:.4f}"
                     f"\t{rep_sub['ari_dataset']:.4f}\t{rep_sub['nmi_type']:.4f}"
                     f"\t{rep_sub['nmi_dataset']:.4f}\n")

    Xs = (rep - rep.mean(0)) / (rep.std(0) + 1e-8)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    n_sub = min(8000, len(idx))
    sub = rng.choice(len(idx), n_sub, replace=False) if len(idx) > n_sub else np.arange(len(idx))
    tsne_in = PCA(n_components=32, random_state=SEED).fit_transform(Xs[sub]) \
        if Xs.shape[1] > 32 else Xs[sub]
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init='pca').fit_transform(tsne_in)

    ari_box = ("GMM in full 96-d space:\n"
              f"ARI vs type     = {rep_all['ari_type']:.3f}\n"
              f"ARI vs organism = {rep_all['ari_dataset']:.3f}"
              + (f"\nunconfounded ARI(type) = {rep_sub['ari_type']:.3f}" if rep_sub else ""))

    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))
    scatter(axes[0, 0], pca, types, TYPE_ORDER, TYPE_COLOR,
           'PCA of DANN embedding\ncolored by TRUE modification type', ari_box)
    scatter(axes[0, 1], pca, orgs, ORG_ORDER, ORG_COLOR,
           'PCA of DANN embedding\ncolored by SOURCE organism', ari_box)
    scatter(axes[1, 0], tsne, types[sub], TYPE_ORDER, TYPE_COLOR,
           't-SNE of DANN embedding\ncolored by TRUE modification type', ari_box)
    scatter(axes[1, 1], tsne, orgs[sub], ORG_ORDER, ORG_COLOR,
           't-SNE of DANN embedding\ncolored by SOURCE organism', ari_box)
    fig.suptitle(
        f"DANN two-head embedding ({ckpt.parent.parent.parent.name}) -- does removing BCE/\n"
        "SupCon/Deep SAD and adversarially erasing type-or-dataset info change what\n"
        "the embedding clusters by?", fontsize=12, y=0.995)
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.03, right=0.97,
                        hspace=0.25, wspace=0.15)
    fig.savefig(out / 'figures' / 'fig_embedding_cluster.png', dpi=150)
    plt.close(fig)
    print(f"\nwrote {out/'figures'/'fig_embedding_cluster.png'}")
    print(f"wrote {out/'clustering_report.txt'}")
    print(f"wrote {out/'clustering_metrics.tsv'}")


if __name__ == '__main__':
    main()
