#!/usr/bin/env python3
"""results15 embedding, sampling ALL organisms now in the pool (original
ONT/SPO1/HP + the 7 EXTRA_ORGANISMS benchmark datasets), not just the
original 3. BENCH:: organisms don't have a 5-way chemistry type the way
ONT/SPO1/HP do (they're single-sample WT/native data, motif- or
bisulfite-derived), so "type" here means the specific dataset name for those
(e.g. 'Ecoli_DM_5kHz'), same as organism -- the interesting panel for them is
mod-vs-unmod and organism, not type.

Usage: python results15_embedding_allorg.py --ckpt <path> --out-dir <dir>
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
for _p in (REPO / 'analysis' / 'orca_remake', REPO / 'scripts' / 'train',
           REPO / 'scripts' / 'train', REPO / 'rawmod', REPO / 'scripts' / 'test'):
    sys.path.insert(0, str(_p))

import os
os.environ.setdefault('RAWMOD_DATA_GEN', 'strand15')
os.environ.setdefault('EXTRA_ORGANISMS', '1')
os.environ.setdefault('INCLUDE_HUMAN', '1')

import run_matched_loco as ML                                   # noqa: E402
from run_matched_loco import R                                  # noqa: E402
from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402
from score_genome import load_model                              # noqa: E402
from sklearn.decomposition import PCA                             # noqa: E402
from sklearn.manifold import TSNE                                 # noqa: E402
from sklearn.mixture import GaussianMixture                       # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

SEED = 0
MAX_PER_GROUP = 200
MOD_COLOR = {'mod': '#D55E00', 'unmod': '#999999'}
MOD_ORDER = ['mod', 'unmod']

CHEM_ORDER = ['5mC', '5hmC', '6mA', '4mC', '5hmU']
ORIG_ORGS = ['ONT', 'SPO1', 'HP']
BENCH_ORGS = ['Anabaena_WT_5kHz', 'Ecoli_DM_5kHz', 'Ecoli_DM_MSssI_5kHz', 'Ecoli_WT_5kHz',
             'Tdenticola_WT_5kHz', 'HPJ99_WT_5kHz', 'arabidopsis', 'hg001', 'hg002']
ORG_ORDER = ORIG_ORGS + BENCH_ORGS
import matplotlib.cm as cm
_palette = cm.get_cmap('tab10').colors + cm.get_cmap('Set3').colors
ORG_COLOR = {o: _palette[i % len(_palette)] for i, o in enumerate(ORG_ORDER)}


def build_selection(pool, chem, is_pos, rng):
    """Balanced sample per (organism, mod/unmod) -- organism here is the
    specific dataset for BENCH:: members, or ONT/SPO1/HP for the original
    pool. type = chem for original organisms, else the organism name."""
    member_of = np.array([pool.names[int(pool.file_of[i])] for i in range(pool.N)])
    org_of = np.array([m.split('::')[0] if not m.startswith('BENCH::') else m.split('::')[1]
                       for m in member_of])

    take_idx, orgs, types, mods = [], [], [], []

    def add(pool_idx, org, is_mod_flag):
        if len(pool_idx) == 0:
            return
        sel = pool_idx if len(pool_idx) <= MAX_PER_GROUP else \
            rng.choice(pool_idx, MAX_PER_GROUP, replace=False)
        sel = np.sort(sel)
        take_idx.append(sel)
        orgs.extend([org] * len(sel))
        mods.extend(['mod' if is_mod_flag else 'unmod'] * len(sel))
        if org in ORIG_ORGS:
            for i in sel:
                t = chem[i] if is_mod_flag else 'unmod'
                types.append(t if t not in ('', 'untyped') else ('mod_untyped' if is_mod_flag else 'unmod'))
        else:
            types.extend([org] * len(sel))
        print(f"  {org:22} {'mod' if is_mod_flag else 'unmod':6} n={len(sel):,} (of {len(pool_idx):,})")

    for org in ORG_ORDER:
        m = org_of == org
        pos = np.nonzero(m & is_pos)[0].astype(np.int64)
        neg = np.nonzero(m & ~is_pos)[0].astype(np.int64)
        add(pos, org, True)
        add(neg, org, False)

    idx = np.concatenate(take_idx)
    return idx, np.array(orgs), np.array(types), np.array(mods)


def extract_rep(pool, idx, device, ckpt):
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(512, 6, device, _worker_init_fn))
    model, arch = load_model(ckpt, device)
    cap = {}
    target = model.head[3] if arch == 'inception' else model.head
    hook = target.register_forward_hook(lambda _m, inp, _o: cap.__setitem__('e', inp[0].detach()))
    reps = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device, non_blocking=True))
            reps.append(cap['e'].float().cpu().numpy())
    hook.remove()
    return np.concatenate(reps, 0)


def gmm_ari(X, labels, n_components):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    # reg_covar floor avoids a singular/non-PD covariance when a group is very
    # small relative to the 96-dim space (e.g. hg002, n~100) -- default 1e-6 is
    # too small once more, smaller organism groups are in the mix.
    gmm = GaussianMixture(n_components=n_components, random_state=SEED, n_init=3,
                          reg_covar=1e-3).fit(Xs)
    pred = gmm.predict(Xs)
    return (adjusted_rand_score(labels, pred), normalized_mutual_info_score(labels, pred))


def scatter(ax, xy, labels, order, cmap, title, info_box):
    for lab in order:
        m = labels == lab
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, color=cmap[lab],
                  label=f'{lab} (n={m.sum():,})')
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6.5, markerscale=2, loc='upper right', ncol=1)
    ax.text(0.02, 0.02, info_box, transform=ax.transAxes, fontsize=8,
           va='bottom', ha='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#797979'))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); (out / 'figures').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    members = ML.build_members()
    pool = R.Group(list(members), members)
    print(f"Full pool (EXTRA_ORGANISMS=1): {pool.N:,} images across {len(members)} files", flush=True)
    mod_map = ML.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = ML.ref_base_center(pool)
    chem = ML.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0

    idx, orgs, types, mods = build_selection(pool, chem, is_pos, rng)
    print(f"\nSelected {len(idx):,} images across {len(set(orgs))} organisms", flush=True)

    print(f"\nEmbedding via {a.ckpt}", flush=True)
    rep = extract_rep(pool, idx, device, a.ckpt)
    print(f"  penultimate {rep.shape}", flush=True)
    np.savez_compressed(out / 'embeddings_allorg.npz', rep=rep.astype(np.float16),
                        orgs=orgs, types=types, mods=mods, idx=idx.astype(np.int64))

    ari_mod, nmi_mod = gmm_ari(rep, (mods == 'mod').astype(int), 2)
    ari_org, nmi_org = gmm_ari(rep, orgs, len(set(orgs)))
    report = (f"2-cluster GMM ARI vs mod/unmod = {ari_mod:.4f}  (NMI={nmi_mod:.4f})\n"
             f"{len(set(orgs))}-way GMM ARI vs organism = {ari_org:.4f}  (NMI={nmi_org:.4f})\n")
    print('\n' + report)
    (out / 'clustering_report_allorg.txt').write_text(
        f"checkpoint: {a.ckpt}\npooled: {len(idx):,} images\norganisms: {sorted(set(orgs))}\n\n" + report)

    Xs = (rep - rep.mean(0)) / (rep.std(0) + 1e-8)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    n_sub = min(8000, len(idx))
    sub = rng.choice(len(idx), n_sub, replace=False) if len(idx) > n_sub else np.arange(len(idx))
    tsne_in = PCA(n_components=32, random_state=SEED).fit_transform(Xs[sub]) \
        if Xs.shape[1] > 32 else Xs[sub]
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init='pca').fit_transform(tsne_in)

    info_box = f"ARI(mod/unmod)={ari_mod:.3f}\nARI(organism,{len(set(orgs))}-way)={ari_org:.3f}"

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), dpi=140)
    scatter(axes[0], pca, orgs, ORG_ORDER, ORG_COLOR,
           'PCA colored by ORGANISM (all 10)', info_box)
    scatter(axes[1], pca, mods, MOD_ORDER, MOD_COLOR,
           'PCA colored by MOD vs UNMOD', info_box)
    fig.suptitle('results15 (EXTRA_ORGANISMS=1) embedding: all 10 organisms', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / 'figures' / 'fig_embedding_allorg_pca.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2), dpi=140)
    scatter(axes[0], tsne, orgs[sub], ORG_ORDER, ORG_COLOR,
           't-SNE colored by ORGANISM (all 10)', info_box)
    scatter(axes[1], tsne, mods[sub], MOD_ORDER, MOD_COLOR,
           't-SNE colored by MOD vs UNMOD', info_box)
    fig.suptitle('results15 (EXTRA_ORGANISMS=1) embedding: all 10 organisms', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / 'figures' / 'fig_embedding_allorg_tsne.png', dpi=150)
    plt.close(fig)

    print(f"\nwrote figures + {out/'embeddings_allorg.npz'}")


if __name__ == '__main__':
    main()
