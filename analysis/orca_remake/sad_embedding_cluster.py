#!/usr/bin/env python3
"""
orca_remake (best-model redo) — does RawMod's BEST checkpoint's own latent
embedding separate modification CHEMISTRY without ever being given a type label?

This is the ORCA "fingerprinting / label-transfer" analog, redone with the
current best architecture: the matched-only, curriculum + SupCon + **Deep SAD**
mixed model (model #4 in results4/MODEL_COMPARISON.md). That checkpoint
(rawmod_matched_loco/results5/models/mixed/mixed/best_model.pt) was trained on
the whole matched pool as a BINARY mod/unmod task (+ the SAD anomaly objective),
and was NEVER given a modification-TYPE label. So any separation of 5mC / 5hmC /
6mA / 4mC / 5hmU in its latent space is emergent, not supervised.

Unlike the earlier orca_remake (ONT+SPO1 only, 4 chemistries), this version pools
all three matched organisms so all FIVE chemistries are present, including 4mC
(H. pylori WT vs WGA).

Two latent spaces are examined from the ONE checkpoint (same forward pass):
  - penultimate 96-d representation  (self.norm(pooled), the shared embedding)
  - 32-d Deep-SAD head output        (the space the anomaly score lives in)

THE CONFOUND, HANDLED EXPLICITLY (same as pipeline3/cluster_types.py):
  4mC occurs only in H. pylori and 5hmU only in SPO1, so for those two "type" and
  "organism" are perfectly confounded. We therefore report BOTH ARI-vs-type and
  ARI-vs-organism, and additionally an UNCONFOUNDED subset restricted to the
  chemistries present in >1 organism (5mC, 5hmC, 6mA). Clusters tracking type on
  that subset is the honest evidence the space encodes chemistry, not provenance.

CAVEAT stated plainly (as in the earlier orca_remake): the checkpoint WAS trained
on exactly these reads (as binary mod/unmod). This is an in-sample embedding-space
proof of concept for TYPE separation, not a zero-shot-typing claim.

Outputs (under results4/embedding_clustering/):
  embeddings.npz                 pooled 96-d + 32-d embeddings, type, organism, source
  clustering_report.txt          GMM ARI/NMI vs type & organism, both spaces + subset
  clustering_metrics.tsv         machine-readable version of the above
  figures/fig_embedding_cluster.png   2x2 PCA/t-SNE, colored by type and by organism
"""
import io
import os
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
for _p in (REPO / 'scripts' / 'train',
           REPO / 'scripts' / 'test',
           REPO / 'scripts' / 'train',
           REPO / 'rawmod'):
    sys.path.insert(0, str(_p))

import run_matched_loco as ML                                   # noqa: E402
from run_matched_loco import R                                  # run_pipeline  # noqa: E402
from score_genome import load_model                             # noqa: E402
from cluster_types import gmm_cluster, bic_sweep, evaluate      # noqa: E402
from model import PileupDataset, make_loader_kwargs, _worker_init_fn  # noqa: E402
from sklearn.decomposition import PCA                           # noqa: E402
from sklearn.manifold import TSNE                               # noqa: E402
from sklearn.metrics import adjusted_rand_score                 # noqa: E402

CKPT = Path('/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results5/'
            'models/mixed/mixed/best_model.pt')
OUT = Path('/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results4/embedding_clustering')

MAX_PER_GROUP = 3000        # cap per (type, organism) group
SEED = 0
CHEMS = ('5mC', '5hmC', '6mA', '4mC', '5hmU')

# amplification / synthetic control members = clean unmodified, per organism
UNMOD_MEMBERS = {
    'ONT::control': 'ONT',
    'HP::WGA': 'HP',
    'SPO1::bc01': 'SPO1', 'SPO1::bc02': 'SPO1', 'SPO1::bc03': 'SPO1',
    'SPO1::bc04': 'SPO1', 'SPO1::bc05': 'SPO1',
}

TYPE_COLOR = {'5mC': '#4878CF', '5hmC': '#6ACC65', '6mA': '#D65F5F',
              '4mC': '#EE854A', '5hmU': '#956CB4', 'unmod': '#B0B0B0'}
TYPE_ORDER = ['5mC', '5hmC', '6mA', '4mC', '5hmU', 'unmod']
ORG_COLOR = {'ONT': '#4878CF', 'SPO1': '#D65F5F', 'HP': '#6ACC65'}
ORG_ORDER = ['ONT', 'SPO1', 'HP']


def build_selection(pool, chem, is_pos, refbase, rng):
    """Pick balanced global image indices; return idx, type[], organism[]."""
    member_of = np.array([pool.names[int(pool.file_of[i])] for i in range(pool.N)])
    org_of = np.array([m.split('::')[0] for m in member_of])

    take_idx, types, orgs = [], [], []

    def add(pool_idx, mtype, organism):
        if len(pool_idx) == 0:
            print(f"  [skip] {mtype:6} {organism:5}: no sites"); return
        sel = pool_idx if len(pool_idx) <= MAX_PER_GROUP else \
            rng.choice(pool_idx, MAX_PER_GROUP, replace=False)
        sel = np.sort(sel)
        take_idx.append(sel)
        types.extend([mtype] * len(sel)); orgs.extend([organism] * len(sel))
        print(f"  {mtype:6} {organism:5} n={len(sel):,} (of {len(pool_idx):,})")

    # modified positives, split by carrying organism
    print("Modified positives (typed, split by organism):")
    for c in CHEMS:
        for org in ORG_ORDER:
            m = np.nonzero(is_pos & (chem == c) & (org_of == org))[0].astype(np.int64)
            add(m, c, org)

    # clean unmodified from amplification / synthetic controls
    print("Unmodified (amplicon / synthetic control, split by organism):")
    unmod_member_arr = np.isin(member_of, list(UNMOD_MEMBERS))
    for org in ORG_ORDER:
        m = np.nonzero((~is_pos) & unmod_member_arr & (org_of == org))[0].astype(np.int64)
        add(m, 'unmod', org)

    idx = np.concatenate(take_idx)
    return idx, np.array(types), np.array(orgs)


def extract_embeddings(pool, idx, device, ckpt=CKPT):
    """One forward pass over idx: penultimate 96-d rep + 32-d Deep-SAD output."""
    ds = PileupDataset(pool.paths, np.asarray(idx, np.int64), pool.file_sizes,
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False)
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(512, 6, device, _worker_init_fn))
    model, arch = load_model(ckpt, device)
    cap = {}
    target = model.head[3] if arch == 'inception' else model.head
    hook = target.register_forward_hook(lambda _m, inp, _o: cap.__setitem__('e', inp[0].detach()))
    reps, sads = [], []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device, non_blocking=True))
            reps.append(cap['e'].float().cpu().numpy())
            sads.append(model._sad.float().cpu().numpy())
    hook.remove()
    return np.concatenate(reps, 0), np.concatenate(sads, 0)


def cluster_space(X, types, orgs, name, report):
    """GMM on X (unsupervised); ARI/NMI vs type and organism, full + unconfounded."""
    report.append(f"\n================  {name}  ================\n")
    n_types = len(set(types))
    sweep = bic_sweep(X, kmax=min(9, n_types + 3), seed=SEED)
    best_k = min(sweep, key=lambda t: t[1])[0]
    report.append(f"BIC-selected components: {best_k}   (true #types = {n_types})\n")

    lab, _ = gmm_cluster(X, n_types, SEED)
    fh = io.StringIO()
    res_all = evaluate(lab, types, orgs, f"ALL TYPES ({name})", fh)
    report.append(fh.getvalue())

    # unconfounded: chemistries present in >1 organism
    multi = [t for t in set(types) if t != 'unmod'
             and len(set(orgs[types == t])) > 1]
    report.append(f"UNCONFOUNDED subset (chemistry in >1 organism): {sorted(multi)}\n")
    res_sub = None
    if len(multi) >= 2:
        m = np.isin(types, multi)
        lab_m, _ = gmm_cluster(X[m], len(multi), SEED)
        fh2 = io.StringIO()
        res_sub = evaluate(lab_m, types[m], orgs[m],
                           f"UNCONFOUNDED {sorted(multi)} ({name})", fh2)
        report.append(fh2.getvalue())
    return res_all, res_sub


def scatter(ax, xy, labels, order, cmap, title, ari_box):
    for lab in order:
        m = labels == lab
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, color=cmap[lab],
                   label=f'{lab} (n={m.sum():,})')
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, markerscale=2, loc='upper right')
    ax.text(0.02, 0.02, ari_box, transform=ax.transAxes, fontsize=8.5,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#797979'))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(CKPT),
                    help='checkpoint to embed (default: results5 mixed SAD)')
    ap.add_argument('--out-dir', default=str(OUT),
                    help='output dir (default: results4/embedding_clustering)')
    a = ap.parse_args()
    ckpt, out = Path(a.ckpt), Path(a.out_dir)

    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- assemble the matched pool + per-site chemistry typing (reuse the driver) ----
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
    print(f"\nSelected {len(idx):,} images: "
          f"{dict(Counter(types))}\n  organisms {dict(Counter(orgs))}", flush=True)

    # ---- embed everything through the ONE best (mixed SAD) checkpoint ----
    print(f"\nEmbedding via {ckpt}", flush=True)
    rep, sad = extract_embeddings(pool, idx, device, ckpt=ckpt)
    print(f"  penultimate {rep.shape}   SAD-space {sad.shape}", flush=True)
    np.savez_compressed(out / 'embeddings.npz', rep=rep.astype(np.float16),
                        sad=sad.astype(np.float16), types=types, orgs=orgs,
                        idx=idx.astype(np.int64))

    # ---- unsupervised clustering in both spaces ----
    report = [
        "Emergent modification-type structure in RawMod's best-model embedding\n",
        "(ORCA fingerprinting analog; the checkpoint is trained BINARY mod/unmod\n",
        " + Deep SAD, NEVER given a type label. Type labels are used ONLY to score\n",
        " the unsupervised GMM clustering, never to fit it or the model.)\n",
        f"\ncheckpoint: {ckpt}\n",
        f"pooled: {len(idx):,} images | types {dict(Counter(types))}\n",
        f"organisms {dict(Counter(orgs))}\n",
        "\nCAVEAT: in-sample embedding space (model was trained on these reads as\n",
        "binary mod/unmod). Proof of concept for emergent TYPE separation, not a\n",
        "zero-shot-typing claim.\n",
    ]
    rep_all, rep_sub = cluster_space(rep, types, orgs, "penultimate 96-d embedding", report)
    sad_all, sad_sub = cluster_space(sad, types, orgs, "Deep-SAD 32-d space", report)
    (out / 'clustering_report.txt').write_text(''.join(report))
    print('\n' + ''.join(report[10:]))

    with open(out / 'clustering_metrics.tsv', 'w') as fh:
        fh.write("space\tsubset\tari_type\tari_organism\tnmi_type\tnmi_organism\n")
        for space, res_all, res_sub in [("penultimate", rep_all, rep_sub),
                                        ("sad", sad_all, sad_sub)]:
            fh.write(f"{space}\tall\t{res_all['ari_type']:.4f}\t{res_all['ari_dataset']:.4f}"
                     f"\t{res_all['nmi_type']:.4f}\t{res_all['nmi_dataset']:.4f}\n")
            if res_sub:
                fh.write(f"{space}\tunconfounded\t{res_sub['ari_type']:.4f}"
                         f"\t{res_sub['ari_dataset']:.4f}\t{res_sub['nmi_type']:.4f}"
                         f"\t{res_sub['nmi_dataset']:.4f}\n")

    # ---- figure: PCA + t-SNE of the penultimate embedding, colored 2 ways ----
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
            'PCA of best-model embedding\ncolored by TRUE modification type', ari_box)
    scatter(axes[0, 1], pca, orgs, ORG_ORDER, ORG_COLOR,
            'PCA of best-model embedding\ncolored by SOURCE organism', ari_box)
    scatter(axes[1, 0], tsne, types[sub], TYPE_ORDER, TYPE_COLOR,
            't-SNE of best-model embedding\ncolored by TRUE modification type', ari_box)
    scatter(axes[1, 1], tsne, orgs[sub], ORG_ORDER, ORG_COLOR,
            't-SNE of best-model embedding\ncolored by SOURCE organism', ari_box)
    fig.suptitle(
        "Does RawMod's best (mixed SAD) embedding separate modification chemistry\n"
        "without ever being given a type label?  (ORCA fingerprinting analog,\n"
        "all 5 chemistries, in-sample proof of concept)", fontsize=12, y=0.995)
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.03, right=0.97,
                        hspace=0.25, wspace=0.15)
    fig.savefig(out / 'figures' / 'fig_embedding_cluster.png', dpi=150)
    plt.close(fig)
    print(f"\nwrote {out/'figures'/'fig_embedding_cluster.png'}")
    print(f"wrote {out/'clustering_report.txt'}")
    print(f"wrote {out/'clustering_metrics.tsv'}")


if __name__ == '__main__':
    main()
