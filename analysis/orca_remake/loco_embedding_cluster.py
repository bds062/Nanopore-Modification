#!/usr/bin/env python3
"""
orca_remake, per-LOCO-fold — does each zero-shot Deep-SAD checkpoint's OWN
embedding space separate ITS held-out chemistry from the chemistries it did see?

The earlier sad_embedding_cluster.py used the 'mixed' checkpoint (trained
in-distribution on all 5 chemistries) and found the embedding organizes by
organism, not chemistry. But the AUROC numbers on methods_comparison.png (e.g.
6mA anomaly = 0.69) come from a DIFFERENT, more relevant checkpoint per
chemistry: the fold-specific zero-shot model results5/models/loco_<CHEM>/loco/
best_model.pt, which NEVER saw that chemistry in training. This script asks the
more faithful question: in the exact model that scored 0.69 zero-shot AUROC on
held-out 6mA, does 6mA actually sit apart from the modifications that model DID
train on? Same architecture (ConvFormerV2, curriculum + SupCon + Deep SAD) in
every fold -- only the held-out chemistry differs.

Same 27,900-image selection (all 5 chemistries + unmod, matched pool, 3
organisms) is embedded through EACH of the 5 fold checkpoints in turn, so the
comparison is apples-to-apples: same data, 5 different (architecture-identical)
models, each blind to one chemistry.

Two clustering questions per fold:
  1. Full 6-way GMM (as in sad_embedding_cluster.py) -- ARI vs type / organism,
     plus the unconfounded (>1 organism) subset.
  2. TARGETED: restricted to modified sites only, 2-way GMM with binary label
     (held-out chemistry) vs (all OTHER modified chemistries, pooled). High ARI
     here means the held-out/unseen chemistry forms its own cluster distinct
     from the modifications the model DID train on -- direct visual/quantitative
     evidence for what the anomaly-AUROC number is picking up on.

Outputs (under results4/embedding_clustering_per_fold/):
  embeddings_<CHEM>.npz            rep/sad embeddings from that fold's checkpoint
  clustering_report.txt            all 5 folds' full + targeted results
  clustering_metrics.tsv           machine-readable version
  figures/fig_loco_embedding_cluster.png   5-panel t-SNE grid, one per held-out
                                            chemistry, colored by type, that
                                            chemistry starred/highlighted
"""
import io
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
for _p in (REPO / 'analysis' / 'orca_remake',
           REPO / 'scripts' / 'train',
           REPO / 'scripts' / 'test',
           REPO / 'scripts' / 'train',
           REPO / 'rawmod'):
    sys.path.insert(0, str(_p))

import run_matched_loco as ML                                     # noqa: E402
from run_matched_loco import R                                    # noqa: E402
from sad_embedding_cluster import (build_selection, extract_embeddings,  # noqa: E402
                                   cluster_space, TYPE_COLOR, TYPE_ORDER,
                                   ORG_COLOR, ORG_ORDER, MAX_PER_GROUP)
from cluster_types import gmm_cluster                              # noqa: E402
from sklearn.decomposition import PCA                              # noqa: E402
from sklearn.manifold import TSNE                                  # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402

RESULTS5 = Path('/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results5')
OUT = Path('/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results4/'
           'embedding_clustering_per_fold')
SEED = 0
CHEMS = ('5mC', '5hmC', '6mA', '4mC', '5hmU')


def targeted_binary_test(rep, sad, types, held_out):
    """Restrict to modified sites; 2-way GMM(held_out vs other-seen-chem), scored
    by ARI/NMI against the true binary split. Answers: does the UNSEEN chemistry
    separate from chemistries this model DID train on, in its own space?"""
    mod_mask = types != 'unmod'
    t = types[mod_mask]
    y = (t == held_out).astype(int)
    out = {}
    for space_name, X in [('penultimate', rep[mod_mask]), ('sad', sad[mod_mask])]:
        lab, _ = gmm_cluster(X, 2, SEED)
        ari = adjusted_rand_score(y, lab)
        nmi = normalized_mutual_info_score(y, lab)
        out[space_name] = {'ari': ari, 'nmi': nmi}
    return out


def scatter_panel(ax, xy, types, held_out, title):
    for t in TYPE_ORDER:
        m = types == t
        if m.sum() == 0:
            continue
        if t == held_out:
            ax.scatter(xy[m, 0], xy[m, 1], s=22, alpha=0.85, color=TYPE_COLOR[t],
                      marker='*', edgecolor='black', linewidth=0.3,
                      label=f'{t} (HELD OUT, n={m.sum():,})', zorder=5)
        else:
            ax.scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.4, color=TYPE_COLOR[t],
                      label=f'{t} (n={m.sum():,})', zorder=3)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6.5, markerscale=1.3, loc='upper right')


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- same selection for every fold (apples-to-apples) ----
    members = ML.build_members()
    pool = R.Group(list(members), members)
    print(f"Matched pool: {pool.N:,} images across {len(members)} files", flush=True)
    mod_map = ML.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = ML.ref_base_center(pool)
    chem = ML.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0

    idx, types, orgs = build_selection(pool, chem, is_pos, refbase, rng)
    print(f"\nSelected {len(idx):,} images: {dict(Counter(types))}", flush=True)

    report = [
        "Per-LOCO-fold embedding structure: does each zero-shot checkpoint's OWN\n",
        "embedding space separate ITS held-out chemistry from ones it trained on?\n",
        "(Same architecture -- ConvFormerV2 + curriculum + SupCon + Deep SAD -- in\n",
        " every fold; only the excluded chemistry differs. Same 27,900-image\n",
        " selection embedded through each of the 5 checkpoints in turn.)\n\n",
    ]
    tsv_rows = []
    tsne_cache = {}

    for held_out in CHEMS:
        ckpt = RESULTS5 / 'models' / f'loco_{held_out}' / 'loco' / 'best_model.pt'
        print(f"\n=== fold loco_{held_out}: {ckpt} ===", flush=True)
        rep, sad = extract_embeddings(pool, idx, device, ckpt=ckpt)
        np.savez_compressed(OUT / f'embeddings_{held_out}.npz',
                            rep=rep.astype(np.float16), sad=sad.astype(np.float16),
                            types=types, orgs=orgs, idx=idx.astype(np.int64))

        report.append(f"\n================  held out: {held_out}  ================\n")
        rep_all, rep_sub = cluster_space(rep, types, orgs, f"penultimate ({held_out})", report)
        sad_all, sad_sub = cluster_space(sad, types, orgs, f"Deep-SAD space ({held_out})", report)

        tb = targeted_binary_test(rep, sad, types, held_out)
        fh = io.StringIO()
        fh.write(f"\nTARGETED: {held_out} (unseen) vs other-seen-chemistries, modified sites only\n")
        for space in ('penultimate', 'sad'):
            fh.write(f"  {space:11} 2-way GMM  ARI={tb[space]['ari']:.3f}  "
                     f"NMI={tb[space]['nmi']:.3f}\n")
        print(fh.getvalue(), flush=True)
        report.append(fh.getvalue())

        tsv_rows.append({
            'held_out': held_out, 'ari_type_all': rep_all['ari_type'],
            'ari_org_all': rep_all['ari_dataset'],
            'ari_type_unconf': rep_sub['ari_type'] if rep_sub else float('nan'),
            'ari_org_unconf': rep_sub['ari_dataset'] if rep_sub else float('nan'),
            'targeted_ari_penult': tb['penultimate']['ari'],
            'targeted_ari_sad': tb['sad']['ari'],
        })

        # cache a 2D projection for the combined figure (t-SNE on a subsample)
        Xs = (rep - rep.mean(0)) / (rep.std(0) + 1e-8)
        n_sub = min(6000, len(idx))
        sub = rng.choice(len(idx), n_sub, replace=False) if len(idx) > n_sub else np.arange(len(idx))
        tsne_in = PCA(n_components=32, random_state=SEED).fit_transform(Xs[sub]) \
            if Xs.shape[1] > 32 else Xs[sub]
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init='pca').fit_transform(tsne_in)
        tsne_cache[held_out] = (tsne, types[sub])

        del rep, sad
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    (OUT / 'clustering_report.txt').write_text(''.join(report))
    with open(OUT / 'clustering_metrics.tsv', 'w') as fh:
        cols = ['held_out', 'ari_type_all', 'ari_org_all', 'ari_type_unconf',
                'ari_org_unconf', 'targeted_ari_penult', 'targeted_ari_sad']
        fh.write('\t'.join(cols) + '\n')
        for r in tsv_rows:
            fh.write('\t'.join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                               for c in cols) + '\n')
    print(f"\nwrote {OUT/'clustering_report.txt'}\nwrote {OUT/'clustering_metrics.tsv'}")

    # ---- combined figure: 1 row x 5 panels, one per held-out chemistry ----
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    for ax, held_out in zip(axes, CHEMS):
        tsne, ty = tsne_cache[held_out]
        scatter_panel(ax, tsne, ty, held_out,
                     f"held out: {held_out}\n(t-SNE of loco_{held_out}'s own embedding)")
    fig.suptitle(
        "Does each zero-shot checkpoint's OWN embedding separate its held-out\n"
        "chemistry (star) from the chemistries it DID train on? Same architecture,\n"
        "5 different fold-specific Deep-SAD models, colored by TRUE modification type.",
        fontsize=12.5, y=1.06)
    fig.subplots_adjust(top=0.78, bottom=0.03, left=0.02, right=0.98, wspace=0.12)
    fig.savefig(OUT / 'figures' / 'fig_loco_embedding_cluster.png', dpi=150,
               bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {OUT/'figures'/'fig_loco_embedding_cluster.png'}")


if __name__ == '__main__':
    main()
