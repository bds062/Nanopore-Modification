#!/usr/bin/env python3
"""
orca_remake — unsupervised modification-TYPE annotation from RawMod's own
latent embeddings (the DNA analog of ORCA's autoencoder + transfer-learning
"label-transfer" typing module).

ORCA types a detected site by embedding it and comparing to a small labeled
reference set. We test the necessary precondition for that to work at all:
does RawMod's own embedding space separate 5mC / 5hmC / 6mA / 5hmU chemistry
WITHOUT ever being given a type label -- using only ONT (synthetic
benchmark) + SPO1/UMCES (native WGS + PCR-amplicon control) as a proof of
concept, per user direction.

Inputs (already produced by run_embed_extraction.sh + build_umces_type_labels.py):
  data/embeddings/{ONT_5mC,ONT_5hmC,ONT_6mA,ONT_control,
                   SPO1_bc06,SPO1_bc07,SPO1_amplicon}_{embed.npy,scores.tsv.gz}
  data/type_labels/SPO1_bc0{6,7}_types.npy

All embeddings come from ONE checkpoint, deepmod_ont+umces/results5/best_model.pt,
so they live in a single comparable space. That checkpoint was trained
EXCLUSIVELY on these 7 files (barcode_unmod=bc01-05 merged, barcode06,
barcode07, ONT 5mC/5hmC/6mA/control -- see its own checkpoint 'args'/'input'
list) and has never touched any dataset outside ONT+SPO1, per explicit user
direction. It is still an in-sample proof of concept (the model WAS trained
on exactly these files), not a zero-shot typing claim. Stated plainly here
and in report.md, not hidden.

Groups pooled (type, source):
  ONT_5mC       -> ('5mC',  'ONT')          label>0 sites in ONT_5mC.h5
  ONT_5hmC      -> ('5hmC', 'ONT')
  ONT_6mA       -> ('6mA',  'ONT')
  ONT_control   -> ('unmod','ONT')          label==0 (all sites; no true positive here)
  SPO1_bc06/07  -> ('5mC'/'5hmC'/'6mA'/'5hmU', 'SPO1_native')   via modkit-derived typing
  SPO1_bc06/07  -> ('unmod', 'SPO1_native')  label==0 sites within the native barcodes
  SPO1_amplicon -> ('unmod', 'SPO1_amplicon')  bc01-05 merged, PCR-stripped, all label==0

Outputs:
  figures/fig_annotation_embedding.png   -- 2x2 PCA/t-SNE scatter, colored by
                                             true type and by source dataset,
                                             with GMM ARI/NMI annotated on
                                             every panel
  data/annotation_clustering_report.txt  -- GMM cluster ARI/NMI vs type & source
"""
import gzip
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
sys.path.insert(0, str(REPO / 'experiments' / 'pipeline3'))
from cluster_types import bic_sweep, evaluate, gmm_cluster  # noqa: E402

EMBED_DIR = Path('/fs/cbcb-scratch/bds062/results/orca_remake/data/embeddings')
TYPE_DIR = Path('/fs/cbcb-scratch/bds062/results/orca_remake/data/type_labels')
OUT_FIG = Path('/fs/cbcb-scratch/bds062/results/orca_remake/figures/fig_annotation_embedding.png')
OUT_REPORT = Path('/fs/cbcb-scratch/bds062/results/orca_remake/data/annotation_clustering_report.txt')

MAX_PER_GROUP = 3000
SEED = 0

# categorical colors, fixed order (matches the project's existing figure palette)
TYPE_COLOR = {
    '5mC':   '#4878CF',
    '5hmC':  '#6ACC65',
    '6mA':   '#D65F5F',
    '5hmU':  '#956CB4',
    'unmod': '#B0B0B0',
}
SOURCE_COLOR = {
    'ONT':            '#4878CF',
    'SPO1_native':     '#D65F5F',
    'SPO1_amplicon':   '#B0B0B0',
}
TYPE_ORDER = ['5mC', '5hmC', '6mA', '5hmU', 'unmod']
SOURCE_ORDER = ['ONT', 'SPO1_native', 'SPO1_amplicon']


def load_labels(dataset):
    """label (0/1) array in the same row order as the embed.npy for `dataset`."""
    labels = []
    with gzip.open(EMBED_DIR / f'{dataset}_scores.tsv.gz', 'rt') as fh:
        next(fh)
        for line in fh:
            labels.append(int(line.rstrip('\n').split('\t')[3]))
    return np.array(labels)


def take_indices(pool, cap, rng):
    if len(pool) <= cap:
        return np.sort(pool)
    return np.sort(rng.choice(pool, cap, replace=False))


def main():
    rng = np.random.default_rng(SEED)
    embs, types, sources = [], [], []

    def add(dataset, mask, mtype, source):
        pool = np.nonzero(mask)[0]
        if len(pool) == 0:
            print(f"  [skip] {dataset} {mtype}: no sites")
            return
        idx = take_indices(pool, MAX_PER_GROUP, rng)
        e = np.load(EMBED_DIR / f'{dataset}_embed.npy').astype(np.float32)[idx]
        embs.append(e)
        types.extend([mtype] * len(idx))
        sources.extend([source] * len(idx))
        print(f"  {dataset:12} {mtype:6} ({source:14}) n={len(idx):,} (of {len(pool):,})")

    print("Pooling ONT (synthetic benchmark, one modification type per file):")
    for name, mtype in [('ONT_5mC', '5mC'), ('ONT_5hmC', '5hmC'), ('ONT_6mA', '6mA')]:
        lab = load_labels(name)
        add(name, lab > 0, mtype, 'ONT')
    lab = load_labels('ONT_control')
    add('ONT_control', lab == 0, 'unmod', 'ONT')

    print("\nPooling SPO1 native WGS (bc06/bc07, modkit-derived per-site typing):")
    for bc in ['SPO1_bc06', 'SPO1_bc07']:
        types_arr = np.load(TYPE_DIR / f'{bc}_types.npy')
        for mtype in ['5mC', '5hmC', '6mA', '5hmU']:
            add(bc, types_arr == mtype, mtype, 'SPO1_native')
        add(bc, types_arr == 'unmod', 'unmod', 'SPO1_native')

    print("\nPooling SPO1 PCR amplicon (bc01-05 merged, chemistry-stripped, all unmod):")
    lab = load_labels('SPO1_amplicon')
    add('SPO1_amplicon', lab == 0, 'unmod', 'SPO1_amplicon')

    X = np.vstack(embs)
    types = np.array(types)
    sources = np.array(sources)
    n_types = len(set(types))
    print(f"\npooled: {X.shape[0]:,} sites, {X.shape[1]}-dim embeddings, "
          f"{n_types} modification types, {len(set(sources))} source contexts")

    # ---- unsupervised clustering (reusing pipeline3's cluster_types utilities) ----
    report_lines = [
        "Unsupervised modification-type annotation from RawMod embeddings\n",
        "(ORCA fingerprinting analog; no type label given to the clustering,\n"
        " used only afterwards to score it)\n\n",
        "CAVEAT: embeddings come from deepmod_ont+umces/results5/best_model.pt,\n"
        "which WAS trained on exactly this ONT+SPO1 data (and nothing outside\n"
        "ONT+SPO1) -- an in-sample proof of concept, not a zero-shot-typing\n"
        "claim. See report.md.\n\n",
    ]
    sweep = bic_sweep(X, kmax=min(8, n_types + 3), seed=SEED)
    best_k = min(sweep, key=lambda t: t[1])[0]
    report_lines.append(f"BIC sweep: {sweep}\n")
    report_lines.append(f"BIC-selected components: {best_k}  (true #types = {n_types})\n\n")

    import io
    lab_pred, _ = gmm_cluster(X, n_types, SEED)
    fh = io.StringIO()
    res_all = evaluate(lab_pred, types, sources, "ALL TYPES (ONT + SPO1 native + SPO1 amplicon)", fh)
    print(fh.getvalue())
    report_lines.append(fh.getvalue() + "\n")

    # unconfounded subset: types present in BOTH ONT and SPO1_native (5mC, 5hmC, 6mA)
    multi = [t for t in set(types) if t != 'unmod'
             and len(set(sources[types == t]) & {'ONT', 'SPO1_native'}) == 2]
    report_lines.append(f"UNCONFOUNDED SUBSET (type present in both ONT and SPO1_native): {multi}\n")
    if len(multi) >= 2:
        m = np.isin(types, multi) & np.isin(sources, ['ONT', 'SPO1_native'])
        lab_m, _ = gmm_cluster(X[m], len(multi), SEED)
        fh2 = io.StringIO()
        evaluate(lab_m, types[m], sources[m], f"UNCONFOUNDED {multi} (ONT vs SPO1_native only)", fh2)
        report_lines.append(fh2.getvalue() + "\n")

    OUT_REPORT.write_text(''.join(report_lines))
    print(f"\nwrote {OUT_REPORT}")

    # ---- visualization: PCA + t-SNE, colored by type and by source ----
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    n_sub = min(8000, X.shape[0])
    sub_idx = rng.choice(X.shape[0], n_sub, replace=False) if X.shape[0] > n_sub \
        else np.arange(X.shape[0])
    tsne_in = PCA(n_components=32, random_state=SEED).fit_transform(Xs[sub_idx]) \
        if Xs.shape[1] > 32 else Xs[sub_idx]
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, init='pca').fit_transform(tsne_in)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11.5))

    # ARI/NMI is computed once, in the original (96-dim) embedding space, by
    # the GMM fit above -- it does not change between the PCA and t-SNE
    # panels below (those are visualization-only 2D projections of the same
    # clustering). Annotated on every panel so it's visible regardless of
    # which panel/projection the reader looks at.
    ari_box = (f"GMM clustering (in full embedding space):\n"
               f"ARI vs true type   = {res_all['ari_type']:.3f}\n"
               f"ARI vs source data = {res_all['ari_dataset']:.3f}")

    def scatter(ax, xy, labels, order, cmap, title):
        for lab in order:
            m = labels == lab
            if m.sum() == 0:
                continue
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, color=cmap[lab], label=f'{lab} (n={m.sum():,})')
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=2, loc='upper right')
        ax.text(0.02, 0.02, ari_box, transform=ax.transAxes, fontsize=8.5,
                va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#797979'))

    scatter(axes[0, 0], pca, types, TYPE_ORDER, TYPE_COLOR,
            'PCA of RawMod embeddings\ncolored by TRUE modification type')
    scatter(axes[0, 1], pca, sources, SOURCE_ORDER, SOURCE_COLOR,
            'PCA of RawMod embeddings\ncolored by SOURCE dataset')
    scatter(axes[1, 0], tsne, types[sub_idx], TYPE_ORDER, TYPE_COLOR,
            't-SNE of RawMod embeddings\ncolored by TRUE modification type')
    scatter(axes[1, 1], tsne, sources[sub_idx], SOURCE_ORDER, SOURCE_COLOR,
            't-SNE of RawMod embeddings\ncolored by SOURCE dataset')

    fig.suptitle(
        "Annotation figure: does RawMod's own latent embedding separate\n"
        "modification chemistry without ever being given a type label?\n"
        "(ORCA fingerprinting analog, ONT+SPO1-only checkpoint, proof of concept)",
        fontsize=12, y=0.995)
    fig.subplots_adjust(top=0.86, bottom=0.03, left=0.03, right=0.97,
                         hspace=0.25, wspace=0.15)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_FIG}")


if __name__ == '__main__':
    main()
