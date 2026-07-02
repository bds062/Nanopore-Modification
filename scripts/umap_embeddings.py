#!/usr/bin/env python3
"""
umap_embeddings.py
==================
Extract penultimate-layer (512-dim) embeddings from PileupInceptionV3 and
visualise them with UMAP (or t-SNE if umap-learn is not installed).

Inputs
------
--predictions   One or more test_predictions.npz files produced by model.py.
                Each .npz encodes h5_paths, test_indices, and test_file_idx.
--model         Path to best_model.pt checkpoint (dict with 'model_state' key).
                Defaults to best_model.pt in the same dir as the first --predictions.

Outputs
-------
<out-dir>/embeddings.npz   Raw embeddings + metadata for later reuse.
<out-dir>/umap_dataset.png UMAP coloured by dataset source.
<out-dir>/umap_label.png   UMAP coloured by true modification label.
<out-dir>/umap_prob.png    UMAP coloured by predicted probability.

Usage examples
--------------
  # UMCES-only model, test set only:
  python umap_embeddings.py \\
      --predictions /fs/cbcb-scratch/bds062/results/deepmod_umces/results4/test_predictions.npz \\
      --out-dir /fs/cbcb-scratch/bds062/results/deepmod_umces/results4/umap/

  # Combined ONT+UMCES model:
  python umap_embeddings.py \\
      --predictions /fs/cbcb-scratch/bds062/results/deepmod_ont+umces/results/test_predictions.npz \\
      --out-dir /fs/cbcb-scratch/bds062/results/deepmod_ont+umces/results/umap/

  # Load embeddings from a previous run (skip inference):
  python umap_embeddings.py \\
      --embeddings /path/to/embeddings.npz \\
      --out-dir /path/to/umap/
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cfg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── optional reducers ─────────────────────────────────────────────────────────

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# ── constants ─────────────────────────────────────────────────────────────────

DATASET_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]


# ── model import ─────────────────────────────────────────────────────────────

def _import_model_class():
    """Import PileupInceptionV3 from deepmod package or from this file's sibling."""
    try:
        from deepmod.model import PileupInceptionV3, PileupDataset, _worker_init_fn
        return PileupInceptionV3, PileupDataset, _worker_init_fn
    except ImportError:
        pass
    # Try sibling directory (scripts/ → deepmod/)
    here = Path(__file__).resolve().parent
    siblings = [here.parent / "deepmod", here / "deepmod", here]
    for p in siblings:
        if (p / "model.py").exists():
            sys.path.insert(0, str(p.parent))
            from deepmod.model import PileupInceptionV3, PileupDataset, _worker_init_fn
            return PileupInceptionV3, PileupDataset, _worker_init_fn
    raise ImportError("Cannot locate deepmod.model — run from the repo root or set PYTHONPATH.")


# ── checkpoint loading ────────────────────────────────────────────────────────

def load_model(model_path: Path, device: torch.device):
    PileupInceptionV3, _, _ = _import_model_class()
    ckpt = torch.load(model_path, map_location="cpu")
    in_channels = ckpt.get("in_channels", 9)
    model = PileupInceptionV3(in_channels=in_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"  Loaded model from epoch {ckpt.get('epoch', '?')}  "
          f"val_auprc={ckpt.get('val_auprc', float('nan')):.4f}  "
          f"in_channels={in_channels}")
    return model


# ── embedding extraction ──────────────────────────────────────────────────────

def extract_embeddings(model, h5_paths, indices, file_idx,
                       batch_size, device, delta_channels=True):
    """
    Run inference with a forward hook on model.head[1] (Flatten, 512-dim).
    Returns embeddings (N, 512) and predicted probabilities (N,).
    """
    _, PileupDataset, _worker_init_fn = _import_model_class()

    # Compute file_sizes from h5 files
    file_sizes = []
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            file_sizes.append(len(f["labels"]))
    file_sizes = np.array(file_sizes, dtype=np.int64)

    dataset = PileupDataset(
        h5_paths=h5_paths,
        indices=indices,
        file_sizes=file_sizes,
        augment=False,
        delta_channels=delta_channels,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=_worker_init_fn,
    )

    # Hook the Flatten layer (head[1]) to capture 512-dim embeddings
    emb_buf = []
    def _hook(module, input, output):
        emb_buf.append(output.detach().cpu().numpy())

    handle = model.head[1].register_forward_hook(_hook)

    probs = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)                      # triggers hook
            probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())

    handle.remove()

    embeddings = np.concatenate(emb_buf, axis=0)   # (N, 512)
    probs_arr  = np.concatenate(probs,   axis=0)   # (N,)
    return embeddings, probs_arr


# ── data loading from predictions .npz ───────────────────────────────────────

def _h5_labels_unsorted(h5file, local_idx: np.ndarray) -> np.ndarray:
    """Fetch labels from an open h5py dataset with potentially unsorted indices.
    h5py fancy indexing requires strictly increasing indices, so we sort,
    fetch, then invert the permutation."""
    order   = np.argsort(local_idx)
    fetched = h5file["labels"][local_idx[order]]
    result  = np.empty_like(fetched)
    result[order] = fetched
    return result


def load_from_predictions(npz_path: Path, split: str, max_per_dataset: int):
    """
    Load indices, file_idx, and labels from a test_predictions.npz.
    split: 'test' | 'train' | 'val' | 'all'
    """
    d = np.load(npz_path, allow_pickle=True)
    h5_paths = list(d["h5_paths"])

    if split == "test":
        indices  = d["test_indices"]
        file_idx = d["test_file_idx"]
        labels   = d["image_y_true"].astype(int)
    elif split == "train":
        indices  = d["train_indices"]
        file_sizes = []
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                file_sizes.append(len(f["labels"]))
        file_sizes = np.array(file_sizes)
        offsets = np.concatenate([[0], np.cumsum(file_sizes)])
        file_idx = np.searchsorted(offsets[1:], indices, side="right").astype(np.int64)
        labels = np.empty(len(indices), dtype=int)
        for fi, p in enumerate(h5_paths):
            mask = file_idx == fi
            if not mask.any():
                continue
            local_idx = (indices[mask] - offsets[fi]).astype(np.int64)
            with h5py.File(p, "r") as f:
                labels[mask] = _h5_labels_unsorted(f, local_idx)
    elif split == "all":
        test_i  = d["test_indices"];  test_fi = d["test_file_idx"]
        train_i = d["train_indices"]; val_i   = d["val_indices"]
        file_sizes = []
        for p in h5_paths:
            with h5py.File(p, "r") as f:
                file_sizes.append(len(f["labels"]))
        file_sizes = np.array(file_sizes)
        offsets = np.concatenate([[0], np.cumsum(file_sizes)])
        tv_i  = np.concatenate([train_i, val_i])
        tv_fi = np.searchsorted(offsets[1:], tv_i, side="right").astype(np.int64)
        indices  = np.concatenate([test_i,  tv_i])
        file_idx = np.concatenate([test_fi, tv_fi])
        test_labels = d["image_y_true"].astype(int)
        tv_labels = np.empty(len(tv_i), dtype=int)
        for fi, p in enumerate(h5_paths):
            mask = tv_fi == fi
            if not mask.any():
                continue
            local_idx = (tv_i[mask] - offsets[fi]).astype(np.int64)
            with h5py.File(p, "r") as f:
                tv_labels[mask] = _h5_labels_unsorted(f, local_idx)
        labels = np.concatenate([test_labels, tv_labels])
    else:
        raise ValueError(f"Unknown split: {split!r}")

    # Per-dataset subsampling
    rng = np.random.default_rng(42)
    keep = []
    for fi in np.unique(file_idx):
        mask = np.where(file_idx == fi)[0]
        if len(mask) > max_per_dataset:
            mask = rng.choice(mask, size=max_per_dataset, replace=False)
        keep.append(mask)
    keep = np.sort(np.concatenate(keep))

    return (h5_paths,
            indices[keep],
            file_idx[keep],
            labels[keep])


# ── dimensionality reduction ──────────────────────────────────────────────────

def reduce(embeddings: np.ndarray, method: str, n_neighbors: int, min_dist: float,
           n_components: int = 2):
    if method == "umap":
        if not HAS_UMAP:
            print("  umap-learn not found — falling back to t-SNE.")
            return reduce(embeddings, "tsne", n_neighbors, min_dist, n_components)
        reducer = umap_lib.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=True,
        )
        return reducer.fit_transform(embeddings)
    elif method == "tsne":
        # PCA first to 50 dims to speed up t-SNE on large inputs
        n_pca = min(50, embeddings.shape[1], embeddings.shape[0] - 1)
        pca_emb = PCA(n_components=n_pca, random_state=42).fit_transform(embeddings)
        return TSNE(
            n_components=n_components,
            perplexity=min(30, len(embeddings) - 1),
            random_state=42,
            n_iter=1000,
            verbose=1,
        ).fit_transform(pca_emb)
    elif method == "pca":
        return PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ── plotting ──────────────────────────────────────────────────────────────────

def _scatter(ax, xy, c, cmap=None, vmin=None, vmax=None,
             alpha=0.5, s=6, title=""):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap,
                    vmin=vmin, vmax=vmax, alpha=alpha, s=s, linewidths=0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def plot_by_dataset(xy, file_idx, h5_paths, out_path, method):
    dataset_names = [Path(p).stem for p in h5_paths]
    n_datasets = len(dataset_names)
    colors = DATASET_COLORS[:n_datasets]

    fig, ax = plt.subplots(figsize=(8, 7))
    for fi, (name, color) in enumerate(zip(dataset_names, colors)):
        mask = file_idx == fi
        if not mask.any():
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=color, alpha=0.5, s=6, linewidths=0, label=name)
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_title(f"{method.upper()} — coloured by dataset", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_label(xy, labels, out_path, method):
    fig, ax = plt.subplots(figsize=(7, 6))
    for lval, lname, color in [(0, "unmod", "#377eb8"), (1, "modified", "#e41a1c")]:
        mask = labels == lval
        if not mask.any():
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=color, alpha=0.5, s=6, linewidths=0, label=lname)
    ax.legend(markerscale=3, fontsize=9, loc="best")
    ax.set_title(f"{method.upper()} — coloured by true label", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_dataset_and_label(xy, file_idx, labels, h5_paths, out_path, method):
    """2x2 (or 1x2) grid: one panel per dataset, coloured by label."""
    dataset_names = [Path(p).stem for p in h5_paths]
    unique_fi = sorted(np.unique(file_idx))
    n = len(unique_fi)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2), squeeze=False)
    for ax_idx, fi in enumerate(unique_fi):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        mask = file_idx == fi
        for lval, lname, color in [(0, "unmod", "#377eb8"), (1, "mod", "#e41a1c")]:
            m = mask & (labels == lval)
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1],
                           c=color, alpha=0.5, s=5, linewidths=0, label=lname)
        ax.set_title(dataset_names[fi], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=2, loc="best")
    # hide unused axes
    for ax_idx in range(n, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)
    fig.suptitle(f"{method.upper()} — per-dataset, coloured by label", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_prob(xy, probs, labels, out_path, method):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = _scatter(axes[0], xy, probs, cmap="RdBu_r", vmin=0, vmax=1,
                  title=f"{method.upper()} — predicted probability")
    plt.colorbar(sc, ax=axes[0], label="P(modified)")

    # Error: FP in orange, FN in green, TP/TN in grey/red
    err = np.zeros(len(labels), dtype=int)     # 0 = correct
    threshold = 0.5
    pred = (probs >= threshold).astype(int)
    tp  = (pred == 1) & (labels == 1)
    tn  = (pred == 0) & (labels == 0)
    fp  = (pred == 1) & (labels == 0)
    fn  = (pred == 0) & (labels == 1)

    ax = axes[1]
    for mask, color, label in [
        (tn,  "#aec7e8", f"TN ({tn.sum()})"),
        (tp,  "#e41a1c", f"TP ({tp.sum()})"),
        (fp,  "#ff7f00", f"FP ({fp.sum()})"),
        (fn,  "#4daf4a", f"FN ({fn.sum()})"),
    ]:
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=color, alpha=0.6, s=6, linewidths=0, label=label)
    ax.legend(markerscale=3, fontsize=8)
    ax.set_title(f"{method.upper()} — TP/TN/FP/FN (thresh={threshold})", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── loading from standalone h5 files (eval datasets) ─────────────────────────

def load_from_h5_direct(h5_paths: list, max_per_dataset: int):
    """
    Load all samples from a list of standalone h5 files (no predictions .npz).
    Returns global indices, file_idx, and labels ready for extract_embeddings.
    """
    rng = np.random.default_rng(42)
    all_indices  = []
    all_file_idx = []
    all_labels   = []
    offset = 0

    for fi, p in enumerate(h5_paths):
        with h5py.File(p, "r") as f:
            n     = len(f["labels"])
            lbls  = f["labels"][:].astype(int)

        idx = np.arange(n, dtype=np.int64)
        if n > max_per_dataset:
            idx = rng.choice(idx, size=max_per_dataset, replace=False)
            idx = np.sort(idx)

        all_indices.append(idx + offset)
        all_file_idx.append(np.full(len(idx), fi, dtype=np.int64))
        all_labels.append(lbls[idx])
        offset += n

        n_mod = int(lbls[idx].sum())
        print(f"  {Path(p).stem:<30}  N={len(idx):>5}  mod={n_mod:>4}")

    return (np.concatenate(all_indices),
            np.concatenate(all_file_idx),
            np.concatenate(all_labels))


def run_eval_split(h5_paths: list, out_dir: Path, model, device, args):
    """Load all images from standalone eval h5 files, embed, reduce, and plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Split: eval  →  {out_dir}")
    print(f"{'='*60}")

    indices, file_idx, labels = load_from_h5_direct(h5_paths, args.max_per_dataset)

    print(f"\nExtracting embeddings ({len(indices)} samples) ...")
    embeddings, probs = extract_embeddings(
        model, h5_paths, indices, file_idx,
        batch_size=args.batch, device=device,
        delta_channels=not args.no_delta_channels,
    )

    emb_path = out_dir / "embeddings.npz"
    np.savez(emb_path,
             embeddings=embeddings, labels=labels,
             probs=probs, file_idx=file_idx,
             h5_paths=np.array(h5_paths, dtype=object))
    print(f"  Saved embeddings → {emb_path}")

    print(f"\nRunning {args.method.upper()} on "
          f"{len(embeddings)} × {embeddings.shape[1]} embeddings ...")
    xy = reduce(embeddings, args.method, args.n_neighbors, args.min_dist)
    np.savez(out_dir / "reduction.npz", xy=xy, method=args.method)
    print(f"  Saved reduction → {out_dir / 'reduction.npz'}")

    print("\nGenerating plots ...")
    plot_by_dataset(
        xy, file_idx, h5_paths,
        out_dir / f"{args.method}_dataset.png", args.method)
    plot_by_label(
        xy, labels,
        out_dir / f"{args.method}_label.png", args.method)
    plot_by_dataset_and_label(
        xy, file_idx, labels, h5_paths,
        out_dir / f"{args.method}_per_dataset_label.png", args.method)
    plot_by_prob(
        xy, probs, labels,
        out_dir / f"{args.method}_prob.png", args.method)

    print(f"  Done — outputs in {out_dir}")


# ── per-split runner ─────────────────────────────────────────────────────────

def run_split(split: str, out_dir: Path, prediction_paths: list, model,
              device, args):
    """Load data for one split, extract embeddings, reduce, and plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Split: {split}  →  {out_dir}")
    print(f"{'='*60}")

    all_h5_paths = []
    all_indices  = []
    all_file_idx = []
    all_labels   = []
    global_fi_offset = 0

    for npz_path in prediction_paths:
        npz_path = Path(npz_path)
        print(f"\nLoading {npz_path} ...")
        h5_paths_i, indices_i, file_idx_i, labels_i = load_from_predictions(
            npz_path, split, args.max_per_dataset)
        all_h5_paths.extend(h5_paths_i)
        all_indices.append(indices_i)
        all_file_idx.append(file_idx_i + global_fi_offset)
        all_labels.append(labels_i)
        global_fi_offset += len(h5_paths_i)

        for fi in sorted(np.unique(file_idx_i)):
            mask = file_idx_i == fi
            cnt  = mask.sum()
            n_mod = labels_i[mask].sum()
            print(f"  {Path(h5_paths_i[fi]).stem:<30}  N={cnt:>5}  mod={n_mod:>4}")

    all_indices  = np.concatenate(all_indices)
    all_file_idx = np.concatenate(all_file_idx)
    all_labels   = np.concatenate(all_labels)

    print(f"\nExtracting embeddings ({len(all_indices)} samples) ...")
    embeddings, probs = extract_embeddings(
        model, all_h5_paths, all_indices, all_file_idx,
        batch_size=args.batch, device=device,
        delta_channels=not args.no_delta_channels,
    )

    emb_path = out_dir / "embeddings.npz"
    np.savez(emb_path,
             embeddings=embeddings, labels=all_labels,
             probs=probs, file_idx=all_file_idx,
             h5_paths=np.array(all_h5_paths, dtype=object))
    print(f"  Saved embeddings → {emb_path}")

    print(f"\nRunning {args.method.upper()} on "
          f"{len(embeddings)} × {embeddings.shape[1]} embeddings ...")
    xy = reduce(embeddings, args.method, args.n_neighbors, args.min_dist)
    np.savez(out_dir / "reduction.npz", xy=xy, method=args.method)
    print(f"  Saved reduction → {out_dir / 'reduction.npz'}")

    print("\nGenerating plots ...")
    plot_by_dataset(
        xy, all_file_idx, all_h5_paths,
        out_dir / f"{args.method}_dataset.png", args.method)
    plot_by_label(
        xy, all_labels,
        out_dir / f"{args.method}_label.png", args.method)
    plot_by_dataset_and_label(
        xy, all_file_idx, all_labels, all_h5_paths,
        out_dir / f"{args.method}_per_dataset_label.png", args.method)
    plot_by_prob(
        xy, probs, all_labels,
        out_dir / f"{args.method}_prob.png", args.method)

    print(f"  Done — outputs in {out_dir}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", metavar="NPZ",
                    help="test_predictions.npz file(s) from model.py")
    ap.add_argument("--model", metavar="PT",
                    help="best_model.pt checkpoint (default: sibling of first --predictions)")
    ap.add_argument("--embeddings", metavar="NPZ",
                    help="Pre-computed embeddings.npz — skip inference, re-plot only")
    ap.add_argument("--out-dir", required=True, metavar="DIR")
    ap.add_argument("--split", default="both",
                    choices=["test", "train", "val", "all", "both"],
                    help="Split(s) to embed: 'both' (default) runs train+test into "
                         "train/ and test/ subdirs; others go directly into --out-dir")
    ap.add_argument("--max-per-dataset", type=int, default=2000,
                    help="Max samples per dataset per split (default: 2000)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto",
                    help="cuda | cpu | auto (default: auto)")
    ap.add_argument("--method", default="umap",
                    choices=["umap", "tsne", "pca"],
                    help="Dimensionality reduction method (default: umap)")
    ap.add_argument("--n-neighbors", type=int, default=15,
                    help="UMAP n_neighbors (default: 15)")
    ap.add_argument("--min-dist", type=float, default=0.1,
                    help="UMAP min_dist (default: 0.1)")
    ap.add_argument("--eval-h5", nargs="+", metavar="H5",
                    help="Standalone h5 files for eval/ subfolder (e.g. barcode01_test.h5 "
                         "6mA.h5). All images are used (subject to --max-per-dataset).")
    ap.add_argument("--no-delta-channels", action="store_true",
                    help="Disable delta channel augmentation (match training flag if used)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── re-plot from pre-computed embeddings (no inference needed) ────────────
    if args.embeddings:
        print(f"\nLoading pre-computed embeddings from {args.embeddings}")
        saved      = np.load(args.embeddings, allow_pickle=True)
        embeddings = saved["embeddings"]
        labels     = saved["labels"]
        probs      = saved["probs"]
        file_idx   = saved["file_idx"]
        h5_paths   = list(saved["h5_paths"])

        print(f"\nRunning {args.method.upper()} on "
              f"{len(embeddings)} × {embeddings.shape[1]} embeddings ...")
        xy = reduce(embeddings, args.method, args.n_neighbors, args.min_dist)
        np.savez(out_dir / "reduction.npz", xy=xy, method=args.method)

        plot_by_dataset(xy, file_idx, h5_paths,
                        out_dir / f"{args.method}_dataset.png", args.method)
        plot_by_label(xy, labels,
                      out_dir / f"{args.method}_label.png", args.method)
        plot_by_dataset_and_label(xy, file_idx, labels, h5_paths,
                                  out_dir / f"{args.method}_per_dataset_label.png",
                                  args.method)
        plot_by_prob(xy, probs, labels,
                     out_dir / f"{args.method}_prob.png", args.method)
        print(f"\nDone. All outputs in {out_dir}")
        return

    # ── normal path: run inference then plot ──────────────────────────────────
    if not args.predictions:
        ap.error("Provide --predictions or --embeddings.")

    model_path = Path(args.model) if args.model else \
                 Path(args.predictions[0]).parent / "best_model.pt"
    print(f"\nLoading model from {model_path}")
    model = load_model(model_path, device)

    if args.split == "both":
        run_split("train", out_dir / "train", args.predictions, model, device, args)
        run_split("test",  out_dir / "test",  args.predictions, model, device, args)
        if args.eval_h5:
            run_eval_split(args.eval_h5, out_dir / "eval", model, device, args)
        subdirs = "train/, test/" + (", eval/" if args.eval_h5 else "")
        print(f"\nAll done. Outputs in {out_dir}/{{{subdirs}}}")
    else:
        # Single split — output goes directly into out_dir (no subdir)
        run_split(args.split, out_dir, args.predictions, model, device, args)
        print(f"\nAll done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
