#!/usr/bin/env python3
"""
test_model.py — evaluate one trained deepmod checkpoint on a new, held-out
featurized dataset and produce accuracy figures.

Not tied to any particular results run: point --model at any
resultsN/models/<model>/<tag>/best_model.pt checkpoint and --h5 at any
features.h5 (or several, comma-separated) produced by featurization.py.
Unlike run_pipeline.py's train/val/test splitting, this script treats the
entire h5 as one held-out test set — it never trains or updates weights.

If the test set is all one class (e.g. a whole-genome-amplified/canonical
control where every site is unmodified), class-contrastive metrics that need
both classes (AUROC, AUPRC, ROC/PR curves) are skipped automatically and the
script reports specificity/false-positive-rate instead.

Usage:
  python test_model.py \\
      --model results6/models/both/base/best_model.pt \\
      --h5 /fs/cbcb-scratch/bds062/results/benchmark_results/HP26695_WGA_5kHz/features.h5 \\
      --dataset-name HP26695_WGA \\
      --out-dir results6/test_HP26695_WGA

Outputs (under --out-dir):
  metrics.tsv                          one-row summary of accuracy metrics
  score_hist_<dataset-name>.png        predicted-score distribution by true label
  roc_pr_<dataset-name>.png            ROC + PR curves (only if both classes present)
  threshold_sweep_<dataset-name>.png   FPR / TPR / accuracy across thresholds
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, confusion_matrix,
)

DEEPMOD = Path('/fs/nexus-scratch/bds062/Nanopore-Modification/deepmod')
PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(DEEPMOD))
sys.path.insert(0, str(PIPE))

from model import (                                    # noqa: E402
    PileupDataset, run_inference, aggregate_by_position,
    make_position_keys, source_position_keys, binary_labels, make_loader_kwargs,
    _worker_init_fn,
)

COLOR_MOD, COLOR_UNMOD, COLOR_ACC = '#D65F5F', '#4878CF', '#6ACC65'

# Every resultsN run in this directory trains a different architecture on the
# identical (11, 31, 210) pileup tensor via run_pipeline.py's model_factory
# hook, and none of the checkpoints record which one made them — so detect it
# from the state-dict's key shapes rather than assume PileupInceptionV3.
ARCHS = ('inception', 'mlp', 'convformer', 'convformer_v2')


def detect_arch(sd_keys) -> str:
    keys = set(sd_keys)
    if any(k.startswith('read_encoder.blockA') for k in keys):
        return 'convformer_v2'          # results6 (run_convformer_v2.py)
    if any(k.startswith('read_encoder.') for k in keys):
        return 'convformer'             # results5 (run_convformer.py)
    if any(k.startswith('net.') for k in keys):
        return 'mlp'                    # results3 (run_mlp.py)
    return 'inception'                  # results1/2/4 (model.py PileupInceptionV3)


def build_model(arch: str, in_channels: int, sd: dict):
    if arch == 'inception':
        from model import PileupInceptionV3
        cross_read_attention = any('cross_read_attn' in k for k in sd)
        supcon_proj_dim = 0
        for k, v in sd.items():
            # ProjectionHead.net = Sequential(Linear, ReLU, Linear) — the
            # *second* Linear's output dim is the real projection dim.
            if k.endswith('proj_head.net.2.weight'):
                supcon_proj_dim = v.shape[0]
                break
        return PileupInceptionV3(in_channels=in_channels, dropout=0.0,
                                 cross_read_attention=cross_read_attention,
                                 supcon_proj_dim=supcon_proj_dim)
    if arch == 'mlp':
        from run_mlp import PileupMLP
        return PileupMLP(dropout=0.0)
    if arch == 'convformer':
        from run_convformer import ConvFormer
        return ConvFormer(dropout=0.0)
    if arch == 'convformer_v2':
        from run_convformer_v2 import ConvFormerV2
        return ConvFormerV2(dropout=0.0)
    raise ValueError(f"Unknown --arch {arch!r}; choose from {ARCHS}")


def load_model(ckpt_path: str, device: torch.device, arch: str = 'auto'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['model_state']
    in_channels = ckpt.get('in_channels', 11)
    if arch == 'auto':
        arch = detect_arch(sd.keys())
        print(f"  auto-detected architecture: {arch}", file=sys.stderr)
    model = build_model(arch, in_channels, sd)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != 'model_state'}
    meta['arch'] = arch
    return model, meta


def load_test_set(h5_paths: list[str]):
    file_sizes = []
    ref_names_all, ref_pos_all, labels_all = [], [], []
    for p in h5_paths:
        with h5py.File(p, 'r') as hf:
            n = hf['tensors'].shape[0]
            file_sizes.append(n)
            ref_names_all.append(hf['ref_names'][:])
            ref_pos_all.append(hf['ref_pos'][:])
            labels_all.append(hf['labels'][:])
    file_sizes = np.array(file_sizes, dtype=np.int64)
    ref_names = np.concatenate(ref_names_all)
    ref_pos = np.concatenate(ref_pos_all)
    labels = np.concatenate(labels_all)
    indices = np.arange(int(file_sizes.sum()), dtype=np.int64)
    position_keys = make_position_keys(ref_names, ref_pos)
    return indices, file_sizes, labels, position_keys


def safe_metric(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception:
        return float('nan')


def plot_confusion_matrix(tn, fp, fn_, tp, name, threshold, out_path):
    cm = np.array([[tn, fp], [fn_, tp]], dtype=np.int64)
    cm_frac = cm / max(cm.sum(), 1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_frac, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Unmodified', 'Modified'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Unmodified', 'Modified'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground truth')
    ax.set_title(f'{name}: confusion matrix (threshold={threshold:g})')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_frac[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}\n({cm_frac[i, j]:.1%})',
                   ha='center', va='center', color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Fraction of positions')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_score_hist(y_true, y_prob, name, threshold, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    if (y_true == 1).any():
        ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, label='Modified (GT)',
                color=COLOR_MOD, density=True)
    if (y_true == 0).any():
        ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, label='Unmodified (GT)',
                color=COLOR_UNMOD, density=True)
    ax.axvline(threshold, color='k', linestyle='--', linewidth=1,
              label=f'threshold={threshold:g}')
    ax.set_xlabel('Predicted P(modified)')
    ax.set_ylabel('Density')
    ax.set_title(f'{name}: predicted-score distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_roc_pr(y_true, y_prob, name, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(fpr, tpr, color=COLOR_MOD, linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'ROC (AUROC={auroc:.3f})')

    axes[1].plot(rec, prec, color=COLOR_UNMOD, linewidth=2)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'PR (AUPRC={auprc:.3f})')

    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_threshold_sweep(y_true, y_prob, name, out_path):
    thresholds = np.linspace(0.01, 0.99, 50)
    has_pos = (y_true == 1).any()
    fprs, tprs, accs = [], [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(np.int64)
        tn, fp, fn_, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fprs.append(fp / max(tn + fp, 1))
        tprs.append(tp / max(tp + fn_, 1) if has_pos else np.nan)
        accs.append((tp + tn) / max(len(y_true), 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, fprs, label='False Positive Rate', color=COLOR_MOD)
    if has_pos:
        ax.plot(thresholds, tprs, label='True Positive Rate', color=COLOR_ACC)
    ax.plot(thresholds, accs, label='Accuracy', color=COLOR_UNMOD, linestyle='--')
    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('Rate')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'{name}: threshold sweep')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True,
                    help='Path to a best_model.pt checkpoint (any resultsN run)')
    ap.add_argument('--h5', required=True,
                    help='Comma-separated features.h5 test file(s)')
    ap.add_argument('--dataset-name', required=True,
                    help='Label used in output filenames/figure titles')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--arch', default='auto', choices=('auto',) + ARCHS,
                    help='Model architecture the checkpoint was trained with. '
                         '"auto" detects it from the state-dict keys (default).')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    print(f"Loading model: {args.model}", flush=True)
    model, meta = load_model(args.model, device, arch=args.arch)
    print(f"  checkpoint meta: {meta}", flush=True)

    h5_paths = [p.strip() for p in args.h5.split(',') if p.strip()]
    print(f"Loading test set: {h5_paths}", flush=True)
    indices, file_sizes, labels, position_keys = load_test_set(h5_paths)
    n_images = len(indices)
    labels_bin = binary_labels(labels)
    print(f"  {n_images:,} images  ({int(labels_bin.sum()):,} modified / "
          f"{int((1 - labels_bin).sum()):,} unmodified, image-level)", flush=True)

    dataset = PileupDataset(h5_paths, indices, file_sizes, augment=False,
                            delta_channels=True)
    loader_kwargs = make_loader_kwargs(args.batch_size, args.num_workers, device,
                                       worker_init_fn=_worker_init_fn)
    loader = DataLoader(dataset, shuffle=False, **loader_kwargs)

    print("Running inference ...", flush=True)
    y_true_img, y_prob_img = run_inference(model, loader, device)
    y_true_img = binary_labels(y_true_img)

    base_keys, _ = source_position_keys(indices, position_keys, file_sizes)
    y_true, y_prob, _ = aggregate_by_position(y_true_img, y_prob_img, base_keys)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    both_classes = n_pos > 0 and n_neg > 0
    print(f"  {len(y_true):,} positions after aggregation "
          f"({n_pos:,} modified / {n_neg:,} unmodified)", flush=True)

    y_pred = (y_prob >= args.threshold).astype(np.int64)
    tn, fp, fn_, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        'dataset': args.dataset_name,
        'model': str(args.model),
        'n_images': n_images,
        'n_positions': len(y_true),
        'n_pos_positions': n_pos,
        'n_neg_positions': n_neg,
        'threshold': args.threshold,
        'accuracy': (tp + tn) / max(len(y_true), 1),
        'specificity': tn / max(tn + fp, 1),
        'fpr': fp / max(tn + fp, 1),
        'sensitivity_recall': (tp / max(tp + fn_, 1)) if n_pos > 0 else float('nan'),
        'precision': safe_metric(precision_score, y_true, y_pred, zero_division=0),
        'f1': safe_metric(f1_score, y_true, y_pred, zero_division=0),
        'auroc': safe_metric(roc_auc_score, y_true, y_prob) if both_classes else float('nan'),
        'auprc': (safe_metric(average_precision_score, y_true, y_prob)
                  if both_classes else float('nan')),
        'mean_score_pos': float(y_prob[y_true == 1].mean()) if n_pos else float('nan'),
        'mean_score_neg': float(y_prob[y_true == 0].mean()) if n_neg else float('nan'),
    }

    tsv_path = out_dir / 'metrics.tsv'
    with open(tsv_path, 'w') as f:
        f.write('\t'.join(metrics.keys()) + '\n')
        f.write('\t'.join(str(v) for v in metrics.values()) + '\n')
    print(f"\nWrote {tsv_path}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print()
    plot_score_hist(y_true, y_prob, args.dataset_name, args.threshold,
                    out_dir / f'score_hist_{args.dataset_name}.png')
    if both_classes:
        plot_roc_pr(y_true, y_prob, args.dataset_name,
                   out_dir / f'roc_pr_{args.dataset_name}.png')
    else:
        print("  (skipping ROC/PR curves — test set has only one true class)")
    plot_threshold_sweep(y_true, y_prob, args.dataset_name,
                         out_dir / f'threshold_sweep_{args.dataset_name}.png')
    plot_confusion_matrix(tn, fp, fn_, tp, args.dataset_name, args.threshold,
                          out_dir / f'confusion_matrix_{args.dataset_name}.png')

    print("\nDone.")


if __name__ == '__main__':
    main()
