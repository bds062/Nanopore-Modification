#!/usr/bin/env python3
"""
train.py — central training entry point for DeepMod.

Reads manifest.tsv to resolve which datasets are for training (emseq /
canonical-control GT) and which are reserved test-only (motif-based GT).
Then runs the InceptionV3 pipeline with one of two split strategies:

  random   All train-role datasets merged; random position-level 70/15/15
            train/val/test split.  The model is evaluated on the held-out
            positions AND on every test-role dataset (external evaluation).

  lodo     Leave-one-dataset-out: for each train-role genome the model is
            retrained on all remaining genomes and evaluated on the held-out
            genome (this is Stage 4 of model.py).  External test-role datasets
            are also evaluated using the full-data best_model.pt.

Usage
-----
  python -m deepmod.train \\
      --manifest  manifest.tsv \\
      --out-dir   results/run1  \\
      [--split-mode random|lodo] \\
      [--epochs N] [--batch N] [--lr F] [--seed N] ...

  # resume a crashed run (default behaviour):
  python -m deepmod.train --manifest manifest.tsv --out-dir results/run1 ...

  # force full rerun:
  python -m deepmod.train --manifest manifest.tsv --out-dir results/run1 --no-resume ...
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.utils.data

try:
    from . import model as model_mod
    from .visualization import plot_loo_results, save_loo_metrics_tsv
except ImportError:
    import model as model_mod
    from visualization import plot_loo_results, save_loo_metrics_tsv


# ── manifest ──────────────────────────────────────────────────────────────────

def load_manifest(path: str) -> tuple[list, list]:
    """
    Parse manifest.tsv and return (train_rows, test_rows).

    Lines beginning with '#' are skipped.  Rows whose features_h5 path does
    not exist on disk are skipped with a warning (so the script runs normally
    while some datasets are still being featurized).
    """
    train_rows: list[dict] = []
    test_rows:  list[dict] = []

    with open(path, newline='') as fh:
        reader = csv.DictReader(
            (line for line in fh if not line.startswith('#')),
            delimiter='\t',
        )
        for row in reader:
            h5 = row.get('features_h5', '').strip()
            if not h5:
                continue
            if not Path(h5).exists():
                print(f"  [manifest skip] {row['name']}: features.h5 missing — "
                      f"{h5}", file=sys.stderr)
                continue
            role = row.get('role', '').strip()
            if role == 'train':
                train_rows.append(row)
            elif role == 'test':
                test_rows.append(row)

    return train_rows, test_rows


# ── external test evaluation ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate_external(
    ckpt_path: Path,
    test_rows: list,
    in_ch: int,
    out_dir: Path,
    device: torch.device,
) -> list[dict]:
    """
    Load best_model.pt and run inference on each test-role dataset.

    Returns a list of result dicts:
      name, organism, modification, n_mod, n_unmod, auprc, auroc
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if not test_rows:
        return []
    if not ckpt_path.exists():
        print(f"  [skip external eval] checkpoint not found: {ckpt_path}",
              file=sys.stderr)
        return []

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = model_mod.PileupInceptionV3(in_channels=in_ch).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    results = []
    for row in test_rows:
        h5   = row['features_h5']
        name = row['name']

        with h5py.File(h5, 'r') as hf:
            labels     = model_mod.binary_labels(hf['labels'][:])
            ref_names  = hf['ref_names'][:]
            ref_pos    = hf['ref_pos'][:]
            n          = len(labels)
            file_sizes = np.array([n], dtype=np.int64)

        ds = model_mod.PileupDataset(
            [h5], np.arange(n, dtype=np.int64), file_sizes, augment=False)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=64, shuffle=False, num_workers=0,
            pin_memory=(device.type == 'cuda'))

        y_true_img, y_prob_img = model_mod.run_inference(model, loader, device)
        position_keys = model_mod.make_position_keys(ref_names, ref_pos)
        y_true, y_prob, _ = model_mod.aggregate_by_position(
            y_true_img, y_prob_img, position_keys)

        n_mod   = int(y_true.sum())
        n_unmod = len(y_true) - n_mod

        if n_mod > 0 and len(np.unique(y_true)) == 2:
            auprc = float(average_precision_score(y_true, y_prob))
            auroc = float(roc_auc_score(y_true, y_prob))
        else:
            auprc = float(np.mean(1.0 - y_prob))
            auroc = float('nan')

        print(f"  [{name}]  N={len(y_true):,}  mod={n_mod:,}  "
              f"AUPRC={auprc:.4f}  AUROC={auroc:.4f}",
              file=sys.stderr)
        results.append({
            'name':         name,
            'organism':     row.get('organism', ''),
            'modification': row.get('modification', ''),
            'n_mod':        n_mod,
            'n_unmod':      n_unmod,
            'auprc':        auprc,
            'auroc':        auroc,
        })

    # ── write TSV (written again at the end via _write_external_tsv) ───────────
    print(f"  Collected {len(results)} external results.", file=sys.stderr)

    # ── bar chart ─────────────────────────────────────────────────────────────
    _plot_external_comparison(results, out_dir / 'external_test_comparison.png')
    return results


def _plot_external_comparison(results: list[dict], out_path: Path) -> None:
    if not results:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names   = [r['name'] for r in results]
    auprcs  = [r['auprc'] for r in results]
    aurocs  = [r['auroc'] for r in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.6), 4))
    ax.bar(x - w / 2, auprcs, w, label='AUPRC', color='#2196F3', alpha=0.85)
    bar_colors = ['#FF9800' if not np.isnan(v) else '#cccccc' for v in aurocs]
    ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in aurocs],
           w, label='AUROC', color=bar_colors, alpha=0.85)

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('External test set performance (motif-based GT, not seen during training)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  External test comparison → {out_path}", file=sys.stderr)


# ── TSV helper ────────────────────────────────────────────────────────────────

def _write_external_tsv(results: list[dict], tsv_path: Path) -> None:
    with open(tsv_path, 'w') as fh:
        fh.write('name\torganism\tmodification\tn_mod\tn_unmod\tauprc\tauroc\n')
        for r in results:
            auroc_str = 'nan' if np.isnan(r['auroc']) else f"{r['auroc']:.4f}"
            fh.write(f"{r['name']}\t{r['organism']}\t{r['modification']}\t"
                     f"{r['n_mod']}\t{r['n_unmod']}\t{r['auprc']:.4f}\t{auroc_str}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train DeepMod from a dataset manifest (manifest.tsv).')

    # ── manifest / mode ───────────────────────────────────────────────────────
    parser.add_argument('--manifest', required=True,
                        help='Path to manifest.tsv')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory (checkpoints, plots, metrics)')
    parser.add_argument('--split-mode', choices=['random', 'lodo'],
                        default='random',
                        help=(
                            '"random": random position-level train/val/test '
                            'split across all training genomes. '
                            '"lodo": leave-one-genome-out retraining — one '
                            'fold per training dataset. '
                            '(default: random)'
                        ))
    parser.add_argument('--skip-external-eval', action='store_true',
                        help='Skip evaluation on test-role datasets after training.')

    # ── training hyperparameters (forwarded to model.py) ─────────────────────
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch',        type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--val-frac',     type=float, default=0.15)
    parser.add_argument('--test-frac',    type=float, default=0.15)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--dropout',      type=float, default=0.4)
    parser.add_argument('--focal',        action='store_true',
                        help='Use focal loss instead of BCE + pos_weight.')
    parser.add_argument('--focal-gamma',  type=float, default=2.0)
    parser.add_argument('--mixup-alpha',  type=float, default=0.2)
    parser.add_argument('--balanced-sampler', action='store_true')
    parser.add_argument('--epoch-samples', type=int,  default=0)
    parser.add_argument('--rc-augment',   action='store_true')
    parser.add_argument('--signal-noise-std', type=float, default=0.05)
    parser.add_argument('--num-workers',  type=int,   default=0)
    parser.add_argument('--log-every',    type=int,   default=200)
    parser.add_argument('--skip-channel-importance', action='store_true')
    parser.add_argument('--channel-importance-repeats', type=int, default=5)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--no-resume',    action='store_true',
                        help='Ignore existing checkpoints and rerun from scratch.')

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load manifest ─────────────────────────────────────────────────────────
    print(f"Manifest: {args.manifest}", file=sys.stderr)
    train_rows, test_rows = load_manifest(args.manifest)

    if not train_rows:
        print(
            'ERROR: No train-role datasets with existing features.h5 found.\n'
            '  Check that featurization jobs have completed and paths in the '
            'manifest are correct.',
            file=sys.stderr,
        )
        sys.exit(1)

    train_h5s = [r['features_h5'] for r in train_rows]

    print(f"\nTraining datasets  ({len(train_rows)}):", file=sys.stderr)
    for r in train_rows:
        print(f"  {r['name']:28s}  {r['organism']:32s}  "
              f"mod={r['modification']:16s}  gt={r['gt_source']}",
              file=sys.stderr)

    if test_rows:
        print(f"\nExternal test datasets  ({len(test_rows)}):", file=sys.stderr)
        for r in test_rows:
            print(f"  {r['name']:28s}  {r['organism']:32s}  "
                  f"mod={r['modification']:16s}  gt={r['gt_source']}",
                  file=sys.stderr)

    # ── read tensor shape from first training file ────────────────────────────
    with h5py.File(train_h5s[0], 'r') as hf:
        in_ch = int(hf.attrs.get('n_channels', 9))

    # ── build model.py argument list and delegate ─────────────────────────────
    #
    # model.py's --split-mode (position|contig) controls how the within-file
    # train/val/test split is made.  We always use 'position' here.
    # model.py's built-in LODO stage (Stage 4) is the leave-one-dataset-out
    # evaluation; we enable it for 'lodo' mode and skip it for 'random' mode.

    model_argv = [
        '--input',         *train_h5s,
        '--out-dir',       str(out_dir),
        '--epochs',        str(args.epochs),
        '--batch',         str(args.batch),
        '--lr',            str(args.lr),
        '--weight-decay',  str(args.weight_decay),
        '--val-frac',      str(args.val_frac),
        '--test-frac',     str(args.test_frac),
        '--patience',      str(args.patience),
        '--dropout',       str(args.dropout),
        '--focal-gamma',   str(args.focal_gamma),
        '--mixup-alpha',   str(args.mixup_alpha),
        '--epoch-samples', str(args.epoch_samples),
        '--num-workers',   str(args.num_workers),
        '--log-every',     str(args.log_every),
        '--channel-importance-repeats', str(args.channel_importance_repeats),
        '--signal-noise-std', str(args.signal_noise_std),
        '--seed',          str(args.seed),
        '--split-mode',    'position',
    ]
    for flag, enabled in [
        ('--focal',                    args.focal),
        ('--rc-augment',               args.rc_augment),
        ('--balanced-sampler',         args.balanced_sampler),
        ('--skip-channel-importance',  args.skip_channel_importance),
        ('--no-resume',                args.no_resume),
    ]:
        if enabled:
            model_argv.append(flag)

    if args.split_mode == 'random':
        # Disable model.py's LODO stage; we'll do a separate external evaluation.
        model_argv.append('--skip-lodo')

    # lodo mode: model.py's built-in LODO is enabled (default); one full
    # retraining per training genome, tested on the held-out genome.

    print(f"\nSplit mode : {args.split_mode}", file=sys.stderr)
    if args.split_mode == 'lodo':
        print(
            f"  LODO folds : {len(train_h5s)} "
            f"({[Path(p).stem for p in train_h5s]})",
            file=sys.stderr,
        )
    print(f"Output dir : {out_dir}", file=sys.stderr)

    # ── call model.py's main() via sys.argv substitution ─────────────────────
    saved_argv = sys.argv
    sys.argv   = ['model.py'] + model_argv
    try:
        model_mod.main()
    finally:
        sys.argv = saved_argv

    # ── external test evaluation ──────────────────────────────────────────────
    if args.skip_external_eval or not test_rows:
        return

    print(f"\n{'═' * 60}", file=sys.stderr)
    print(f"  External test evaluation  ({len(test_rows)} datasets)",
          file=sys.stderr)
    print(f"{'═' * 60}", file=sys.stderr)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = out_dir / 'best_model.pt'

    ext_results = evaluate_external(
        ckpt_path=ckpt_path,
        test_rows=test_rows,
        in_ch=in_ch,
        out_dir=out_dir,
        device=device,
    )

    if ext_results:
        _write_external_tsv(ext_results, out_dir / 'external_test_results.tsv')
        print(f"\n  External test summary:", file=sys.stderr)
        print(f"  {'Dataset':28s}  {'AUPRC':>7}  {'AUROC':>7}  "
              f"{'mod':>7}  {'unmod':>8}", file=sys.stderr)
        for r in ext_results:
            auroc_s = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else '    N/A'
            print(f"  {r['name']:28s}  {r['auprc']:>7.4f}  {auroc_s:>7}  "
                  f"{r['n_mod']:>7,}  {r['n_unmod']:>8,}", file=sys.stderr)


if __name__ == '__main__':
    main()
