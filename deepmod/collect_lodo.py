#!/usr/bin/env python3
"""
Collect LODO fold results and regenerate summary plots/TSV.

Run this after all parallel LODO fold jobs complete.  It loads:
  - test_predictions.npz + best_model.pt  → main model metrics
  - lodo_STEM_result.npz for each input H5 → per-fold results
  - train_base_y_true from test_predictions.npz → ZeroR baseline

Writes:
  lodo_comparison*.png   — modified/unmodified/macro-F1 comparison plots
  lodo_metrics.tsv       — combined metrics table

Usage:
  python collect_lodo.py --out-dir RESULTS_DIR --input H5 [H5 ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from . import lodo as loo_mod
    from .model import (
        load_lodo_result,
        load_main_metrics,
        lodo_sentinel,
        lodo_stem,
    )
    from .visualization import plot_loo_results, save_loo_metrics_tsv
except ImportError:
    import lodo as loo_mod
    from model import (
        load_lodo_result,
        load_main_metrics,
        lodo_sentinel,
        lodo_stem,
    )
    from visualization import plot_loo_results, save_loo_metrics_tsv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True,
                        help='Training output directory containing best_model.pt, '
                             'test_predictions.npz, and lodo_*_result.npz files')
    parser.add_argument('--input', required=True, nargs='+', metavar='H5',
                        help='HDF5 files used for training (same as --input to model.py)')
    args = parser.parse_args()

    out_dir   = Path(args.out_dir)
    ckpt_path = out_dir / 'best_model.pt'
    pred_path = out_dir / 'test_predictions.npz'

    # ── Main model metrics ─────────────────────────────────────────────────────
    if not ckpt_path.exists() or not pred_path.exists():
        print(f"ERROR: {ckpt_path} and/or {pred_path} not found in {out_dir}.\n"
              f"Run main training first.", file=sys.stderr)
        sys.exit(1)

    main_metrics = load_main_metrics(pred_path, ckpt_path)
    if main_metrics is None:
        print("ERROR: could not load main metrics from checkpoint files.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Main model:  AUPRC={main_metrics['auprc']:.4f}  "
          f"AUROC={main_metrics['auroc']:.4f}  "
          f"F1={main_metrics['f1']:.4f}", flush=True)

    # ── ZeroR baseline ─────────────────────────────────────────────────────────
    d = np.load(pred_path, allow_pickle=True)
    train_labels = d['train_base_y_true'].astype(np.int64)
    test_labels  = d['base_y_true'].astype(np.int64)
    zeror_base   = loo_mod.zeror_metrics(train_labels, test_labels)
    print(f"ZeroR: majority={zeror_base['majority_class']}  "
          f"F1={zeror_base['f1']:.4f}", flush=True)

    # ── Per-fold results ───────────────────────────────────────────────────────
    loo_results = []
    missing     = []
    for h5_path in args.input:
        sentinel = lodo_sentinel(out_dir, h5_path)
        stem     = lodo_stem(h5_path)
        if sentinel.exists():
            result = load_lodo_result(out_dir, h5_path)
            loo_results.append(result)
            print(f"  [{stem}]  AUPRC={result['auprc']:.4f}  "
                  f"F1={result['f1']:.4f}", flush=True)
        else:
            missing.append(stem)
            print(f"  [{stem}]  MISSING sentinel: {sentinel}", flush=True)

    if missing:
        print(f"\nWARNING: {len(missing)} fold(s) not yet complete: {missing}",
              file=sys.stderr)
        print("Proceeding with available folds.", file=sys.stderr)

    if not loo_results:
        print("ERROR: no LODO fold results found. Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # ── Write plots and TSV ────────────────────────────────────────────────────
    print(f"\nWriting summary plots to {out_dir}/", flush=True)
    plot_paths = plot_loo_results(
        loo_results=loo_results,
        main_metrics=main_metrics,
        zeror_base=zeror_base,
        out_path=str(out_dir / 'lodo_comparison.png'),
    )
    for p in plot_paths:
        print(f"  {p}", flush=True)

    tsv_path = str(out_dir / 'lodo_metrics.tsv')
    save_loo_metrics_tsv(
        loo_results=loo_results,
        main_metrics=main_metrics,
        zeror_base=zeror_base,
        out_path=tsv_path,
    )
    print(f"  {tsv_path}", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
