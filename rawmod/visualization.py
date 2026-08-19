#!/usr/bin/env python3
"""
Reusable visualization and artifact-loading helpers for deep modification runs.

This module intentionally has no training dependency.  It can be imported by
deep_mod_model.py and loo.py, and it can also be run through
visualize_deep_mod.py to regenerate plots from saved result artifacts.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ModuleNotFoundError:
    def confusion_matrix(y_true, y_pred, labels=None):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        labels = [0, 1] if labels is None else list(labels)
        out = np.zeros((len(labels), len(labels)), dtype=np.int64)
        label_to_i = {label: i for i, label in enumerate(labels)}
        for true, pred in zip(y_true, y_pred):
            if true in label_to_i and pred in label_to_i:
                out[label_to_i[true], label_to_i[pred]] += 1
        return out

    def precision_score(y_true, y_pred, pos_label=1, zero_division=0):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
        fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
        denom = tp + fp
        return float(zero_division) if denom == 0 else float(tp / denom)

    def recall_score(y_true, y_pred, pos_label=1, zero_division=0):
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
        fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())
        denom = tp + fn
        return float(zero_division) if denom == 0 else float(tp / denom)

    def f1_score(y_true, y_pred, pos_label=1, zero_division=0):
        precision = precision_score(
            y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
        recall = recall_score(
            y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
        denom = precision + recall
        return float(zero_division) if denom == 0 else float(2 * precision * recall / denom)

    def precision_recall_curve(y_true, y_score):
        y_true = (np.asarray(y_true).astype(int) == 1).astype(np.int64)
        y_score = np.asarray(y_score).astype(float)
        if y_true.size == 0:
            return np.array([1.0]), np.array([0.0]), np.array([])

        order = np.argsort(y_score, kind='mergesort')[::-1]
        y_true = y_true[order]
        y_score = y_score[order]

        distinct = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        thresholds = y_score[threshold_idxs]

        precision = tps / np.maximum(tps + fps, 1)
        if tps[-1] == 0:
            recall = np.ones_like(tps, dtype=float)
        else:
            recall = tps / tps[-1]

        sl = slice(None, None, -1)
        return (
            np.r_[precision[sl], 1.0],
            np.r_[recall[sl], 0.0],
            thresholds[sl],
        )

    def average_precision_score(y_true, y_score):
        y_true = (np.asarray(y_true).astype(int) == 1).astype(np.int64)
        if int(y_true.sum()) == 0:
            return 0.0
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(-np.sum(np.diff(recall) * precision[:-1]))

    def roc_auc_score(y_true, y_score):
        y_true = (np.asarray(y_true).astype(int) == 1)
        y_score = np.asarray(y_score).astype(float)
        n_pos = int(y_true.sum())
        n_neg = int((~y_true).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError('ROC AUC is undefined with one class.')

        order = np.argsort(y_score, kind='mergesort')
        sorted_scores = y_score[order]
        ranks = np.empty_like(sorted_scores, dtype=float)
        i = 0
        while i < len(sorted_scores):
            j = i + 1
            while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            ranks[i:j] = (i + 1 + j) / 2.0
            i = j
        original_ranks = np.empty_like(ranks)
        original_ranks[order] = ranks
        pos_rank_sum = float(original_ranks[y_true].sum())
        return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def classification_report(y_true, y_pred, labels=None, target_names=None,
                              zero_division=0):
        labels = [0, 1] if labels is None else list(labels)
        target_names = target_names or [str(label) for label in labels]
        rows = ['              precision    recall  f1-score   support']
        for label, name in zip(labels, target_names):
            support = int((np.asarray(y_true).astype(int) == label).sum())
            p = precision_score(y_true, y_pred, pos_label=label,
                                zero_division=zero_division)
            r = recall_score(y_true, y_pred, pos_label=label,
                             zero_division=zero_division)
            f = f1_score(y_true, y_pred, pos_label=label,
                         zero_division=zero_division)
            rows.append(f'{name:>12s} {p:>10.2f} {r:>9.2f} {f:>9.2f} {support:>9d}')
        return '\n'.join(rows)

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    plt = None
    _MPL_IMPORT_ERROR = exc


CHANNEL_NAMES = [
    'raw_signal',
    'dwell_log1p',
    'is_A',
    'is_C',
    'is_G',
    'is_T',
    'strand',
    'mapq_norm',
    'matches_ref',
]


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to generate figures. Activate the same "
            "Python environment used for training, or install matplotlib in "
            "the current environment."
        ) from _MPL_IMPORT_ERROR


def _scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def _float_or_nan(value) -> float:
    if value in ('', 'n/a', None):
        return float('nan')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _round_metric(value):
    value = _float_or_nan(value)
    return round(value, 4) if not np.isnan(value) else float('nan')


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the threshold that maximizes positive-class F1."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or int(y_true.sum()) == 0:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1s = (2 * precisions[:-1] * recalls[:-1]
           / (precisions[:-1] + recalls[:-1] + 1e-8))
    return float(thresholds[int(np.argmax(f1s))])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Compute precision, recall, F1, AUPRC, and AUROC.

    If there are no modified positives, metrics are reported for correctly
    predicting the unmodified class, matching loo.py's historical behavior.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'auprc': float('nan'),
            'auroc': float('nan'),
            'threshold': 0.5,
            'all_unmod': False,
        }

    if int(y_true.sum()) == 0:
        thresh = 0.5 if threshold is None else float(threshold)
        y_pred = (y_prob >= thresh).astype(int)
        return {
            'precision': float(precision_score(
                y_true, y_pred, pos_label=0, zero_division=0)),
            'recall': float(recall_score(
                y_true, y_pred, pos_label=0, zero_division=0)),
            'f1': float(f1_score(
                y_true, y_pred, pos_label=0, zero_division=0)),
            'auprc': float(np.mean(1.0 - y_prob)),
            'auroc': float('nan'),
            'threshold': thresh,
            'all_unmod': True,
        }

    thresh = optimal_threshold(y_true, y_prob) if threshold is None else float(threshold)
    y_pred = (y_prob >= thresh).astype(int)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auprc': float(average_precision_score(y_true, y_prob)),
        'auroc': (float(roc_auc_score(y_true, y_prob))
                  if len(np.unique(y_true)) == 2 else float('nan')),
        'threshold': thresh,
        'all_unmod': False,
    }


def label_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: int,
) -> dict:
    """Precision/recall/F1 for a specific class label."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    label = int(label)
    return {
        'precision': float(precision_score(
            y_true, y_pred, pos_label=label, zero_division=0)),
        'recall': float(recall_score(
            y_true, y_pred, pos_label=label, zero_division=0)),
        'f1': float(f1_score(
            y_true, y_pred, pos_label=label, zero_division=0)),
        'support': int((y_true == label).sum()),
    }


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Average the modified-class and unmodified-class F1 scores.

    This is a compact single-model summary that combines both label-specific
    views while giving label=0 and label=1 equal influence.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return float('nan')

    # Always include both classes.  If a class is absent, its F1 is 0 via
    # zero_division=0, which keeps all-unmodified/all-modified test sets from
    # receiving an artificially perfect one-class macro score.
    return float(np.mean([
        f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    ]))


def zeror_metrics(train_labels: np.ndarray, test_labels: np.ndarray) -> dict:
    """ZeroR baseline: always predict the majority class in train_labels."""
    train_labels = np.asarray(train_labels).astype(int)
    test_labels = np.asarray(test_labels).astype(int)
    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    majority = 1 if n_pos >= n_neg else 0
    y_pred = np.full(len(test_labels), majority, dtype=int)
    return {
        'precision': float(precision_score(
            test_labels, y_pred, pos_label=majority, zero_division=0)),
        'recall': float(recall_score(
            test_labels, y_pred, pos_label=majority, zero_division=0)),
        'f1': float(f1_score(
            test_labels, y_pred, pos_label=majority, zero_division=0)),
        'auprc': float('nan'),
        'majority_class': majority,
        'all_unmod': bool(int(test_labels.sum()) == 0),
        'test_labels': test_labels,
    }


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    unit_name: str = 'examples',
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    auroc = (roc_auc_score(y_true, y_prob)
             if len(np.unique(y_true)) == 2 else float('nan'))
    auprc = (average_precision_score(y_true, y_prob)
             if int(y_true.sum()) > 0 else float(np.mean(1.0 - y_prob)))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n{'-' * 55}")
    print(f"  Unit   : {unit_name}  "
          f"N={len(y_true):,}  modified={int(y_true.sum()):,}  "
          f"unmodified={len(y_true) - int(y_true.sum()):,}")
    print(f"  AUROC  : {auroc:.4f}")
    print(f"  AUPRC  : {auprc:.4f}  (primary metric)")
    print("\n  Confusion matrix (rows=true, cols=pred):")
    print("               pred_unmod  pred_mod")
    print(f"  true_unmod   {cm[0, 0]:>9}  {cm[0, 1]:>8}")
    print(f"  true_mod     {cm[1, 0]:>9}  {cm[1, 1]:>8}")
    print()
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=['unmod', 'modified'],
        zero_division=0,
    ))
    print(f"{'-' * 55}\n")
    return auroc, auprc


def plot_training_curves(train_losses, val_losses, val_auprcs, best_epoch, out_path):
    _require_matplotlib()
    epochs = list(range(1, len(train_losses) + 1))
    if not epochs:
        print(f"  Skipping training curves; no history available for {out_path}")
        return

    best_epoch = int(best_epoch) if best_epoch else int(np.argmax(val_auprcs) + 1)
    best_epoch = max(1, min(best_epoch, len(epochs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='Train loss', color='steelblue')
    ax1.plot(epochs, val_losses, label='Validation loss', color='darkorange')
    ax1.axvline(best_epoch, color='red', linestyle='--', linewidth=1.0,
                label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss - PileupInceptionV3 (scaled)')
    ax1.legend(fontsize=8)
    if len(epochs) > 1:
        ax1.set_xlim(1, max(epochs))

    best_auprc = float(val_auprcs[best_epoch - 1])
    ax2.plot(epochs, val_auprcs, label='Val AUPRC', color='seagreen')
    ax2.axvline(best_epoch, color='red', linestyle='--', linewidth=1.0,
                label=f'Best epoch ({best_epoch})  AUPRC={best_auprc:.4f}')
    ax2.scatter([best_epoch], [best_auprc], color='red', zorder=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUPRC')
    ax2.set_title('Validation AUPRC - PileupInceptionV3 (scaled)')
    if len(epochs) > 1:
        ax2.set_xlim(1, max(epochs))
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle('PileupInceptionV3 (scaled) - Training Curves')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Training curves -> {out_path}")


def plot_precision_recall(y_true, y_prob, auprc, out_path):
    _require_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    all_unmod = int(y_true.sum()) == 0
    if all_unmod:
        plot_true = 1 - y_true
        plot_prob = 1.0 - y_prob
        plot_auprc = average_precision_score(plot_true, plot_prob)
        curve_label = f'Unmodified AUPRC = {plot_auprc:.4f}'
        best_thresh = 0.5
        y_pred_mod = (y_prob >= best_thresh).astype(int)
        best_prec = precision_score(y_true, y_pred_mod,
                                    pos_label=0, zero_division=0)
        best_rec = recall_score(y_true, y_pred_mod,
                                pos_label=0, zero_division=0)
        best_f1 = f1_score(y_true, y_pred_mod,
                           pos_label=0, zero_division=0)
        sweep_t = np.linspace(0, 1, 300)
        sweep_f1 = [
            f1_score(y_true, (y_prob >= t).astype(int),
                     pos_label=0, zero_division=0)
            for t in sweep_t
        ]
        precisions, recalls, _ = precision_recall_curve(plot_true, plot_prob)
    else:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1s = (2 * precisions[:-1] * recalls[:-1]
               / (precisions[:-1] + recalls[:-1] + 1e-8))
        best_idx = int(np.argmax(f1s))
        best_thresh = float(thresholds[best_idx])
        best_prec = float(precisions[best_idx])
        best_rec = float(recalls[best_idx])
        best_f1 = float(f1s[best_idx])
        sweep_t = np.linspace(0, 1, 300)
        sweep_f1 = [
            f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
            for t in sweep_t
        ]
        curve_label = f'AUPRC = {auprc:.4f}'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={'width_ratios': [2, 1]})
    ax = axes[0]
    ax.plot(recalls, precisions, label=curve_label, color='steelblue')
    ax.scatter([best_rec], [best_prec], color='red', zorder=5,
               label=f'thresh={best_thresh:.3f}  '
                     f'prec={best_prec:.3f}  rec={best_rec:.3f}  F1={best_f1:.3f}')
    ax.axvline(best_rec, color='red', linewidth=0.8, linestyle=':')
    ax.axhline(best_prec, color='red', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(sweep_t, sweep_f1, color='seagreen')
    ax2.axvline(best_thresh, color='red', linestyle=':',
                label=f'optimal threshold = {best_thresh:.3f}')
    ax2.axhline(best_f1, color='red', linestyle=':',
                label=f'optimal F1 = {best_f1:.3f}')
    ax2.scatter([best_thresh], [best_f1], color='red', zorder=5)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 vs. Threshold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle('PileupInceptionV3 (scaled) - Precision-Recall / Threshold Sweep')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"  PR curve -> {out_path}")
    print(f"  Optimal threshold (max F1): {best_thresh:.4f}")
    print(f"    Precision : {best_prec:.4f}")
    print(f"    Recall    : {best_rec:.4f}")
    print(f"    F1        : {best_f1:.4f}")
    return best_thresh


def plot_confusion_matrix(y_true, y_prob, threshold, out_path):
    _require_matplotlib()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm.astype(float) / np.maximum(row_sums, 1)
    labels = ['Unmodified', 'Modified']

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                   vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion of true class', fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix (threshold = {threshold:.4f})')

    for i in range(2):
        for j in range(2):
            col = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i - 0.1, f'{cm[i, j]:,}',
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', color=col)
            ax.text(j, i + 0.18, f'({cm_norm[i, j] * 100:.1f}%)',
                    ha='center', va='center', fontsize=9, color=col)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix -> {out_path}")


def _prediction_group(
    label: str,
    y_true,
    y_prob=None,
    threshold: float | None = None,
    y_pred=None,
    held_out: str = '',
    model: str = '',
    notes: str = '',
) -> dict:
    group = {
        'label': label,
        'held_out': held_out,
        'model': model,
        'notes': notes,
    }
    if y_true is not None:
        group['y_true'] = np.asarray(y_true).astype(int)
    if y_prob is not None:
        group['y_prob'] = np.asarray(y_prob).astype(float)
    if threshold is not None:
        group['threshold'] = float(threshold)
    if y_pred is not None:
        group['y_pred'] = np.asarray(y_pred).astype(int)
    return group


def _group_y_pred(group: dict):
    if 'y_pred' in group:
        return group['y_pred']
    if 'y_prob' not in group:
        return None
    threshold = float(group.get('threshold', 0.5))
    return (group['y_prob'] >= threshold).astype(int)


def _legacy_metric_group(label: str, result: dict, held_out: str, model: str) -> dict:
    legacy_label = 0 if result.get('all_unmod') else 1
    return {
        'label': label,
        'held_out': held_out,
        'model': model,
        'legacy_label': legacy_label,
        'legacy_metrics': {
            'precision': _float_or_nan(result.get('precision')),
            'recall': _float_or_nan(result.get('recall')),
            'f1': _float_or_nan(result.get('f1')),
        },
        'threshold': result.get('threshold', float('nan')),
        'n_test': result.get('n_test', ''),
        'notes': (
            f"legacy artifact only has metrics for label={legacy_label}; "
            "rerun LODO or regenerate lodo result npz files for all plots"
        ),
    }


def _label_metrics_for_group(group: dict, label: int) -> dict:
    if 'y_true' in group:
        y_pred = _group_y_pred(group)
        if y_pred is not None:
            return label_metrics_from_predictions(group['y_true'], y_pred, label)

    if group.get('legacy_label') == int(label):
        metrics = dict(group['legacy_metrics'])
        metrics['support'] = ''
        return metrics

    return {
        'precision': float('nan'),
        'recall': float('nan'),
        'f1': float('nan'),
        'support': '',
    }


def _macro_f1_for_group(group: dict) -> float:
    if 'y_true' not in group:
        return float('nan')
    y_pred = _group_y_pred(group)
    if y_pred is None:
        return float('nan')
    return macro_f1_score(group['y_true'], y_pred)


def _build_lodo_comparison_groups(
    loo_results: list[dict],
    main_metrics: dict,
    zeror_base: dict | None,
) -> list[dict]:
    groups: list[dict] = []

    groups.append(_prediction_group(
        label='Base\n(all datasets)',
        y_true=main_metrics.get('y_true'),
        y_prob=main_metrics.get('y_prob'),
        threshold=main_metrics.get('threshold', 0.5),
        held_out='all (stratified split)',
        model='Base',
    ) if 'y_true' in main_metrics and 'y_prob' in main_metrics else
        _legacy_metric_group(
            'Base\n(all datasets)',
            main_metrics,
            'all (stratified split)',
            'Base',
        ))

    for result in loo_results:
        held_out = str(result['held_out'])
        label = f"LODO\n(held: {held_out})"
        if 'test_labels' in result and 'y_prob' in result:
            threshold = optimal_threshold(result['test_labels'], result['y_prob'])
            groups.append(_prediction_group(
                label=label,
                y_true=result['test_labels'],
                y_prob=result['y_prob'],
                threshold=threshold,
                held_out=held_out,
                model='LODO',
            ))
        else:
            groups.append(_legacy_metric_group(label, result, held_out, 'LODO'))

    if zeror_base is not None and 'y_true' in main_metrics:
        majority = int(zeror_base['majority_class'])
        y_true = np.asarray(main_metrics['y_true']).astype(int)
        y_pred = np.full(len(y_true), majority, dtype=int)
        groups.append(_prediction_group(
            label=(f"ZeroR\n(always {majority}"
                   f"={'unmod' if majority == 0 else 'mod'})"),
            y_true=y_true,
            y_pred=y_pred,
            held_out='base test set',
            model='ZeroR',
            notes=f'majority={majority}',
        ))
    elif zeror_base is not None:
        groups.append({
            'label': (f"ZeroR\n(always {int(zeror_base['majority_class'])})"),
            'held_out': 'base test set',
            'model': 'ZeroR',
            'legacy_label': int(zeror_base['majority_class']),
            'legacy_metrics': {
                'precision': _float_or_nan(zeror_base.get('precision')),
                'recall': _float_or_nan(zeror_base.get('recall')),
                'f1': _float_or_nan(zeror_base.get('f1')),
            },
            'notes': 'legacy ZeroR metrics only',
        })

    return groups


def _comparison_plot_paths(out_path: str | Path) -> dict[str, Path]:
    out_path = Path(out_path)
    suffix = out_path.suffix or '.png'
    stem = out_path.stem
    return {
        'label1': out_path.with_name(f'{stem}_label1{suffix}'),
        'label0': out_path.with_name(f'{stem}_label0{suffix}'),
        'macro_f1': out_path.with_name(f'{stem}_macro_f1{suffix}'),
    }


def _annotate_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        if np.isnan(value):
            text = 'n/a'
            height = 0.02
        else:
            text = f'{value:.3f}'
            height = bar.get_height() + 0.012
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            text,
            ha='center',
            va='bottom',
            fontsize=7,
            rotation=90,
            color='dimgrey',
        )


def _plot_label_specific_lodo(
    groups: list[dict],
    label: int,
    out_path: str | Path,
) -> None:
    colors = {
        'Precision': '#2166ac',
        'Recall': '#b2182b',
        'F1': '#1b7837',
    }
    metrics = ['Precision', 'Recall', 'F1']
    keys = ['precision', 'recall', 'f1']
    x = np.arange(len(groups))
    width = 0.22
    offsets = [-width, 0.0, width]
    fig_width = max(10, 2.0 * len(groups))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for metric, key, offset in zip(metrics, keys, offsets):
        values = [_label_metrics_for_group(group, label)[key] for group in groups]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=colors[metric],
            edgecolor='white',
            linewidth=0.6,
        )
        _annotate_bars(ax, bars, values)

    ax.set_xticks(x)
    ax.set_xticklabels([group['label'] for group in groups], fontsize=9)
    ax.set_ylim(0, 1.28)
    ax.set_ylabel('Score', fontsize=11)
    class_name = 'Modified' if label == 1 else 'Unmodified'
    ax.set_title(
        f'Base Model vs. LODO Models vs. ZeroR - {class_name} Metrics '
        f'(label={label})\n'
        'Precision, recall, and F1 are all computed with the same positive '
        f'class label={label}.',
        fontsize=9,
    )
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)

    notes = sorted({group.get('notes', '') for group in groups if group.get('notes')})
    if notes:
        fig.text(0.5, 0.01, '  '.join(notes),
                 ha='center', fontsize=8, color='dimgrey')
        rect = [0, 0.04, 1, 1]
    else:
        rect = [0, 0.01, 1, 1]

    plt.tight_layout(rect=rect)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  LODO label={label} comparison plot -> {out_path}", flush=True)


def _plot_macro_f1_lodo(
    groups: list[dict],
    out_path: str | Path,
) -> None:
    x = np.arange(len(groups))
    values = [_macro_f1_for_group(group) for group in groups]
    fig_width = max(10, 2.0 * len(groups))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    bars = ax.bar(
        x,
        values,
        width=0.55,
        color='#5aae61',
        edgecolor='white',
        linewidth=0.6,
    )
    _annotate_bars(ax, bars, values)

    ax.set_xticks(x)
    ax.set_xticklabels([group['label'] for group in groups], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Macro F1', fontsize=11)
    ax.set_title(
        'Base Model vs. LODO Models vs. ZeroR - Macro F1\n'
        'Macro F1 is the mean of modified-class F1 and unmodified-class F1.',
        fontsize=9,
    )
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    notes = sorted({group.get('notes', '') for group in groups if group.get('notes')})
    if notes:
        fig.text(0.5, 0.01, '  '.join(notes),
                 ha='center', fontsize=8, color='dimgrey')
        rect = [0, 0.04, 1, 1]
    else:
        rect = [0, 0.01, 1, 1]

    plt.tight_layout(rect=rect)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  LODO macro-F1 plot -> {out_path}", flush=True)


def plot_loo_results(
    loo_results: list[dict],
    main_metrics: dict,
    zeror_base: dict | None,
    out_path: str,
) -> list[Path]:
    """
    Write three LODO comparison plots:
      1. metrics with respect to label=1 (modified)
      2. metrics with respect to label=0 (unmodified)
      3. macro F1, averaging label=1 F1 and label=0 F1
    """
    _require_matplotlib()
    groups = _build_lodo_comparison_groups(loo_results, main_metrics, zeror_base)
    paths = _comparison_plot_paths(out_path)
    _plot_label_specific_lodo(groups, label=1, out_path=paths['label1'])
    _plot_label_specific_lodo(groups, label=0, out_path=paths['label0'])
    _plot_macro_f1_lodo(groups, out_path=paths['macro_f1'])
    return list(paths.values())


def plot_channel_importance(
    importance: np.ndarray,
    baseline_auprc: float,
    out_path: str,
    channel_names: list[str] | None = None,
) -> None:
    """Horizontal bar chart of per-channel AUPRC drop under permutation."""
    _require_matplotlib()
    importance = np.asarray(importance).astype(float)
    if channel_names is None:
        channel_names = CHANNEL_NAMES[:len(importance)]

    order = np.argsort(importance)[::-1]
    sorted_imp = importance[order]
    sorted_lbl = [channel_names[i] for i in order]
    colors = ['#d73027' if v >= 0 else '#4575b4' for v in sorted_imp]

    fig, ax = plt.subplots(figsize=(8, max(4, len(importance) * 0.55)))
    y_pos = np.arange(len(sorted_imp))
    ax.barh(y_pos, sorted_imp, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

    span = max(float(np.max(np.abs(sorted_imp))) if len(sorted_imp) else 0.0, 0.001)
    offset = span * 0.02
    for i, v in enumerate(sorted_imp):
        ha = 'left' if v >= 0 else 'right'
        off = offset if v >= 0 else -offset
        ax.text(v + off, i, f'{v:+.4f}', va='center', ha=ha, fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_lbl, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('AUPRC drop (baseline - permuted)', fontsize=10)
    ax.set_title(
        f'Permutation Channel Importance\n'
        f'Baseline AUPRC = {baseline_auprc:.4f}  '
        f'(positive = channel is helpful; negative = channel hurts)',
        fontsize=10,
    )
    ax.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Channel importance plot -> {out_path}", flush=True)


def save_loo_metrics_tsv(
    loo_results: list[dict],
    main_metrics: dict,
    zeror_base: dict | None,
    out_path: str,
) -> None:
    """Write a combined TSV matching the split LODO comparison plots."""
    groups = _build_lodo_comparison_groups(loo_results, main_metrics, zeror_base)
    rows = []
    for group in groups:
        label1 = _label_metrics_for_group(group, 1)
        label0 = _label_metrics_for_group(group, 0)
        rows.append({
            'model': group.get('model', ''),
            'held_out': group.get('held_out', ''),
            'precision_label1': _round_metric(label1['precision']),
            'recall_label1': _round_metric(label1['recall']),
            'f1_label1': _round_metric(label1['f1']),
            'support_label1': label1.get('support', ''),
            'precision_label0': _round_metric(label0['precision']),
            'recall_label0': _round_metric(label0['recall']),
            'f1_label0': _round_metric(label0['f1']),
            'support_label0': label0.get('support', ''),
            'macro_f1': _round_metric(_macro_f1_for_group(group)),
            'threshold': _round_metric(group.get('threshold', float('nan'))),
            'n_test': (len(group['y_true']) if 'y_true' in group
                       else group.get('n_test', '')),
            'notes': group.get('notes', ''),
        })

    fieldnames = [
        'model', 'held_out',
        'precision_label1', 'recall_label1', 'f1_label1', 'support_label1',
        'precision_label0', 'recall_label0', 'f1_label0', 'support_label0',
        'macro_f1', 'threshold', 'n_test', 'notes',
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f"  LOO metrics TSV -> {out_path}", flush=True)


def save_training_history(
    out_path: str | Path,
    train_losses,
    val_losses,
    val_auprcs,
    best_epoch: int,
) -> None:
    """Persist training history so plots can be regenerated without retraining."""
    np.savez(
        out_path,
        train_losses=np.asarray(train_losses, dtype=np.float32),
        val_losses=np.asarray(val_losses, dtype=np.float32),
        val_auprcs=np.asarray(val_auprcs, dtype=np.float32),
        best_epoch=int(best_epoch),
    )
    print(f"  Training history -> {out_path}")


def load_main_predictions(pred_path: str | Path) -> dict:
    """Load base-model predictions from test_predictions.npz."""
    with np.load(pred_path, allow_pickle=True) as d:
        files = set(d.files)
        y_true_key = 'base_y_true' if 'base_y_true' in files else 'y_true'
        y_prob_key = 'base_y_prob' if 'base_y_prob' in files else 'y_prob'
        out = {
            'y_true': np.asarray(d[y_true_key]).astype(int),
            'y_prob': np.asarray(d[y_prob_key]).astype(float),
        }
        for key in [
            'image_y_true',
            'image_y_prob',
            'train_base_y_true',
            'train_base_y_prob',
            'h5_paths',
        ]:
            if key in files:
                out[key] = np.asarray(d[key])
    return out


def main_metrics_from_predictions(predictions: dict) -> dict:
    """Recompute base model metrics from saved predictions."""
    metrics = compute_binary_metrics(predictions['y_true'], predictions['y_prob'])
    metrics['y_true'] = predictions['y_true']
    metrics['y_prob'] = predictions['y_prob']
    metrics['n_test'] = int(len(predictions['y_true']))
    if 'train_base_y_true' in predictions:
        metrics['n_train'] = int(len(predictions['train_base_y_true']))
    if 'image_y_true' in predictions and 'image_y_prob' in predictions:
        image_metrics = compute_binary_metrics(
            predictions['image_y_true'],
            predictions['image_y_prob'],
        )
        metrics['image_auprc'] = image_metrics['auprc']
        metrics['image_auroc'] = image_metrics['auroc']
        metrics['n_test_images'] = int(len(predictions['image_y_true']))
    return metrics


def load_lodo_result(path: str | Path) -> dict:
    """Load one lodo_<name>_result.npz file."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as d:
        result = {
            'artifact_path': path,
            'held_out': str(_scalar(d['held_out'])),
            'precision': float(_scalar(d['precision'])),
            'recall': float(_scalar(d['recall'])),
            'f1': float(_scalar(d['f1'])),
            'auprc': float(_scalar(d['auprc'])),
            'auroc': float(_scalar(d['auroc'])),
            'threshold': float(_scalar(d['threshold'])),
            'n_train': int(_scalar(d['n_train'])),
            'n_test': int(_scalar(d['n_test'])),
            'all_unmod': bool(_scalar(d['all_unmod'])),
        }
        if 'test_labels' in d.files and len(d['test_labels']) > 0:
            result['test_labels'] = np.asarray(d['test_labels']).astype(int)
        if 'y_prob' in d.files and len(d['y_prob']) > 0:
            result['y_prob'] = np.asarray(d['y_prob']).astype(float)
        if 'train_labels' in d.files and len(d['train_labels']) > 0:
            result['train_labels'] = np.asarray(d['train_labels']).astype(int)
        return result


def discover_lodo_results(out_dir: str | Path) -> list[dict]:
    out_dir = Path(out_dir)
    results = [load_lodo_result(p) for p in sorted(out_dir.glob('lodo_*_result.npz'))]
    return results


def _resolve_h5_paths(raw_paths, out_dir: Path) -> list[Path]:
    paths = []
    for raw in raw_paths:
        raw = Path(str(raw))
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([
                out_dir / raw,
                out_dir.parent / raw,
                Path.cwd() / raw,
                raw,
            ])
        for candidate in candidates:
            if candidate.exists():
                paths.append(candidate.resolve())
                break
    return paths


def _write_lodo_result_artifact(result: dict) -> None:
    path = result.get('artifact_path')
    if path is None:
        return
    np.savez(
        path,
        held_out=str(result['held_out']),
        precision=float(result['precision']),
        recall=float(result['recall']),
        f1=float(result['f1']),
        auprc=float(result['auprc']),
        auroc=float(result['auroc']),
        threshold=float(result['threshold']),
        n_train=int(result['n_train']),
        n_test=int(result['n_test']),
        all_unmod=bool(result['all_unmod']),
        train_labels=np.asarray(result.get('train_labels', []), dtype=np.int64),
        test_labels=np.asarray(result.get('test_labels', []), dtype=np.int64),
        y_prob=np.asarray(result.get('y_prob', []), dtype=np.float32),
    )


def backfill_lodo_predictions(
    out_dir: str | Path,
    lodo_results: list[dict],
    h5_paths,
    batch_size: int = 32,
) -> list[dict]:
    """
    Fill missing LODO y_true/y_prob arrays from saved fold checkpoints.

    This performs inference only.  It does not retrain any model.
    """
    missing = [
        result for result in lodo_results
        if 'test_labels' not in result or 'y_prob' not in result
    ]
    if not missing:
        return lodo_results

    out_dir = Path(out_dir)
    print(f"  Backfilling LODO predictions for {len(missing)} fold(s) "
          f"from saved checkpoints; this is inference-only.", flush=True)
    resolved_h5 = _resolve_h5_paths(h5_paths, out_dir)
    by_stem = {path.stem: path for path in resolved_h5}
    if not by_stem:
        print("  Could not resolve HDF5 paths; LODO plots will use available "
              "legacy metrics only.", file=sys.stderr)
        return lodo_results

    try:
        import torch
        from torch.utils.data import DataLoader
        try:
            from . import model as model_mod
        except ImportError:  # support direct execution from this directory
            import model as model_mod
    except Exception as exc:
        print(f"  Could not backfill LODO predictions ({exc}); LODO plots will "
              "use available legacy metrics only.", file=sys.stderr)
        return lodo_results

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  LODO backfill device: {device}", flush=True)
    for result in missing:
        held_out = str(result['held_out'])
        h5_path = by_stem.get(held_out)
        ckpt_path = out_dir / f'lodo_{held_out}_best.pt'
        if h5_path is None or not ckpt_path.exists():
            print(f"  Missing HDF5/checkpoint for LODO '{held_out}'; keeping "
                  "legacy metrics only.", file=sys.stderr)
            continue

        print(f"  Backfilling LODO '{held_out}' from {h5_path.name} ...",
              flush=True)
        with model_mod.h5py.File(h5_path, 'r') as hf:
            test_labels_full = model_mod.binary_labels(hf['labels'][:])
            test_position_keys = model_mod.make_position_keys(
                hf['ref_names'][:], hf['ref_pos'][:])
            attrs = dict(hf.attrs)

        file_sizes = np.array([len(test_labels_full)], dtype=np.int64)
        test_idx = np.arange(len(test_labels_full), dtype=np.int64)
        test_ds = model_mod.PileupDataset(
            [str(h5_path)], test_idx, file_sizes, augment=False, seed=42)
        loader_kwargs = model_mod.make_loader_kwargs(
            batch_size=batch_size,
            num_workers=0,
            device=device,
            worker_init_fn=None,
        )
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

        in_ch = int(attrs.get('n_channels', 9))
        model = model_mod.PileupInceptionV3(in_channels=in_ch).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])

        y_true_img, y_prob_img = model_mod.run_inference(model, test_loader, device)
        y_true, y_prob, _ = model_mod.aggregate_by_position(
            y_true_img, y_prob_img, test_position_keys)
        result['test_labels'] = y_true.astype(np.int64)
        result['y_prob'] = y_prob.astype(np.float32)
        _write_lodo_result_artifact(result)
        print(f"  Backfilled LODO predictions -> {result['artifact_path']}")

    return lodo_results


def load_zeror_from_metrics_tsv(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('model') != 'ZeroR':
                continue
            match = re.search(
                r'majority=([01])',
                f"{row.get('threshold', '')} {row.get('notes', '')}",
            )
            majority = int(match.group(1)) if match else 0
            return {
                'precision': _float_or_nan(
                    row.get(f'precision_label{majority}', row.get('precision'))),
                'recall': _float_or_nan(
                    row.get(f'recall_label{majority}', row.get('recall'))),
                'f1': _float_or_nan(
                    row.get(f'f1_label{majority}', row.get('f1'))),
                'auprc': float('nan'),
                'majority_class': majority,
                'all_unmod': 'all-unmod' in row.get('notes', ''),
            }
    return None


def plot_training_histories(out_dir: Path, plots_dir: Path) -> list[Path]:
    generated = []
    for history_path in sorted(out_dir.glob('training_history*.npz')):
        with np.load(history_path, allow_pickle=True) as d:
            suffix = history_path.stem.replace('training_history', '')
            out_name = 'training_curves' + suffix + '.png'
            out_path = plots_dir / out_name
            plot_training_curves(
                d['train_losses'],
                d['val_losses'],
                d['val_auprcs'],
                int(_scalar(d['best_epoch'])),
                out_path,
            )
            generated.append(out_path)
    return generated


def visualize_result_dir(
    out_dir: str | Path,
    plots_dir: str | Path | None = None,
    include_training_curves: bool = True,
    backfill_lodo: bool = True,
    backfill_batch_size: int = 32,
) -> list[Path]:
    """Regenerate all available visualizations from a saved output directory."""
    out_dir = Path(out_dir)
    plots_dir = Path(plots_dir) if plots_dir is not None else out_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    pred_path = out_dir / 'test_predictions.npz'
    predictions = None
    main_metrics = None

    if pred_path.exists():
        predictions = load_main_predictions(pred_path)
        main_metrics = main_metrics_from_predictions(predictions)
        auprc = main_metrics['auprc']
        pr_path = plots_dir / 'precision_recall.png'
        threshold = plot_precision_recall(
            predictions['y_true'], predictions['y_prob'], auprc, pr_path)
        generated.append(pr_path)

        cm_path = plots_dir / 'confusion_matrix.png'
        plot_confusion_matrix(
            predictions['y_true'], predictions['y_prob'], threshold, cm_path)
        generated.append(cm_path)
    else:
        print(f"  No test_predictions.npz found in {out_dir}", file=sys.stderr)

    if include_training_curves:
        generated.extend(plot_training_histories(out_dir, plots_dir))

    ci_path = out_dir / 'channel_importance.npz'
    if ci_path.exists():
        with np.load(ci_path, allow_pickle=True) as d:
            importance = np.asarray(d['importance']).astype(float)
            if 'baseline_auprc' in d.files:
                baseline_auprc = float(_scalar(d['baseline_auprc']))
            elif main_metrics is not None:
                baseline_auprc = float(main_metrics.get(
                    'image_auprc', main_metrics.get('auprc', float('nan'))))
            else:
                baseline_auprc = float('nan')
            if 'channel_names' in d.files:
                channel_names = [str(x) for x in d['channel_names'].tolist()]
            else:
                channel_names = CHANNEL_NAMES[:len(importance)]
        ch_path = plots_dir / 'channel_importance.png'
        plot_channel_importance(importance, baseline_auprc, ch_path, channel_names)
        generated.append(ch_path)

    lodo_results = discover_lodo_results(out_dir)
    if lodo_results and main_metrics is not None:
        if backfill_lodo and predictions is not None and 'h5_paths' in predictions:
            lodo_results = backfill_lodo_predictions(
                out_dir, lodo_results, predictions['h5_paths'],
                batch_size=backfill_batch_size)
        elif not backfill_lodo:
            print("  Skipping LODO prediction backfill; plots will use any "
                  "metrics already stored in lodo_*_result.npz.", flush=True)

        if predictions is not None and 'train_base_y_true' in predictions:
            zeror_base = zeror_metrics(
                predictions['train_base_y_true'],
                predictions['y_true'],
            )
        else:
            zeror_base = load_zeror_from_metrics_tsv(out_dir / 'lodo_metrics.tsv')
            if zeror_base is None:
                print("  ZeroR unavailable: no train split labels in predictions "
                      "and no existing lodo_metrics.tsv ZeroR row.",
                      file=sys.stderr)

        lodo_png = plots_dir / 'lodo_comparison.png'
        generated.extend(plot_loo_results(
            lodo_results, main_metrics, zeror_base, lodo_png))

        lodo_tsv = plots_dir / 'lodo_metrics.tsv'
        save_loo_metrics_tsv(lodo_results, main_metrics, zeror_base, lodo_tsv)
        generated.append(lodo_tsv)

    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Regenerate deep_mod_model visualizations without training.')
    parser.add_argument('--out-dir', required=True,
                        help='Directory containing test_predictions.npz and LODO artifacts.')
    parser.add_argument('--plots-dir', default=None,
                        help='Optional output directory for regenerated plots. '
                             'Defaults to --out-dir.')
    parser.add_argument('--no-training-curves', action='store_true',
                        help='Skip training_history*.npz curve regeneration.')
    parser.add_argument('--skip-lodo-backfill', action='store_true',
                        help='Do not run inference to fill old LODO result '
                             'files that lack test_labels/y_prob arrays.')
    parser.add_argument('--backfill-batch', type=int, default=32,
                        help='Batch size for inference-only LODO backfill '
                             '(default: 32).')
    args = parser.parse_args(argv)

    try:
        generated = visualize_result_dir(
            out_dir=args.out_dir,
            plots_dir=args.plots_dir,
            include_training_curves=not args.no_training_curves,
            backfill_lodo=not args.skip_lodo_backfill,
            backfill_batch_size=args.backfill_batch,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("\nGenerated files:")
    for path in generated:
        print(f"  {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
