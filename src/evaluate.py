"""
evaluate.py
===========
Evaluation utilities for the Transformer modification classifier.

  evaluate                     — print AUROC, AUPRC, confusion matrix, report
  plot_precision_recall        — PR curve + F1-vs-threshold figure
  plot_training_curves         — loss and AUPRC over epochs
  plot_confusion_matrix        — row-normalised heatmap with raw counts
  plot_feature_importance      — horizontal bar chart of permutation importances
  compute_permutation_importance — ΔAUPRC for each input feature
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)

from config import (
    THRESHOLD, SEED, BATCH_SIZE, DEVICE,
    PR_FIG_OUT, TRAIN_FIG_OUT,
)
from dataset import ContigTileDataset, collate_fn


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> None:
    """Print AUROC, AUPRC, confusion matrix, and full classification report."""
    y_pred = (y_prob >= threshold).astype(int)
    auroc  = roc_auc_score(y_true, y_prob)
    auprc  = average_precision_score(y_true, y_prob)
    cm     = confusion_matrix(y_true, y_pred)

    print(f"\n{'─'*50}")
    print(f"  AUROC  : {auroc:.4f}")
    print(f"  AUPRC  : {auprc:.4f}  (primary metric under class imbalance)")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"               pred_unmod  pred_mod")
    print(f"  true_unmod   {cm[0,0]:>9}  {cm[0,1]:>8}")
    print(f"  true_mod     {cm[1,0]:>9}  {cm[1,1]:>8}")
    print()
    print(classification_report(y_true, y_pred, target_names=["unmod", "modified"]))
    print(f"{'─'*50}\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_precision_recall(
    y_true:   np.ndarray,
    y_prob:   np.ndarray,
    auprc:    float,
    out_path: str = PR_FIG_OUT,
) -> float:
    """
    Two-panel figure:
      Left  — precision-recall curve with the optimal F1 point highlighted
      Right — F1 score swept over [0, 1] thresholds

    Returns the threshold that maximises F1.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-8)
    )
    best_idx    = int(np.argmax(f1_scores))
    best_thresh = float(thresholds[best_idx])
    best_prec   = float(precisions[best_idx])
    best_rec    = float(recalls[best_idx])
    best_f1     = float(f1_scores[best_idx])

    sweep_t  = np.linspace(0.0, 1.0, 300)
    sweep_f1 = np.array([
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in sweep_t
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(recalls, precisions, label=f"AUPRC = {auprc:.4f}", color="steelblue")
    ax.scatter([best_rec], [best_prec], color="red", zorder=5,
               label=f"thresh={best_thresh:.3f}  prec={best_prec:.3f}  "
                     f"rec={best_rec:.3f}  F1={best_f1:.3f}")
    ax.axvline(best_rec,  color="red", linewidth=0.8, linestyle=":")
    ax.axhline(best_prec, color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Recall");  ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Transformer)")
    ax.set_xlim(0, 1.05);    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(sweep_t, sweep_f1, color="seagreen")
    ax2.axvline(best_thresh, color="red", linestyle=":",
                label=f"optimal threshold = {best_thresh:.3f}")
    ax2.axhline(best_f1, color="red", linestyle=":", label=f"optimal F1 = {best_f1:.3f}")
    ax2.scatter([best_thresh], [best_f1], color="red", zorder=5)
    ax2.set_xlabel("Threshold"); ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 vs. Threshold")
    ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle("Transformer — Precision-Recall / Threshold Sweep")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  PR curve saved → {out_path}")
    print(f"  Optimal threshold (max F1): {best_thresh:.4f}")
    print(f"    Precision : {best_prec:.4f}")
    print(f"    Recall    : {best_rec:.4f}")
    print(f"    F1        : {best_f1:.4f}")
    return best_thresh


def plot_training_curves(
    train_losses: list[float],
    val_losses:   list[float],
    val_auprcs:   list[float],
    best_epoch:   int,
    out_path:     str = TRAIN_FIG_OUT,
) -> None:
    """
    Two-panel figure:
      Left  — training and validation BCE loss
      Right — validation AUPRC over epochs

    A dashed vertical line marks the epoch restored by early stopping.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train loss",      color="steelblue")
    ax1.plot(epochs, val_losses,   label="Validation loss", color="darkorange")
    ax1.axvline(best_epoch, color="red", linestyle="--", linewidth=1.0,
                label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend(fontsize=8); ax1.set_xlim(1, max(epochs))

    ax2.plot(epochs, val_auprcs, label="Val AUPRC", color="seagreen")
    ax2.axvline(best_epoch, color="red", linestyle="--", linewidth=1.0,
                label=f"Best epoch ({best_epoch})  AUPRC={max(val_auprcs):.4f}")
    ax2.scatter([best_epoch], [val_auprcs[best_epoch - 1]], color="red", zorder=5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUPRC")
    ax2.set_title("Validation AUPRC over Epochs")
    ax2.set_xlim(1, max(epochs)); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle("Transformer — Training Curves")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved → {out_path}")


def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float,
    out_path:  str = "transformer_confusion_matrix.png",
) -> None:
    """
    Saves a confusion matrix heatmap normalised by true-class totals (row-wise),
    so colour intensity reflects per-class recall rather than raw counts.
    Each cell shows both the raw count and the row percentage.
    """
    y_pred  = (y_prob >= threshold).astype(int)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = ["Unmodified", "Modified"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion of true class", fontsize=9)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.4f})")

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i - 0.1, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color=text_color)
            ax.text(j, i + 0.18, f"({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color=text_color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")


def plot_feature_importance(
    imp_df:         pd.DataFrame,
    baseline_auprc: float,
    top_n:          int = 20,
    out_path:       str = "transformer_feature_importance.png",
) -> None:
    """
    Horizontal bar chart of permutation feature importances (top_n features).
    Error bars show ±1 std across permutation repeats.
    """
    df  = imp_df.head(top_n).iloc[::-1]    # reverse so highest is at top
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))

    colors = ["#d73027" if v > 0 else "#4575b4" for v in df["importance"]]
    ax.barh(df["feature"], df["importance"], xerr=df["std"],
            color=colors, edgecolor="white", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔAUPRC (baseline − permuted)")
    ax.set_title(
        f"Transformer — Permutation Feature Importance  "
        f"(baseline AUPRC = {baseline_auprc:.4f})"
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance plot saved → {out_path}")


# ── Permutation feature importance ────────────────────────────────────────────

def _infer(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model in eval mode; return (y_true, y_prob) for real positions only.
    Used exclusively by compute_permutation_importance.
    """
    model.eval()
    all_true, all_prob = [], []
    with torch.no_grad():
        for x, y_batch, mask, _ in loader:
            x       = x.to(DEVICE)
            mask_d  = mask.to(DEVICE)
            logits  = model(x, mask=mask_d)           # (B, 1, W)
            probs   = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            y_np    = y_batch.squeeze(1).numpy()
            mask_np = mask.numpy()
            all_true.append(y_np[mask_np])
            all_prob.append(probs[mask_np])
    return np.concatenate(all_true), np.concatenate(all_prob)


def compute_permutation_importance(
    model:       torch.nn.Module,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    grp_test:    np.ndarray,
    pos_test:    np.ndarray,
    rn_test:     np.ndarray,
    feat_cols:   list[str],
    num_workers: int = 1,
    n_repeats:   int = 5,
    seed:        int = SEED,
) -> tuple[pd.DataFrame, float]:
    """
    Permutation feature importance.

    For each feature column:
      1. Shuffle its values randomly across all test positions.
      2. Re-compute AUPRC with the shuffled feature.
      3. importance = baseline_AUPRC − mean(AUPRC after permutation)

    Returns (DataFrame[feature, importance, std], baseline_auprc).
    """
    rng = np.random.default_rng(seed)

    base_ds     = ContigTileDataset(X_test, y_test, grp_test, pos_test, rn_test)
    base_loader = DataLoader(base_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers,
                             pin_memory=True)
    y_true, y_prob_base = _infer(model, base_loader)
    baseline_auprc      = average_precision_score(y_true, y_prob_base)
    print(f"  Baseline AUPRC: {baseline_auprc:.4f}")

    rows = []
    for i, feat_name in enumerate(feat_cols):
        drops = []
        for _ in range(n_repeats):
            X_perm       = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])

            perm_ds     = ContigTileDataset(X_perm, y_test, grp_test, pos_test, rn_test)
            perm_loader = DataLoader(perm_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     collate_fn=collate_fn, num_workers=num_workers,
                                     pin_memory=True)
            _, y_prob_perm = _infer(model, perm_loader)
            drops.append(baseline_auprc - average_precision_score(y_true, y_prob_perm))

        rows.append({
            "feature":    feat_name,
            "importance": float(np.mean(drops)),
            "std":        float(np.std(drops)),
        })
        print(f"  [{i+1:2d}/{len(feat_cols)}] {feat_name:<22}  "
              f"ΔAUPRC = {rows[-1]['importance']:+.4f} ± {rows[-1]['std']:.4f}")

    imp_df = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp_df, baseline_auprc