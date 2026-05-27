"""
loo.py
======
Leave-one-dataset-out (LODO) evaluation module for the Transformer modification
classifier.  Imported and called by main.py — not a standalone script.

Public API
----------
  get_datasets      — returns ordered list of (name, tsv_path, is_modified) tuples
  run_lodo          — train on 3 datasets, evaluate on the held-out one
  optimal_threshold — find the threshold that maximises F1
  plot_loo_results  — single-panel bar chart: main model + 4 LOO models
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

import config as cfg
from dataset import ContigTileDataset, collate_fn
from evaluate import plot_training_curves
from features import build_feature_matrix
from model import ModTransformer, count_parameters

# ── Dataset registry ──────────────────────────────────────────────────────────
# Returned as a function so cfg.TSV_* are read at call time, after any CLI
# overrides have been applied (module-level lists capture values at import time).

def get_datasets() -> list[tuple[str, str, bool]]:
    """Return the ordered list of (name, tsv_path, is_modified) tuples."""
    return [
        ("unmod", cfg.TSV_UNMOD, False),
        ("5mC",   cfg.TSV_5MC,   True),
        ("5hmC",  cfg.TSV_5HMC,  True),
        ("6mA",   cfg.TSV_6MA,   True),
    ]

LOO_FIG_OUT     = "transformer_loo_results.png"
LOO_METRICS_OUT = "transformer_loo_metrics.tsv"


# ── Epoch helper (duplicated from main.py to avoid circular import) ───────────

def _run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    context  = torch.enable_grad() if training else torch.no_grad()

    total_loss, all_true, all_prob = 0.0, [], []
    with context:
        for x, y_batch, mask, _meta in loader:
            x       = x.to(cfg.DEVICE)
            y_batch = y_batch.to(cfg.DEVICE)
            mask    = mask.to(cfg.DEVICE)

            logits      = model(x, mask=mask)
            logits_flat = logits.squeeze(1)[mask]
            y_flat      = y_batch.squeeze(1)[mask]
            loss        = criterion(logits_flat, y_flat)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            probs_flat  = torch.sigmoid(logits_flat)
            total_loss += loss.item() * y_flat.numel()
            all_true.append(y_flat.detach().cpu().numpy())
            all_prob.append(probs_flat.detach().cpu().numpy())

    n = sum(a.size for a in all_true)
    return total_loss / max(n, 1), np.concatenate(all_true), np.concatenate(all_prob)


# ── Threshold selection ───────────────────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the decision threshold that maximises F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1s)])


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Compute precision, recall, F1, and AUPRC.

    If the test set has no positive labels (e.g. the unmod dataset), metrics
    are reported for the negative class (label=0) at threshold=0.5, and AUPRC
    is approximated as the mean model confidence of predicting unmodified.
    The returned dict carries an `all_unmod` flag so callers can annotate plots.
    """
    if y_true.sum() == 0:
        thresh = 0.5
        y_pred = (y_prob >= thresh).astype(int)
        return {
            "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
            "recall":    float(recall_score(   y_true, y_pred, pos_label=0, zero_division=0)),
            "f1":        float(f1_score(       y_true, y_pred, pos_label=0, zero_division=0)),
            "auprc":     float(np.mean(1 - y_prob)),
            "threshold": thresh,
            "all_unmod": True,
        }

    thresh = threshold if threshold is not None else optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(   y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(       y_true, y_pred, zero_division=0)),
        "auprc":     float(average_precision_score(y_true, y_prob)),
        "threshold": thresh,
        "all_unmod": False,
    }


# ── Single LODO run ───────────────────────────────────────────────────────────

def run_lodo(
    held_out_name: str,
    all_frames:    dict[str, pd.DataFrame],
    num_workers:   int,
) -> dict:
    """
    Train on every dataset except `held_out_name`; test on ALL contigs of
    `held_out_name` (no train/test split — the entire held-out dataset is the
    test set).

    Parameters
    ----------
    held_out_name : key into all_frames — the dataset to withhold
    all_frames    : dict of pre-loaded DataFrames, keyed by dataset name
    num_workers   : DataLoader worker count

    Returns
    -------
    dict with keys: held_out, precision, recall, f1, auprc, threshold,
                    n_train, n_test, all_unmod
    """
    print(f"\n{'═'*60}")
    print(f"  LODO fold: held-out = {held_out_name}")
    print(f"{'═'*60}")

    train_df = pd.concat(
        [df for name, df in all_frames.items() if name != held_out_name],
        ignore_index=True,
    )
    test_df = all_frames[held_out_name].copy()

    X_train_df = build_feature_matrix(train_df)
    X_test_df  = build_feature_matrix(test_df)

    X_train = X_train_df.values.astype(np.float32)
    X_test  = X_test_df.values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.float32)

    grp_train = train_df["ref_name"].values
    grp_test  = test_df["ref_name"].values
    pos_train = train_df["ref_pos"].values
    pos_test  = test_df["ref_pos"].values

    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train   = (X_train - feat_mean) / feat_std
    X_test    = (X_test  - feat_mean) / feat_std

    print(f"  Train: {len(X_train):,} positions ({len(np.unique(grp_train))} contigs)")
    print(f"  Test : {len(X_test):,} positions  ({len(np.unique(grp_test))} contigs)  "
          f"[mod={int(y_test.sum())}, unmod={int((y_test==0).sum())}]")

    train_ds = ContigTileDataset(X_train, y_train, grp_train, pos_train, grp_train,
                                 stride=cfg.WINDOW_STRIDE)
    test_ds  = ContigTileDataset(X_test,  y_test,  grp_test,  pos_test,  grp_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    in_ch = X_train.shape[1]
    model = ModTransformer(in_channels=in_ch).to(cfg.DEVICE)

    n_neg      = int((y_train == 0).sum())
    n_pos      = int((y_train == 1).sum())
    pos_weight = torch.tensor([math.sqrt(n_neg / max(n_pos, 1))], device=cfg.DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR / 20
    )

    print(f"  Trainable params: {count_parameters(model):,}  "
          f"|  pos_weight={pos_weight.item():.2f}")

    best_auprc, best_epoch, patience_count, best_state = -1.0, 1, 0, None
    train_losses, val_losses, val_auprcs = [], [], []

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        tr_loss, _, _    = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, vt, vp = _run_epoch(model, test_loader,  criterion, None)
        scheduler.step()

        val_auprc = average_precision_score(vt, vp) if vt.sum() > 0 else float(np.mean(1 - vp))
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_auprcs.append(val_auprc)

        print(f"    Epoch {epoch:3d}/{cfg.NUM_EPOCHS}  "
              f"train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}  "
              f"val_AUPRC={val_auprc:.4f}")

        if val_auprc > best_auprc:
            best_auprc     = val_auprc
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.PATIENCE:
                print(f"    Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    model.to(cfg.DEVICE)
    print(f"  Best AUPRC={best_auprc:.4f} at epoch {best_epoch}")

    plot_training_curves(
        train_losses, val_losses, val_auprcs, best_epoch,
        out_path=f"transformer_loo_training_curves_{held_out_name}.png",
    )

    _, y_true, y_prob = _run_epoch(model, test_loader, criterion, None)
    m = _compute_metrics(y_true, y_prob)

    note = " [all-unmod: metrics vs label=0]" if m["all_unmod"] else ""
    print(f"  Threshold={m['threshold']:.4f}  Prec={m['precision']:.4f}  "
          f"Rec={m['recall']:.4f}  F1={m['f1']:.4f}  AUPRC={m['auprc']:.4f}{note}")

    return {
        "held_out":  held_out_name,
        "precision": round(m["precision"], 4),
        "recall":    round(m["recall"],    4),
        "f1":        round(m["f1"],        4),
        "auprc":     round(m["auprc"],     4),
        "threshold": round(m["threshold"], 4),
        "n_train":   int(len(X_train)),
        "n_test":    int(len(X_test)),
        "all_unmod": m["all_unmod"],
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_loo_results(
    loo_metrics:   list[dict],
    main_overall:  dict,
    out_path:      str = LOO_FIG_OUT,
) -> None:
    """
    Single-panel figure with 5 groups of 3 bars each.

    Groups
    ------
      Group 0  — Main model (trained on all 4 datasets, tested on held-out
                 contigs from the 80/20 contig-level split, all datasets pooled)
      Groups 1–4 — LOO models, one per held-out dataset (trained on the other
                   3 datasets, tested on ALL contigs of the held-out dataset)

    Within each group there are 3 bars: Precision, Recall, F1.

    Parameters
    ----------
    loo_metrics   : list of dicts returned by run_lodo, one per held-out dataset
    main_overall  : metric dict for the main model's pooled test set
                    (keys: precision, recall, f1)
    out_path      : output PNG path
    """
    # ── Build groups ──────────────────────────────────────────────────────────
    # Each entry: (group_label, precision, recall, f1, all_unmod)
    groups: list[tuple[str, float, float, float, bool]] = []

    groups.append((
        "Main\n(all datasets)",
        main_overall["precision"],
        main_overall["recall"],
        main_overall["f1"],
        main_overall.get("all_unmod", False),
    ))

    for m in loo_metrics:
        label = f"LOO\n(held-out: {m['held_out']})"
        if m["all_unmod"]:
            label += "*"
        groups.append((label, m["precision"], m["recall"], m["f1"], m["all_unmod"]))

    n_groups = len(groups)
    x        = np.arange(n_groups)
    width    = 0.22          # width of each individual bar
    offsets  = [-width, 0, width]   # Precision, Recall, F1

    COLOURS = {
        "Precision": "#2166ac",
        "Recall":    "#b2182b",
        "F1":        "#1b7837",
    }

    fig, ax = plt.subplots(figsize=(13, 6))

    metric_keys = ["Precision", "Recall", "F1"]
    value_index = [1, 2, 3]   # indices into the groups tuple

    bars_by_metric: dict[str, list] = {}
    for metric, offset, vi in zip(metric_keys, offsets, value_index):
        vals = [g[vi] for g in groups]
        bars = ax.bar(
            x + offset, vals, width,
            label=metric,
            color=COLOURS[metric],
            edgecolor="white",
            linewidth=0.6,
        )
        bars_by_metric[metric] = bars

        # Value annotations (rotated, above each bar)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.012,
                f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=7, rotation=90,
                color="dimgrey",
            )

    # Vertical separator between the Main group and the LOO groups
    ax.axvline(x=0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    # x-axis labels
    tick_labels = [g[0] for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)

    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Main Model vs. Leave-One-Dataset-Out Models\n"
        "Main: trained on all 4 datasets, tested on held-out contigs (80/20 split)  |  "
        "LOO: trained on 3 datasets, tested on entire held-out dataset",
        fontsize=10,
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

    # Footnote if any LOO dataset is all-unmodified
    has_unmod_note = any(g[4] for g in groups)
    if has_unmod_note:
        fig.text(
            0.5, 0.01,
            "* Unmodified dataset: no true positives — metrics computed w.r.t. predicting label=0.",
            ha="center", fontsize=8, color="dimgrey",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        plt.tight_layout()

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  LOO comparison plot saved → {out_path}")


def save_loo_metrics_tsv(
    loo_metrics:  list[dict],
    main_overall: dict,
    out_path:     str = LOO_METRICS_OUT,
) -> None:
    """Write a combined TSV with one row per model (main + 4 LOO)."""
    rows = []

    # Main model row
    rows.append({
        "model":     "Main",
        "held_out":  "all (pooled test set)",
        "precision": round(main_overall["precision"], 4),
        "recall":    round(main_overall["recall"],    4),
        "f1":        round(main_overall["f1"],        4),
        "auprc":     round(main_overall.get("auprc", float("nan")), 4),
        "threshold": round(main_overall.get("threshold", float("nan")), 4),
        "n_train":   main_overall.get("n_train", ""),
        "n_test":    main_overall.get("n_test",  ""),
    })

    # LOO rows
    for m in loo_metrics:
        rows.append({"model": "LODO", **{k: v for k, v in m.items() if k != "all_unmod"}})

    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"  LOO metrics saved → {out_path}")