"""
main.py
=======
Training loop, checkpoint saving, per-position TSV export, leave-one-dataset-out
(LODO) evaluation, and CLI for the Transformer modification classifier.

Pipeline stages
---------------
  [1/7] Load all 4 datasets
  [2/7] Build feature matrix
  [3/7] Contig-level train / test split
  [4/7] Build tiled DataLoaders
  [5/7] Initialise model, loss, optimiser
  [6/7] Train with early stopping
  [7/7] Evaluate + LODO comparison

Usage
-----
  python main.py                              # all defaults
  python main.py --lr 1e-3 --num-layers 4    # override hyperparams
  python main.py --out-dir results/run_01    # redirect all outputs
  python main.py --data-dir /path/to/data    # set data directory
  python main.py --help
"""

import math
import os
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

import config as cfg
from features import load_and_label, build_feature_matrix
from dataset import ContigTileDataset, collate_fn
from model import ModTransformer, count_parameters
from evaluate import (
    evaluate,
    plot_precision_recall,
    plot_training_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    compute_permutation_importance,
)
from loo import (
    get_datasets,
    optimal_threshold,
    run_lodo,
    plot_loo_results,
    save_loo_metrics_tsv,
)


# ── Logging helper ─────────────────────────────────────────────────────────────

def _log(msg: str = "", indent: int = 0) -> None:
    ts     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    prefix = f"[{ts}]"
    pad    = " " * indent
    for line in msg.split("\n"):
        print(f"{prefix}{pad} {line}" if line.strip() else f"{prefix}")


# ── One epoch of training or evaluation ───────────────────────────────────────

def run_epoch(
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


# ── Per-dataset metrics for the main model (printed summary only) ─────────────

def _main_model_metrics_by_dataset(
    pred_df:       pd.DataFrame,
    all_frames:    dict[str, pd.DataFrame],
    opt_threshold: float,
    datasets:      list,
) -> dict[str, dict]:
    tag_rows = []
    for name, frame in all_frames.items():
        tag_rows.append(frame[["ref_name", "ref_pos"]].assign(dataset=name))
    tag_df = pd.concat(tag_rows, ignore_index=True).drop_duplicates(["ref_name", "ref_pos"])

    merged = pred_df.merge(tag_df, on=["ref_name", "ref_pos"], how="left")

    results: dict[str, dict] = {}
    for name, _, _ in datasets:
        subset = merged[merged["dataset"] == name]
        if subset.empty:
            results[name] = {
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "auprc": float("nan"),
                "threshold": opt_threshold, "n_test": 0, "all_unmod": False,
            }
            continue

        y_true = subset["label"].values.astype(np.float32)
        y_prob = subset["prob_modified"].values.astype(np.float32)
        n_test = int(len(subset))

        if y_true.sum() == 0:
            y_pred = (y_prob >= opt_threshold).astype(int)
            results[name] = {
                "precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
                "recall":    float(recall_score(   y_true, y_pred, pos_label=0, zero_division=0)),
                "f1":        float(f1_score(       y_true, y_pred, pos_label=0, zero_division=0)),
                "auprc":     float(np.mean(1 - y_prob)),
                "threshold": opt_threshold,
                "n_test":    n_test,
                "all_unmod": True,
            }
        else:
            y_pred = (y_prob >= opt_threshold).astype(int)
            results[name] = {
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall":    float(recall_score(   y_true, y_pred, zero_division=0)),
                "f1":        float(f1_score(       y_true, y_pred, zero_division=0)),
                "auprc":     float(average_precision_score(y_true, y_prob)),
                "threshold": opt_threshold,
                "n_test":    n_test,
                "all_unmod": False,
            }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    t_start = datetime.now(timezone.utc)
    _log("=" * 68)
    _log("  Transformer Modification Classifier  —  Training Pipeline")
    _log(f"  Run started  : {t_start.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    _log(f"  Random seed  : {cfg.SEED}")
    _log(f"  Compute device : {cfg.DEVICE}")
    _log("=" * 68)

    DATASETS = get_datasets()

    # ── 1. Load all 4 datasets ────────────────────────────────────────────────
    _log()
    _log("[Stage 1/7]  Loading input datasets …")
    all_frames: dict[str, pd.DataFrame] = {}
    for name, path, is_mod in DATASETS:
        _log(f"  Reading  '{name}'  ←  {path}", indent=2)
        all_frames[name] = load_and_label(path, is_mod, name)

    df    = pd.concat(list(all_frames.values()), ignore_index=True)
    total = len(df)
    n_mod = int(df["label"].sum())
    _log()
    _log("  Dataset composition after pooling:", indent=2)
    _log(f"    Total positions  : {total:>12,}", indent=2)
    _log(f"    Unmodified       : {total - n_mod:>12,}  ({100*(total - n_mod)/total:.1f} %)", indent=2)
    _log(f"    Modified         : {n_mod:>12,}  ({100*n_mod/total:.1f} %)", indent=2)

    # ── 2. Build feature matrix ────────────────────────────────────────────────
    _log()
    _log("[Stage 2/7]  Constructing feature matrix …")
    X_df      = build_feature_matrix(df)
    feat_cols = list(X_df.columns)
    _log(f"  Feature dimensionality : {len(feat_cols)}", indent=2)
    _log(f"  Feature columns        : {feat_cols}", indent=2)

    X        = X_df.values.astype(np.float32)
    y        = df["label"].values.astype(np.float32)
    groups   = df["ref_name"].values
    ref_pos  = df["ref_pos"].values
    ref_name = df["ref_name"].values

    # ── 3. Contig-level train / test split ────────────────────────────────────
    _log()
    _log(f"[Stage 3/7]  Partitioning data by contig  "
         f"(test fraction = {cfg.TEST_SIZE:.0%}) …")
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test     = X[train_idx],        X[test_idx]
    y_train, y_test     = y[train_idx],        y[test_idx]
    grp_train, grp_test = groups[train_idx],   groups[test_idx]
    pos_train, pos_test = ref_pos[train_idx],  ref_pos[test_idx]
    rn_train,  rn_test  = ref_name[train_idx], ref_name[test_idx]

    feat_mean = X_train.mean(axis=0, keepdims=True)
    feat_std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train   = (X_train - feat_mean) / feat_std
    X_test    = (X_test  - feat_mean) / feat_std

    _log(f"  Training partition : {len(X_train):>10,} positions  "
         f"({len(np.unique(grp_train))} contigs)", indent=2)
    _log(f"  Test partition     : {len(X_test):>10,} positions  "
         f"({len(np.unique(grp_test))} contigs)", indent=2)

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    _log()
    _log("[Stage 4/7]  Constructing tiled DataLoaders …")
    train_ds = ContigTileDataset(X_train, y_train, grp_train, pos_train, rn_train,
                                 stride=cfg.WINDOW_STRIDE)
    test_ds  = ContigTileDataset(X_test,  y_test,  grp_test,  pos_test,  rn_test)

    num_workers = min(4, os.cpu_count() or 1)
    _log(f"  Training tiles   : {len(train_ds):>8,}  (stride = {cfg.WINDOW_STRIDE})", indent=2)
    _log(f"  Test tiles       : {len(test_ds):>8,}", indent=2)
    _log(f"  DataLoader workers : {num_workers}", indent=2)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    # ── 5. Initialise model, loss, optimiser ─────────────────────────────────
    _log()
    _log("[Stage 5/7]  Initialising Transformer encoder …")
    in_ch = X.shape[1]
    model = ModTransformer(in_channels=in_ch).to(cfg.DEVICE)
    _log(f"  Trainable parameters : {count_parameters(model):,}", indent=2)
    _log(f"  Input channels       : {in_ch}", indent=2)
    _log(f"  Embedding dimension  : {cfg.D_MODEL}", indent=2)
    _log(f"  Attention heads      : {cfg.NHEAD}", indent=2)
    _log(f"  Encoder layers       : {cfg.NUM_LAYERS}", indent=2)
    _log(f"  FFN inner dimension  : {cfg.DIM_FEEDFORWARD}", indent=2)

    n_neg      = int((y_train == 0).sum())
    n_pos      = int((y_train == 1).sum())
    #original weight
    # pos_weight = torch.tensor([math.sqrt(n_neg / max(n_pos, 1))], device=cfg.DEVICE)
    #geometric mean
    # raw_ratio  = n_neg / max(n_pos, 1)
    # pos_weight = torch.tensor([math.sqrt(raw_ratio * math.sqrt(raw_ratio))], device=cfg.DEVICE)
    #raw ratio
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=cfg.DEVICE)
    _log(f"  Class imbalance      : neg = {n_neg:,}  /  pos = {n_pos:,}", indent=2)
    _log(f"  BCEWithLogitsLoss pos_weight (√-scaled) : {pos_weight.item():.4f}", indent=2)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR / 20
    )
    _log(f"  Optimiser            : AdamW  "
         f"(lr = {cfg.LR},  weight_decay = {cfg.WEIGHT_DECAY})", indent=2)
    _log(f"  LR schedule          : CosineAnnealingLR  "
         f"(T_max = {cfg.NUM_EPOCHS},  η_min = {cfg.LR / 20:.2e})", indent=2)

    # ── 6. Training loop with early stopping ─────────────────────────────────
    _log()
    _log(f"[Stage 6/7]  Training  "
         f"(max epochs = {cfg.NUM_EPOCHS},  patience = {cfg.PATIENCE}) …")
    _log(f"  {'Epoch':>6}   {'Train Loss':>12}   {'Val Loss':>12}   {'Val AUPRC':>10}",
         indent=2)
    _log(f"  {'-'*6}   {'-'*12}   {'-'*12}   {'-'*10}", indent=2)

    best_auprc, best_epoch, patience_count, best_state = -1.0, 1, 0, None
    train_losses, val_losses, val_auprcs = [], [], []

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        tr_loss, _, _    = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, vt, vp = run_epoch(model, test_loader,  criterion, None)
        scheduler.step()

        val_auprc = average_precision_score(vt, vp)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_auprcs.append(val_auprc)

        marker = "  ◀  best" if val_auprc > best_auprc else ""
        _log(f"  {epoch:>6d}   {tr_loss:>12.5f}   {val_loss:>12.5f}   "
             f"{val_auprc:>10.4f}{marker}", indent=2)

        if val_auprc > best_auprc:
            best_auprc     = val_auprc
            best_epoch     = epoch
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.PATIENCE:
                _log()
                _log(f"  Early stopping triggered at epoch {epoch}  "
                     f"(no improvement for {cfg.PATIENCE} consecutive epochs).", indent=2)
                break

    model.load_state_dict(best_state)
    model.to(cfg.DEVICE)
    _log()
    _log(f"  Training complete.  Best validation AUPRC = {best_auprc:.4f}  "
         f"(epoch {best_epoch})", indent=2)

    # ── 7. Evaluate + LODO ───────────────────────────────────────────────────
    _log()
    _log("[Stage 7/7]  Evaluation and leave-one-dataset-out (LODO) analysis …")

    plot_training_curves(train_losses, val_losses, val_auprcs, best_epoch)
    _log("  Training curve figure written.", indent=2)

    _log()
    _log("  Evaluating on held-out test contigs …", indent=2)
    _, y_true_all, y_prob_all = run_epoch(model, test_loader, criterion, None)

    auprc         = average_precision_score(y_true_all, y_prob_all)
    opt_threshold = plot_precision_recall(y_true_all, y_prob_all, auprc)

    _log()
    _log(f"  ── Metrics at default threshold ({cfg.THRESHOLD}) ──", indent=2)
    evaluate(y_true_all, y_prob_all, threshold=cfg.THRESHOLD)
    _log(f"  ── Metrics at optimal threshold ({opt_threshold:.4f}) ──", indent=2)
    evaluate(y_true_all, y_prob_all, threshold=opt_threshold)

    plot_confusion_matrix(y_true_all, y_prob_all, cfg.THRESHOLD,
                          out_path=cfg.CONFUSION_DEFAULT_OUT)
    plot_confusion_matrix(y_true_all, y_prob_all, opt_threshold,
                          out_path=cfg.CONFUSION_OPTIMAL_OUT)
    _log("  Confusion matrix figures written (default and optimal thresholds).", indent=2)

    # ── Save model checkpoint ─────────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feat_mean":        feat_mean,
            "feat_std":         feat_std,
            "feat_cols":        feat_cols,
            "opt_threshold":    opt_threshold,
            "in_channels":      in_ch,
            "d_model":          cfg.D_MODEL,
            "nhead":            cfg.NHEAD,
            "num_layers":       cfg.NUM_LAYERS,
            "dim_feedforward":  cfg.DIM_FEEDFORWARD,
        },
        cfg.MODEL_OUT,
    )
    _log(f"  Model checkpoint saved  →  {cfg.MODEL_OUT}", indent=2)

    # ── Per-position predictions TSV ──────────────────────────────────────────
    model.eval()
    records = []
    with torch.no_grad():
        for x, y_batch, mask, meta in test_loader:
            x       = x.to(cfg.DEVICE)
            mask_d  = mask.to(cfg.DEVICE)
            logits  = model(x, mask=mask_d)
            probs   = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            y_np    = y_batch.squeeze(1).numpy()
            mask_np = mask.numpy()

            for real, prob, label, (rn, rpos) in zip(
                mask_np.reshape(-1), probs.reshape(-1),
                y_np.reshape(-1), meta,
            ):
                if real:
                    records.append({
                        "ref_name":           rn,
                        "ref_pos":            int(rpos),
                        "label":              int(label),
                        "prob_modified":      float(prob),
                        "pred_label_default": int(prob >= cfg.THRESHOLD),
                        "pred_label_optimal": int(prob >= opt_threshold),
                    })

    pred_df = pd.DataFrame(records)
    pred_df["pred_class_optimal"] = pred_df["pred_label_optimal"].map(
        {0: "unmod", 1: "modified"}
    )
    pred_df.to_csv(cfg.PRED_OUT, sep="\t", index=False)
    _log(f"  Per-position predictions saved  →  {cfg.PRED_OUT}", indent=2)
    _log(f"  Recommended deployment threshold : {opt_threshold:.4f}", indent=2)

    # ── Permutation feature importance ────────────────────────────────────────
    _log()
    _log("  Computing permutation feature importance  "
         "(n_repeats = 5; this may take several minutes) …", indent=2)
    imp_df, baseline_auprc = compute_permutation_importance(
        model, X_test, y_test, grp_test, pos_test, rn_test,
        feat_cols, num_workers=num_workers, n_repeats=5,
    )
    _log()
    _log("  Feature importance — top 20 features by ΔAUPRC:", indent=2)
    _log(imp_df.head(20).to_string(index=False), indent=4)
    plot_feature_importance(imp_df, baseline_auprc)
    _log("  Feature importance figure written.", indent=2)

    # ── Main model: pooled overall metrics (for the LOO comparison plot) ──────
    y_pred_opt = (y_prob_all >= opt_threshold).astype(int)
    main_overall = {
        "precision": float(precision_score(y_true_all, y_pred_opt, zero_division=0)),
        "recall":    float(recall_score(   y_true_all, y_pred_opt, zero_division=0)),
        "f1":        float(f1_score(       y_true_all, y_pred_opt, zero_division=0)),
        "auprc":     float(auprc),
        "threshold": float(opt_threshold),
        "n_train":   int(len(X_train)),
        "n_test":    int(len(X_test)),
        "all_unmod": False,
    }

    # ── Per-dataset breakdown (printed summary only) ───────────────────────────
    _log()
    _log("  Computing per-dataset performance breakdown for the main model …", indent=2)
    main_metrics_by_ds = _main_model_metrics_by_dataset(pred_df, all_frames, opt_threshold, DATASETS)
    _log(f"  {'Dataset':<10}  {'Precision':>10}  {'Recall':>8}  "
         f"{'F1':>8}  {'AUPRC':>8}  {'N test':>10}  Notes", indent=2)
    _log(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*30}",
         indent=2)
    for name, mm in main_metrics_by_ds.items():
        note = "all-unmodified: metrics computed against label=0" if mm["all_unmod"] else ""
        _log(f"  {name:<10}  {mm['precision']:>10.4f}  {mm['recall']:>8.4f}  "
             f"{mm['f1']:>8.4f}  {mm['auprc']:>8.4f}  {mm['n_test']:>10,}  {note}", indent=2)

    # ── LODO: train 4 models, each leaving one dataset out ───────────────────
    _log()
    _log("  Initiating leave-one-dataset-out (LODO) evaluation …", indent=2)
    loo_metrics: list[dict] = []
    for name, _, _ in DATASETS:
        _log(f"    Holding out dataset '{name}' …", indent=2)
        result = run_lodo(name, all_frames, num_workers)
        loo_metrics.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    _log()
    _log("=" * 68)
    _log("  LODO Evaluation Summary")
    _log("=" * 68)
    loo_df = pd.DataFrame([{k: v for k, v in m.items() if k != "all_unmod"}
                           for m in loo_metrics])
    _log(loo_df.to_string(index=False))
    _log("=" * 68)

    # ── LODO comparison plot + TSV ────────────────────────────────────────────
    plot_loo_results(loo_metrics, main_overall)
    save_loo_metrics_tsv(loo_metrics, main_overall)
    _log("  LODO comparison plot and metrics TSV written.", indent=2)

    t_end     = datetime.now(timezone.utc)
    elapsed   = t_end - t_start
    h, rem    = divmod(int(elapsed.total_seconds()), 3600)
    m, s      = divmod(rem, 60)
    _log()
    _log("=" * 68)
    _log(f"  Pipeline complete.")
    _log(f"  Run finished : {t_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    _log(f"  Elapsed time : {h:02d}h {m:02d}m {s:02d}s")
    _log("=" * 68)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Transformer encoder for per-base modification detection."
    )
    p.add_argument("--data-dir",        type=str,   default=None, metavar="DIR",
                   help=f"Directory containing control.tsv, 5mC.tsv, 5hmC.tsv, 6mA.tsv "
                        f"(default: {cfg.DATA_DIR})")
    p.add_argument("--batch-size",      type=int,   default=None, metavar="N",
                   help=f"Mini-batch size (default: {cfg.BATCH_SIZE})")
    p.add_argument("--lr",              type=float, default=None, metavar="F",
                   help=f"Learning rate (default: {cfg.LR})")
    p.add_argument("--weight-decay",    type=float, default=None, metavar="F",
                   help=f"AdamW weight decay (default: {cfg.WEIGHT_DECAY})")
    p.add_argument("--window-size",     type=int,   default=None, metavar="N",
                   help=f"Tile / context window in positions (default: {cfg.WINDOW_SIZE})")
    p.add_argument("--window-stride",   type=int,   default=None, metavar="N",
                   help="Stride for training tiles; default = window-size")
    p.add_argument("--num-epochs",      type=int,   default=None, metavar="N",
                   help=f"Maximum training epochs (default: {cfg.NUM_EPOCHS})")
    p.add_argument("--patience",        type=int,   default=None, metavar="N",
                   help=f"Early-stopping patience (default: {cfg.PATIENCE})")
    p.add_argument("--threshold",       type=float, default=None, metavar="F",
                   help=f"Classification threshold (default: {cfg.THRESHOLD})")
    p.add_argument("--seed",            type=int,   default=None, metavar="N",
                   help=f"Random seed (default: {cfg.SEED})")
    p.add_argument("--d-model",         type=int,   default=None, metavar="N",
                   help=f"Attention / embedding dimension (default: {cfg.D_MODEL})")
    p.add_argument("--nhead",           type=int,   default=None, metavar="N",
                   help=f"Attention heads; must divide d-model (default: {cfg.NHEAD})")
    p.add_argument("--num-layers",      type=int,   default=None, metavar="N",
                   help=f"Encoder layer stacks (default: {cfg.NUM_LAYERS})")
    p.add_argument("--dim-feedforward", type=int,   default=None, metavar="N",
                   help=f"FFN inner dimension (default: {cfg.DIM_FEEDFORWARD})")
    p.add_argument("--dropout",         type=float, default=None, metavar="F",
                   help=f"Dropout probability (default: {cfg.DROPOUT})")
    p.add_argument("--out-dir",         type=str,   default=None, metavar="DIR",
                   help="Directory for all output files (default: current directory)")
    return p.parse_args()


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """Push CLI arguments into the config module so all imports pick them up."""
    if args.data_dir is not None:
        cfg.DATA_DIR  = args.data_dir
        cfg.TSV_UNMOD = os.path.join(args.data_dir, "control.tsv")
        cfg.TSV_5MC   = os.path.join(args.data_dir, "5mC.tsv")
        cfg.TSV_5HMC  = os.path.join(args.data_dir, "5hmC.tsv")
        cfg.TSV_6MA   = os.path.join(args.data_dir, "6mA.tsv")
    if args.batch_size      is not None: cfg.BATCH_SIZE      = args.batch_size
    if args.lr              is not None: cfg.LR              = args.lr
    if args.weight_decay    is not None: cfg.WEIGHT_DECAY    = args.weight_decay
    if args.window_size     is not None: cfg.WINDOW_SIZE     = args.window_size
    if args.window_stride   is not None: cfg.WINDOW_STRIDE   = args.window_stride
    if args.num_epochs      is not None: cfg.NUM_EPOCHS      = args.num_epochs
    if args.patience        is not None: cfg.PATIENCE        = args.patience
    if args.threshold       is not None: cfg.THRESHOLD       = args.threshold
    if args.seed            is not None: cfg.SEED            = args.seed
    if args.d_model         is not None: cfg.D_MODEL         = args.d_model
    if args.nhead           is not None: cfg.NHEAD           = args.nhead
    if args.num_layers      is not None: cfg.NUM_LAYERS      = args.num_layers
    if args.dim_feedforward is not None: cfg.DIM_FEEDFORWARD = args.dim_feedforward
    if args.dropout         is not None: cfg.DROPOUT         = args.dropout

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        cfg.OUT_DIR = args.out_dir
        for attr in (
            "MODEL_OUT", "PRED_OUT", "PR_FIG_OUT", "TRAIN_FIG_OUT",
            "CONFUSION_DEFAULT_OUT", "CONFUSION_OPTIMAL_OUT",
            "FEAT_IMP_FIG_OUT", "LOO_FIG_OUT", "LOO_METRICS_OUT",
        ):
            setattr(cfg, attr, os.path.join(args.out_dir, os.path.basename(getattr(cfg, attr))))
        cfg.LOO_TRAIN_FIG_PREFIX = os.path.join(args.out_dir, "transformer_loo_training_curves")


if __name__ == "__main__":
    args = _parse_cli()
    _apply_cli_overrides(args)
    main()