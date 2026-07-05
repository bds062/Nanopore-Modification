#!/usr/bin/env python3
"""
loo.py
======
Leave-one-dataset-out (LODO) evaluation for the Pileup InceptionV3 CNN.
Imported and called by train_cnn.py — not a standalone script.

Public API
----------
  infer_dataset_name   — derive a short label from an HDF5 filename
  run_lodo             — train on N-1 files, evaluate on the held-out file
  zeror_metrics        — compute ZeroR (majority-class) baseline metrics
  optimal_threshold    — threshold that maximises F1
  plot_loo_results     — 6-group bar chart: base + 4 LOO + ZeroR
  save_loo_metrics_tsv — write results table to TSV
"""

import copy
import sys
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
try:
    from .visualization import (
        CHANNEL_NAMES,
        plot_channel_importance,
        plot_loo_results,
        save_loo_metrics_tsv,
        save_training_history,
    )
except ImportError:  # support direct execution from this directory
    from visualization import (
        CHANNEL_NAMES,
        plot_channel_importance,
        plot_loo_results,
        save_loo_metrics_tsv,
        save_training_history,
    )


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _lodo_best_path(out_dir: Path, held_out_name: str) -> Path:
    return out_dir / f'lodo_{held_out_name}_best.pt'


def _lodo_state_path(out_dir: Path, held_out_name: str) -> Path:
    return out_dir / f'lodo_{held_out_name}_state.pt'


def _save_lodo_state(
    path: Path,
    *,
    held_out_name: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    best_auprc: float,
    best_epoch: int,
    patience_count: int,
    train_losses: list[float],
    val_losses: list[float],
    val_auprcs: list[float],
    global_step: int = 0,
    warmup_sched=None,
) -> None:
    state = {
        'held_out': held_out_name,
        'epoch': int(epoch),
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_auprc': float(best_auprc),
        'best_epoch': int(best_epoch),
        'patience_count': int(patience_count),
        'train_losses': list(train_losses),
        'val_losses': list(val_losses),
        'val_auprcs': list(val_auprcs),
        'global_step': int(global_step),
    }
    if warmup_sched is not None:
        state['warmup_sched_state'] = warmup_sched.state_dict()
    torch.save(state, path)


# ── dataset name helper ───────────────────────────────────────────────────────

def infer_dataset_name(h5_path: str) -> str:
    """
    Derive a short human-readable label from an HDF5 filename.

    Examples
    --------
    'data/control.h5'  → 'control'
    'data/5mC.h5'      → '5mC'
    'data/6mA.h5'      → '6mA'
    """
    return Path(h5_path).stem


# ── threshold & metrics helpers ───────────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the decision threshold that maximises F1 on y_true / y_prob."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions[:-1] * recalls[:-1]
           / (precisions[:-1] + recalls[:-1] + 1e-8))
    return float(thresholds[np.argmax(f1s)])


def _compute_metrics(y_true: np.ndarray,
                     y_prob: np.ndarray,
                     threshold: float | None = None) -> dict:
    """
    Compute precision, recall, F1, AUPRC at the optimal (or supplied) threshold.

    All-unmodified case
    -------------------
    If the test set contains no positives (e.g. the control file), standard
    metrics are undefined for the positive class.  We flip to the negative
    class (label=0) at threshold=0.5 and approximate AUPRC as the mean
    probability of predicting unmodified.  The returned dict carries
    ``all_unmod=True`` so callers can annotate plots.
    """
    if int(y_true.sum()) == 0:
        thresh = 0.5
        y_pred = (y_prob >= thresh).astype(int)
        return {
            'precision': float(precision_score(y_true, y_pred,
                                               pos_label=0, zero_division=0)),
            'recall':    float(recall_score(y_true, y_pred,
                                            pos_label=0, zero_division=0)),
            'f1':        float(f1_score(y_true, y_pred,
                                        pos_label=0, zero_division=0)),
            'auprc':     float(np.mean(1.0 - y_prob)),
            'auroc':     float('nan'),
            'threshold': thresh,
            'all_unmod': True,
        }

    thresh = threshold if threshold is not None else optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'auprc':     float(average_precision_score(y_true, y_prob)),
        'auroc':     float(roc_auc_score(y_true, y_prob)),
        'threshold': thresh,
        'all_unmod': False,
    }


def zeror_metrics(train_labels: np.ndarray,
                  test_labels:  np.ndarray) -> dict:
    """
    ZeroR baseline: always predict the majority class from the training set.

    Metrics are always reported w.r.t. the predicted (majority) class, not
    hardcoded to the positive class.  This avoids the degenerate all-zeros
    result that occurs when majority=0 and sklearn evaluates against label=1.

    For example, if ZeroR always predicts 0 (unmodified), we report
    precision/recall/F1 for label=0 — which reflects how well the trivial
    "always say unmodified" baseline actually performs.

    Parameters
    ----------
    train_labels : used to determine the majority class
    test_labels  : evaluated against

    Returns
    -------
    dict with precision, recall, f1, majority_class, all_unmod
    """
    n_pos    = int(train_labels.sum())
    n_neg    = len(train_labels) - n_pos
    majority = 1 if n_pos >= n_neg else 0

    y_pred   = np.full(len(test_labels), majority, dtype=int)
    all_unmod = bool(int(test_labels.sum()) == 0)

    # Always evaluate w.r.t. the class ZeroR actually predicts
    return {
        'precision':      float(precision_score(test_labels, y_pred,
                                                pos_label=majority,
                                                zero_division=0)),
        'recall':         float(recall_score(test_labels, y_pred,
                                             pos_label=majority,
                                             zero_division=0)),
        'f1':             float(f1_score(test_labels, y_pred,
                                         pos_label=majority,
                                         zero_division=0)),
        'auprc':          float('nan'),   # ZeroR has no probability output
        'majority_class': majority,
        'all_unmod':      all_unmod,
        'test_labels':    test_labels,
    }


def _make_loader_kwargs(batch_size: int,
                        num_workers: int,
                        device: torch.device,
                        worker_init_fn=None) -> dict:
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )
    if num_workers > 0:
        kwargs['worker_init_fn'] = worker_init_fn
        kwargs['persistent_workers'] = True
    return kwargs


def _binary_labels(labels: np.ndarray) -> np.ndarray:
    return (labels.astype(np.int64) > 0).astype(np.int64)


def _decode_ref_names(values: np.ndarray) -> list[str]:
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode('utf-8'))
        else:
            names.append(str(value))
    return names


def _make_position_keys(ref_names: np.ndarray, ref_pos: np.ndarray) -> list[tuple]:
    names = _decode_ref_names(ref_names)
    return list(zip(names, ref_pos.astype(np.int64).tolist()))


def _split_position_groups(labels: np.ndarray,
                           position_keys: list[tuple],
                           val_frac: float,
                           seed: int):
    group_to_indices: dict[tuple, list[int]] = {}
    group_to_label: dict[tuple, int] = {}
    for i, key in enumerate(position_keys):
        group_to_indices.setdefault(key, []).append(i)
        group_to_label[key] = max(group_to_label.get(key, 0), int(labels[i] > 0))

    by_label = {
        0: [k for k, v in group_to_label.items() if v == 0],
        1: [k for k, v in group_to_label.items() if v == 1],
    }
    rng = np.random.default_rng(seed)
    train_keys, val_keys = [], []

    for label in (0, 1):
        keys = list(by_label[label])
        rng.shuffle(keys)
        n = len(keys)
        if n == 0:
            continue
        n_val = int(n * val_frac)
        if val_frac > 0 and n_val == 0 and n >= 2:
            n_val = 1
        if n_val >= n:
            n_val = max(0, n - 1)
        val_keys.extend(keys[:n_val])
        train_keys.extend(keys[n_val:])

    def expand(keys: list[tuple]) -> np.ndarray:
        idx = []
        for key in keys:
            idx.extend(group_to_indices[key])
        return np.array(idx, dtype=np.int64)

    stats = {
        'train_positions': len(train_keys),
        'val_positions':   len(val_keys),
    }
    return expand(train_keys), expand(val_keys), stats


def _aggregate_by_position(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           position_keys: list[tuple]):
    grouped: dict[tuple, list] = {}
    for yt, yp, key in zip(y_true, y_prob, position_keys):
        if key not in grouped:
            grouped[key] = [0, []]
        grouped[key][0] = max(grouped[key][0], int(yt > 0))
        grouped[key][1].append(float(yp))

    keys, labels, probs = [], [], []
    for key, (label, values) in grouped.items():
        keys.append(key)
        labels.append(label)
        probs.append(float(np.mean(values)))
    return (np.array(labels, dtype=np.int64),
            np.array(probs, dtype=np.float32),
            keys)


# ── LOO training loop ─────────────────────────────────────────────────────────

def run_lodo(
    held_out_path:  str,
    all_h5_paths:   list[str],
    PileupDataset,           # class — passed in to avoid circular import
    PileupInceptionV3,       # class
    get_pos_weight,          # function
    make_sampler,            # function
    run_inference,           # function
    plot_training_curves,    # function
    device:         torch.device,
    out_dir:        Path,
    # training hyper-parameters
    epochs:         int   = 50,
    batch:          int   = 32,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    patience:       int   = 10,
    dropout:        float = 0.5,
    num_workers:    int   = 4,
    no_oversample:  bool  = False,
    seed:           int   = 42,
    val_frac:       float = 0.15,
    use_focal:      bool  = False,
    focal_gamma:    float = 2.0,
    rc_augment:     bool  = False,
    signal_noise_std: float = 0.05,
    resume:         bool  = True,
    delta_channels: bool  = True,
    lr_warmup_steps: int  = 0,
    cross_read_attention: bool = False,
    grad_clip: float = 1.0,
    supcon_weight: float = 0.0,
    supcon_temp:   float = 0.07,
    supcon_proj_dim: int = 128,
) -> dict:
    """
    Train on every HDF5 file except ``held_out_path``.
    Test on ALL positions in ``held_out_path`` (no train/val split within it —
    all filters were already applied at featurization time).

    Returns
    -------
    dict with keys:
      held_out, precision, recall, f1, auprc, auroc, threshold,
      n_train, n_test, all_unmod,
      train_labels (np.ndarray),
      test_labels  (np.ndarray),
      y_prob       (np.ndarray)   — model probabilities on the test set
    """
    held_out_name = infer_dataset_name(held_out_path)
    train_paths   = [p for p in all_h5_paths if p != held_out_path]

    print(f"\n{'═'*60}", flush=True)
    print(f"  LODO fold: held-out = {held_out_name}")
    print(f"  Train files: {[infer_dataset_name(p) for p in train_paths]}")
    print(f"{'═'*60}", flush=True)

    # ── read sizes and labels from each file ──────────────────────────────────
    train_sizes  = []
    train_labels_list = []
    train_position_keys = []
    for p in train_paths:
        with h5py.File(p, 'r') as hf:
            lbl = _binary_labels(hf['labels'][:])
            train_position_keys.extend(_make_position_keys(
                hf['ref_names'][:], hf['ref_pos'][:]))
            train_labels_list.append(lbl)
            train_sizes.append(len(lbl))

    with h5py.File(held_out_path, 'r') as hf:
        test_labels_full = _binary_labels(hf['labels'][:])
        test_position_keys_full = _make_position_keys(
            hf['ref_names'][:], hf['ref_pos'][:])
        attrs            = dict(hf.attrs)

    train_labels_all = np.concatenate(train_labels_list)
    train_file_sizes = np.array(train_sizes, dtype=np.int64)
    test_file_sizes  = np.array([len(test_labels_full)], dtype=np.int64)

    # ── validation split within training data (stratified by position) ───────
    train_idx, val_idx, split_stats = _split_position_groups(
        train_labels_all, train_position_keys, val_frac=val_frac, seed=seed)

    # Test set = all positions in the held-out file
    test_idx = np.arange(len(test_labels_full), dtype=np.int64)

    in_ch  = int(attrs.get('n_channels', 9)) + (2 if delta_channels else 0)
    height = int(attrs.get('height', attrs.get('max_reads', 30) + 1))
    W      = int(attrs.get('W', 21))
    L      = int(attrs.get('L', 10))

    n_mod   = int(test_labels_full.sum())
    n_unmod = len(test_labels_full) - n_mod
    print(f"  Train images: {len(train_idx):,}  |  Val images: {len(val_idx):,}  |  "
          f"Test (held-out): {len(test_idx):,}  "
          f"[mod={n_mod}, unmod={n_unmod}]", flush=True)
    print(f"  Train positions: {split_stats['train_positions']:,}  |  "
          f"Val positions: {split_stats['val_positions']:,}", flush=True)

    train_ds = PileupDataset(train_paths, train_idx, train_file_sizes,
                              augment=True,  seed=seed,
                              rc_augment=rc_augment,
                              signal_noise_std=signal_noise_std,
                              delta_channels=delta_channels)
    val_ds   = PileupDataset(train_paths, val_idx,   train_file_sizes,
                              augment=False, seed=seed,
                              delta_channels=delta_channels)
    test_ds  = PileupDataset([held_out_path], test_idx, test_file_sizes,
                              augment=False, seed=seed,
                              delta_channels=delta_channels)

    dataset_mod = sys.modules.get(PileupDataset.__module__)
    _worker_init_fn = getattr(dataset_mod, '_worker_init_fn', None)
    FocalBCELoss = getattr(dataset_mod, 'FocalBCELoss', None)
    SupConLoss_cls = getattr(dataset_mod, 'SupConLoss', None)
    DatasetBalancedSampler_cls = getattr(dataset_mod, 'DatasetBalancedSampler', None)
    if FocalBCELoss is None:
        try:
            from .model import FocalBCELoss as _FocalBCELoss
            FocalBCELoss = _FocalBCELoss
        except ImportError:
            try:
                from model import FocalBCELoss as _FocalBCELoss
                FocalBCELoss = _FocalBCELoss
            except ImportError:
                FocalBCELoss = None

    train_labels_split = train_labels_all[train_idx]
    if supcon_weight > 0 and DatasetBalancedSampler_cls is not None:
        sampler = DatasetBalancedSampler_cls(
            global_indices=train_idx,
            file_sizes=train_file_sizes,
            labels=train_labels_all,
            num_samples=len(train_idx),
            seed=seed,
        )
    else:
        sampler = None if no_oversample else make_sampler(train_labels_split)

    loader_kwargs = _make_loader_kwargs(
        batch_size=batch,
        num_workers=num_workers,
        device=device,
        worker_init_fn=_worker_init_fn,
    )
    if num_workers == 0:
        print("  DataLoader: num_workers=0 (single-process HDF5 reads)",
              flush=True)
    else:
        print(f"  DataLoader: num_workers={num_workers}  "
              f"pin_memory={loader_kwargs['pin_memory']}",
              flush=True)
    train_loader = DataLoader(train_ds, sampler=sampler,
                               shuffle=(sampler is None), **loader_kwargs)
    val_loader   = DataLoader(val_ds,  shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # ── model — fresh weights for each fold ───────────────────────────────────
    model = PileupInceptionV3(
        in_channels=in_ch,
        dropout=dropout,
        cross_read_attention=cross_read_attention,
        supcon_proj_dim=supcon_proj_dim if supcon_weight > 0 else 0,
    ).to(device)

    pw         = float(get_pos_weight(train_labels_split, device).item())
    pw_tensor  = torch.tensor(pw, device=device)
    if use_focal and FocalBCELoss is not None:
        criterion = FocalBCELoss(pos_weight=pw, gamma=focal_gamma)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  |  "
              f"Focal pos_weight={pw:.2f}  gamma={focal_gamma}", flush=True)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  |  "
              f"BCE pos_weight={pw:.2f}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)

    warmup_sched = None
    if lr_warmup_steps > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(lr_warmup_steps, 1),
            end_factor=1.0,
            total_iters=lr_warmup_steps,
        )

    supcon_criterion = None
    if supcon_weight > 0 and SupConLoss_cls is not None:
        supcon_criterion = SupConLoss_cls(temperature=supcon_temp)

    best_auprc     = -1.0
    best_epoch     = 1
    patience_count = 0
    global_step    = 0
    best_state     = None
    train_losses, val_losses, val_auprcs = [], [], []
    start_epoch = 1

    state_path = _lodo_state_path(out_dir, held_out_name)
    ckpt_path = _lodo_best_path(out_dir, held_out_name)
    if resume and state_path.exists():
        try:
            state = torch.load(state_path, map_location=device, weights_only=False)
            if state.get('held_out') == held_out_name:
                model.load_state_dict(state['model_state'])
                optimizer.load_state_dict(state['optimizer_state'])
                scheduler.load_state_dict(state['scheduler_state'])
                best_auprc = float(state.get('best_auprc', best_auprc))
                best_epoch = int(state.get('best_epoch', best_epoch))
                patience_count = int(state.get('patience_count', patience_count))
                train_losses = list(state.get('train_losses', []))
                val_losses = list(state.get('val_losses', []))
                val_auprcs = list(state.get('val_auprcs', []))
                global_step = int(state.get('global_step', 0))
                start_epoch = int(state['epoch']) + 1
                if warmup_sched is not None and 'warmup_sched_state' in state:
                    warmup_sched.load_state_dict(state['warmup_sched_state'])
                print(f"  [RESUME] Continuing LODO '{held_out_name}' from "
                      f"epoch {start_epoch}/{epochs} "
                      f"(best epoch {best_epoch}, AUPRC={best_auprc:.4f})",
                      flush=True)
            else:
                print(f"  WARNING: ignoring mismatched LODO state {state_path}",
                      flush=True)
        except Exception as exc:
            print(f"  WARNING: could not load LODO state {state_path}: {exc}. "
                  f"Restarting fold.", flush=True)

    if start_epoch > epochs:
        print(f"  [RESUME] LODO '{held_out_name}' already completed "
              f"{epochs} epochs; evaluating best checkpoint.", flush=True)
    elif patience_count >= patience:
        print(f"  [RESUME] LODO '{held_out_name}' had already reached "
              f"early-stopping patience; evaluating best checkpoint.",
              flush=True)
        start_epoch = epochs + 1

    for epoch in range(start_epoch, epochs + 1):
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        # ── train ────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_int = y.long()
            optimizer.zero_grad()
            if supcon_criterion is not None:
                logits, z = model(x, return_embedding=True)
                bce_loss = criterion(logits.squeeze(1), y)
                sc_loss  = supcon_criterion(z, y_int)
                loss = bce_loss + supcon_weight * sc_loss
            else:
                loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            global_step += 1
            if warmup_sched is not None and global_step <= lr_warmup_steps:
                warmup_sched.step()
            epoch_loss += loss.item() * len(y)
        train_loss = epoch_loss / len(train_ds)

        # ── validate ─────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss_sum += criterion(model(x).squeeze(1), y).item() * len(y)
        val_loss = val_loss_sum / len(val_ds)

        y_true_val_img, y_prob_val_img = run_inference(model, val_loader, device)
        val_position_keys = [train_position_keys[i] for i in val_idx]
        y_true_val, y_prob_val, _ = _aggregate_by_position(
            y_true_val_img, y_prob_val_img, val_position_keys)

        # Val AUPRC: use negative-class proxy if no positives in val split
        if int(y_true_val.sum()) > 0:
            val_auprc = average_precision_score(y_true_val, y_prob_val)
        else:
            val_auprc = float(np.mean(1.0 - y_prob_val))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_auprcs.append(val_auprc)
        if warmup_sched is None or global_step > lr_warmup_steps:
            scheduler.step(val_auprc)

        # ReduceLROnPlateau has no get_last_lr() — read directly from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"valAUPRC={val_auprc:.4f}  "
              f"lr={current_lr:.2e}", flush=True)

        if val_auprc > best_auprc:
            best_auprc     = val_auprc
            best_epoch     = epoch
            patience_count = 0
            best_state     = copy.deepcopy(model.state_dict())
            torch.save({'model_state': best_state,
                        'held_out':    held_out_name,
                        'epoch':       best_epoch,
                        'val_auprc':   best_auprc}, ckpt_path)
            print(f"    ✓ New best  AUPRC={best_auprc:.4f}", flush=True)
        else:
            patience_count += 1

        _save_lodo_state(
            state_path,
            held_out_name=held_out_name,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_auprc=best_auprc,
            best_epoch=best_epoch,
            patience_count=patience_count,
            train_losses=train_losses,
            val_losses=val_losses,
            val_auprcs=val_auprcs,
            global_step=global_step,
            warmup_sched=warmup_sched,
        )

        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch}.", flush=True)
            break

    # ── restore best & evaluate on held-out set ───────────────────────────────
    if best_state is None:
        if not ckpt_path.exists():
            raise RuntimeError(
                f"No best checkpoint available for LODO fold {held_out_name}; "
                f"cannot evaluate.")
        best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        best_state = best_ckpt['model_state']
        best_epoch = int(best_ckpt.get('epoch', best_epoch))
        best_auprc = float(best_ckpt.get('val_auprc', best_auprc))
    model.load_state_dict(best_state)
    print(f"  Restored best model (epoch {best_epoch}, "
          f"val AUPRC={best_auprc:.4f})", flush=True)

    # Save per-fold training curves
    curves_path = str(out_dir / f'training_curves_lodo_{held_out_name}.png')
    plot_training_curves(train_losses, val_losses, val_auprcs,
                          best_epoch, out_path=curves_path)
    save_training_history(
        out_dir / f'training_history_lodo_{held_out_name}.npz',
        train_losses, val_losses, val_auprcs, best_epoch,
    )

    # Keep best checkpoint fresh even when resumed from a completed state.
    torch.save({'model_state': best_state,
                'held_out':    held_out_name,
                'epoch':       best_epoch,
                'val_auprc':   best_auprc}, ckpt_path)

    y_true_test_img, y_prob_test_img = run_inference(model, test_loader, device)
    y_true_test, y_prob_test, _ = _aggregate_by_position(
        y_true_test_img, y_prob_test_img, test_position_keys_full)
    m = _compute_metrics(y_true_test, y_prob_test)

    note = ' [all-unmod: metrics vs label=0]' if m['all_unmod'] else ''
    print(f"  Held-out results{note}:", flush=True)
    print(f"    Prec={m['precision']:.4f}  Rec={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  AUPRC={m['auprc']:.4f}", flush=True)

    return {
        'held_out':     held_out_name,
        'precision':    round(m['precision'], 4),
        'recall':       round(m['recall'],    4),
        'f1':           round(m['f1'],        4),
        'auprc':        round(m['auprc'],     4),
        'auroc':        round(m['auroc'],     4) if not np.isnan(m['auroc']) else float('nan'),
        'threshold':    round(m['threshold'], 4),
        'n_train':      int(len(train_idx)),
        'n_test':       int(len(y_true_test)),
        'all_unmod':    m['all_unmod'],
        'train_labels': train_labels_split,   # needed for ZeroR computation
        'test_labels':  y_true_test,
        'y_prob':       y_prob_test,
    }


# ── channel importance computation ────────────────────────────────────────────

@torch.no_grad()
def permutation_channel_importance(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    n_channels:  int = 9,
    n_repeats:   int = 5,
    seed:        int = 42,
) -> np.ndarray:
    """
    Permutation-based channel importance for the pileup CNN.

    For each channel c:
      1. Replace that channel across every sample in the loader with values
         drawn from its empirical distribution (shuffle within the batch along
         the batch dimension — equivalent to random permutation without needing
         to load the full dataset into memory).
      2. Run inference and compute AUPRC.
      3. Importance = baseline_AUPRC − permuted_AUPRC.
         Positive values mean the channel is useful; negative values indicate
         the model performs *better* without it (rare, but possible with
         correlated channels).

    Repeats the permutation ``n_repeats`` times and returns the mean drop.

    Parameters
    ----------
    model      : trained PileupInceptionV3 in eval mode
    loader     : DataLoader (no augmentation, fixed order)
    device
    n_channels : number of input channels (default 9)
    n_repeats  : permutation repeats for stability
    seed       : RNG seed

    Returns
    -------
    importance : np.ndarray shape (n_channels,)
                 Mean AUPRC drop when each channel is permuted.
    """
    rng = np.random.default_rng(seed)
    model.eval()

    # ── baseline AUPRC ────────────────────────────────────────────────────────
    all_y, all_p = [], []
    for x, y in loader:
        logits = model(x.to(device)).squeeze(1)
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(y.numpy())
    y_true    = np.concatenate(all_y)
    y_base    = np.concatenate(all_p)
    base_auprc = (average_precision_score(y_true, y_base)
                  if int(y_true.sum()) > 0 else float(np.mean(1.0 - y_base)))

    # ── per-channel permutation ───────────────────────────────────────────────
    drops = np.zeros((n_channels, n_repeats), dtype=np.float64)

    for c in range(n_channels):
        for r in range(n_repeats):
            all_p_perm = []
            for x, _ in loader:
                x = x.clone()
                # Permute channel c along the batch dimension
                perm = rng.permutation(x.shape[0])
                x[:, c, :, :] = x[perm, c, :, :]
                logits = model(x.to(device)).squeeze(1)
                all_p_perm.append(torch.sigmoid(logits).cpu().numpy())
            y_perm = np.concatenate(all_p_perm)
            perm_auprc = (average_precision_score(y_true, y_perm)
                          if int(y_true.sum()) > 0
                          else float(np.mean(1.0 - y_perm)))
            drops[c, r] = base_auprc - perm_auprc

    return drops.mean(axis=1)   # shape (n_channels,)
