#!/usr/bin/env python3
"""
deepmod_full_pipeline1 — three-model cross-dataset + LOMO evaluation.

One SLURM job per model (--model ont_only|umces_only|both). Each job:
  • trains a BASE model with a fixed genomic region held out of training
    (Mode A), then evaluates it on both datasets' held-out region test sets;
  • runs standard leave-one-modification-out (Mode B, results6-style): for each
    modification, hold it out of training entirely and test on all of its data.

All models share one architecture + hyperparameters (11-channel delta pileup,
BCE+pos_weight, batch 512, LR warmup, grad clip). Data is preloaded to RAM
(PileupDataset(preload=True)) to remove the gzip-per-item decompression
bottleneck (~25× faster).

Outputs into <out-dir> (default results1/):
  models/<model>/<run>/best_model.pt (+ training_curve.png, history.npz)
  metrics/<model>.tsv        (one row per evaluation)
  config.json                (written by --model ont_only, the first job)

Run collect.py after all three jobs to merge metrics and draw the 4 figures.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, precision_score, recall_score,
)

DEEPMOD = Path('/fs/nexus-scratch/bds062/Nanopore-Modification/deepmod')
sys.path.insert(0, str(DEEPMOD))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    PileupDataset, PileupInceptionV3, run_inference, aggregate_by_position,
    make_position_keys, source_position_keys, binary_labels,
    make_loader_kwargs, set_seed, SupConLoss,
)
from mod_types import build_umces_mod_map, UMCES_MODS

# Optional Weights & Biases logging. A no-op unless a caller has started a run
# (wandb.run is not None), so pipeline1 behaviour is unchanged when nothing
# initializes wandb. run_pipeline2.py opts in per fold.
try:
    import wandb as _wandb
except Exception:
    _wandb = None


def _wandb_log(payload: dict) -> None:
    if _wandb is not None and getattr(_wandb, 'run', None) is not None:
        try:
            _wandb.log(payload)
        except Exception:
            pass

# ── fixed data locations ──────────────────────────────────────────────────────
RESULTS9 = Path('/fs/nexus-scratch/bds062/results/deep_modification/results9')
ONT_FILES = {           # modification -> file  (control has no modification)
    '5mC':     str(RESULTS9 / '5mC.h5'),
    '5hmC':    str(RESULTS9 / '5hmC.h5'),
    '6mA':     str(RESULTS9 / '6mA.h5'),
    'control': str(RESULTS9 / 'control.h5'),
}
ONT_ORDER = ['5mC', '5hmC', '6mA', 'control']            # deterministic file order
ONT_LOMO_MODS = ['5mC', '5hmC', '6mA']

UMCES_ROOT   = Path('/fs/cbcb-scratch/bds062/results')
UMCES_FILES  = {         # role handled per-file in the split builders
    'bc06': str(UMCES_ROOT / 'deepmod_ont+umces/features/barcode06.h5'),
    'bc07': str(UMCES_ROOT / 'deepmod_ont+umces/features/barcode07.h5'),
    'bc02': str(UMCES_ROOT / 'deepmod_umces/features/train/barcode02.h5'),
    'bc03': str(UMCES_ROOT / 'deepmod_umces/features/train/barcode03.h5'),
    'bc04': str(UMCES_ROOT / 'deepmod_umces/features/train/barcode04.h5'),
    'bc05': str(UMCES_ROOT / 'deepmod_umces/features/train/barcode05.h5'),
    'bc01': str(UMCES_ROOT / 'deepmod_umces/features/test/barcode01_test.h5'),
}
UMCES_ORDER  = ['bc06', 'bc07', 'bc02', 'bc03', 'bc04', 'bc05', 'bc01']
UMCES_WGS    = {'bc06', 'bc07'}          # carry modifications; split by region
UMCES_REGION = (34597, 53675)            # FJ230960.1 barcode01 amplicon (test region)
UMCES_LOMO_MODS = ['5mC', '5hmC', '6mA', '5hmU']

UMCES_REF = '/fs/cbcb-lab/storm/shared/umbc-ont-data/ref/SPO1_FJ230960.1.fasta'
UMCES_PILEUPS = [
    str(UMCES_ROOT / 'deepmod_ont+umces/modbam/barcode06_pileup.bed'),
    str(UMCES_ROOT / 'deepmod_ont+umces/modbam/barcode07_pileup.bed'),
]

N_ONT_TEST_CONTIGS = 5


# ── hyperparameters (identical for every model) ───────────────────────────────
class HP:
    epochs        = 50
    batch         = 512
    lr            = 1.2e-3
    weight_decay  = 1e-3
    warmup_steps  = 500
    grad_clip     = 1.0
    dropout       = 0.4
    patience      = 15
    val_frac      = 0.15
    signal_noise  = 0.05
    # Streaming (preload=False) reads more per-item, so use more workers when set.
    # Override both from the environment (PILEUP_WORKERS, PILEUP_PRELOAD).
    num_workers   = int(os.environ.get('PILEUP_WORKERS', '4'))
    seed          = 42


# ── group = a file list forming one concatenated global index space ───────────
class Group:
    """Per-image metadata for a list of H5 files (tensors loaded lazily later)."""
    def __init__(self, names, files_map):
        self.names      = list(names)
        self.paths      = [files_map[n] for n in self.names]
        sizes, labels, keys, refpos, fileof = [], [], [], [], []
        for fi, (nm, p) in enumerate(zip(self.names, self.paths)):
            with h5py.File(p, 'r') as hf:
                n = hf['tensors'].shape[0]
                lab = binary_labels(hf['labels'][:])
                rn  = hf['ref_names'][:]
                rp  = hf['ref_pos'][:].astype(np.int64)
            sizes.append(n)
            labels.append(lab)
            keys.extend(make_position_keys(rn, rp))
            refpos.append(rp)
            fileof.append(np.full(n, fi, dtype=np.int64))
        self.file_sizes    = np.array(sizes, dtype=np.int64)
        self.labels        = np.concatenate(labels)
        self.position_keys = keys
        self.ref_pos       = np.concatenate(refpos)
        self.file_of       = np.concatenate(fileof)
        self.contig        = np.array([k[0] for k in keys])
        self.N             = int(self.file_sizes.sum())

    def name_index(self, name):
        return self.names.index(name)

    def source_keys(self, idx):
        keys, _ = source_position_keys(idx, self.position_keys, self.file_sizes)
        return keys


# ── metrics ───────────────────────────────────────────────────────────────────
def optimal_threshold(y_true, y_prob):
    """Threshold maximizing modified-class (label=1) F1; 0.5 if degenerate."""
    if int(y_true.sum()) == 0 or int(y_true.sum()) == len(y_true):
        return 0.5
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-8)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def compute_metrics(y_true, y_prob):
    y_true = y_true.astype(int)
    thr    = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    n_cls  = len(np.unique(y_true))
    return {
        'micro_f1':  float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'mod_f1':    float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'unmod_f1':  float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'macro_f1':  float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'mod_prec':  float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'mod_rec':   float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'auprc':     float(average_precision_score(y_true, y_prob)) if int(y_true.sum()) > 0
                     else float(np.mean(1.0 - y_prob)),
        'auroc':     float(roc_auc_score(y_true, y_prob)) if n_cls == 2 else float('nan'),
        'threshold': thr,
        'n_pos':     int(y_true.sum()),
        'n_test':    int(len(y_true)),
    }


# ── position-grouped validation carve (leakage-safe) ──────────────────────────
def carve_val(train_idx, group, val_frac, seed):
    """Hold out val_frac of position-groups (source-aware) from train_idx."""
    keys = group.source_keys(train_idx)
    by_key = {}
    for i, k in zip(train_idx, keys):
        by_key.setdefault(k, []).append(int(i))
    rng = np.random.default_rng(seed)
    ks = list(by_key.keys()); rng.shuffle(ks)
    n_val = max(1, int(round(len(ks) * val_frac)))
    val_keys = set(ks[:n_val])
    tr, va = [], []
    for k, idxs in by_key.items():
        (va if k in val_keys else tr).extend(idxs)
    return np.array(sorted(tr), dtype=np.int64), np.array(sorted(va), dtype=np.int64)


# ── training ──────────────────────────────────────────────────────────────────
def make_ds(group, idx, augment, hp):
    # preload=True loads the whole split into RAM (~390 GB for the pooled
    # pipeline2 set -> mem=460G). Set PILEUP_PRELOAD=0 to stream from disk
    # instead (mem ~a few GB -> qos=high); only fast once the tensors are
    # re-chunked to few-images/chunk (see scripts/rechunk_features.py).
    preload = os.environ.get('PILEUP_PRELOAD', '1') != '0'
    # PILEUP_MASK_FLANK=1 blanks base identity (ch 2-5) everywhere except the
    # candidate position, so a recognition motif such as GATC cannot be read off
    # the tensor. The candidate base itself is kept: it is legitimate evidence
    # (6mA only occurs at A, 5mC only at C).
    mask_flank = os.environ.get('PILEUP_MASK_FLANK', '0') == '1'
    # PILEUP_MASK_BASES=1 blanks ALL base-identity channels (is_A/C/G/T): the model
    # sees no nucleotide identity and must detect a modification from signal alone.
    mask_all = os.environ.get('PILEUP_MASK_BASES', '0') == '1'
    return PileupDataset(group.paths, idx, group.file_sizes,
                         augment=augment, seed=hp.seed,
                         signal_noise_std=hp.signal_noise,
                         delta_channels=True, preload=preload,
                         mask_flank_bases=mask_flank, mask_all_bases=mask_all)


def _sad_val_auroc(model, val_loader, device):
    """Deep-SAD anomaly-distance AUROC over a val loader (labels vs ||sad-centre||).
    Used for model selection when BCE_WEIGHT=0, since the classification head is
    untrained noise in that mode (mirrors run_svdd_loco.py's pure-SVDD selection)."""
    model.eval(); ds, ys = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            model(xb.to(device, non_blocking=True))
            d = ((model._sad - model.sad_center) ** 2).sum(1).sqrt().cpu().numpy()
            ds.append(d); ys.append(yb.numpy())
    model.train()
    d = np.concatenate(ds); y = np.concatenate(ys)
    return float(roc_auc_score(y, d)) if len(np.unique(y)) > 1 else float('nan')


def train_one_model(group, train_idx, hp, device, out_dir, tag, model_factory=None,
                    init_state=None):
    """init_state: optional state_dict to warm-start from (curriculum stage 2
    resumes from stage 1's weights). None = fresh init (default)."""
    print(f"\n{'='*66}\n  TRAIN [{tag}]  train_images={len(train_idx):,}", flush=True)
    sub_train, val_idx = carve_val(train_idx, group, hp.val_frac, hp.seed)
    train_labels = group.labels[sub_train]
    n_pos, n_neg = int(train_labels.sum()), len(train_labels) - int(train_labels.sum())
    pw = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
    print(f"  sub_train={len(sub_train):,}  val={len(val_idx):,}  "
          f"pos={n_pos:,} neg={n_neg:,}  pos_weight={pw.item():.2f}", flush=True)

    train_ds = make_ds(group, sub_train, True,  hp)
    val_ds   = make_ds(group, val_idx,   False, hp)
    val_keys = group.source_keys(val_idx)
    lk = make_loader_kwargs(hp.batch, hp.num_workers, device, _wif)
    train_loader = DataLoader(train_ds, shuffle=True,  **lk)
    val_loader   = DataLoader(val_ds,   shuffle=False, **lk)

    model = (model_factory() if model_factory is not None
             else PileupInceptionV3(in_channels=11, dropout=hp.dropout)).to(device)
    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
        print(f"  warm-started from prior stage ({len(init_state)} tensors)", flush=True)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    # BCE_WEIGHT=0 disables the classification loss entirely -- the model is then
    # trained ONLY by whichever of SupCon / Deep SAD are active (still label-driven,
    # just via a metric-learning objective instead of a cross-entropy decision
    # boundary). The classification head itself remains in the graph (untrained,
    # effectively random) -- its logit/AUPRC is meaningless when bce_w==0, so model
    # selection below switches to the Deep-SAD anomaly val-AUROC instead.
    bce_w = float(os.environ.get('BCE_WEIGHT', '1.0'))
    if bce_w != 1.0:
        print(f"  [BCE] weight={bce_w}" +
              ("  (classification loss DISABLED; representation trained by "
               "SupCon/SAD only)" if bce_w == 0 else ""), flush=True)

    # Supervised-contrastive auxiliary loss: active only when the model exposes a
    # projection head (ConvFormerV2(supcon_dim>0)). Weight/temperature are training
    # hyperparameters read from the environment so the architecture stays fixed.
    #   total_loss = BCE + adv_loss(DANN) + SUPCON_WEIGHT * SupCon(proj, y)
    supcon_on = getattr(model, 'proj', None) is not None
    supcon_w = float(os.environ.get('SUPCON_WEIGHT', '0.1')) if supcon_on else 0.0
    supcon_crit = SupConLoss(float(os.environ.get('SUPCON_TEMP', '0.07'))) if supcon_on else None
    if supcon_on:
        print(f"  [SupCon] proj_dim={model.supcon_dim}  weight={supcon_w}  "
              f"temp={os.environ.get('SUPCON_TEMP', '0.07')}", flush=True)

    # Deep SAD one-class loss (active when the model exposes a sad_head): normals
    # (unmodified) are pulled to a fixed centre c, labelled anomalies (the seen
    # modifications) are pushed away via inverse distance, so an UNSEEN chemistry at
    # test also lands far from c. Inference score = ||sad_head(rep) - c||.
    sad_on = getattr(model, 'sad_head', None) is not None
    sad_w = float(os.environ.get('SAD_WEIGHT', '1.0')) if sad_on else 0.0
    sad_eta = float(os.environ.get('SAD_ETA', '1.0'))
    if sad_on:
        model.eval(); acc = []; seen = 0
        with torch.no_grad():
            for xb, yb in train_loader:
                model(xb.to(device, non_blocking=True))
                m = (yb == 0)
                if m.any():
                    acc.append(model._sad[m.to(device)].detach().cpu())
                    seen += int(m.sum())
                if seen >= 8000:
                    break
        c = torch.cat(acc, 0).mean(0)
        c[c.abs() < 1e-6] = 1e-6              # no centre component exactly at 0
        model.sad_center.copy_(c.to(device))
        model.train()
        print(f"  [DeepSAD] sad_dim={model.sad_dim} weight={sad_w} eta={sad_eta} "
              f"centre from {seen:,} normals  ||c||={c.norm():.3f}", flush=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                       patience=7, min_lr=1e-6)
    warm  = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0 / max(hp.warmup_steps, 1), end_factor=1.0,
        total_iters=hp.warmup_steps) if hp.warmup_steps > 0 else None

    best_auprc, best_state, best_ep, patience = -1.0, None, 0, 0
    gstep = 0
    tr_hist, va_hist, ap_hist = [], [], []
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, hp.epochs + 1):
        model.train(); t0 = time.time(); eloss = eseen = eadv = esup = esad = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            # An auxiliary-enabled model returns (logit, aux) in train mode, where
            # aux is a dict with keys among {'adv_loss', 'proj'}:
            #   'adv_loss' — DANN adversary CE; the gradient-reversal layer already
            #                carries the sign, so it is simply added.
            #   'proj'     — L2-normalised SupCon embedding; the contrastive loss is
            #                computed here because it needs the batch labels y.
            if isinstance(out, tuple):
                logit, aux = out
                loss = bce_w * crit(logit.squeeze(1), y)
                if 'adv_loss' in aux:
                    loss = loss + aux['adv_loss']
                    eadv += float(aux['adv_loss']) * len(y)
                if 'proj' in aux and supcon_w > 0:
                    sc = supcon_crit(aux['proj'], y.long())
                    loss = loss + supcon_w * sc
                    esup += float(sc) * len(y)
                if 'sad' in aux and sad_w > 0:
                    d2 = ((aux['sad'] - model.sad_center) ** 2).sum(1)
                    nmask = (y == 0); amask = (y == 1)
                    ln = d2[nmask].mean() if nmask.any() else d2.new_zeros(())
                    la = (1.0 / (d2[amask] + 1e-6)).mean() if amask.any() else d2.new_zeros(())
                    sadl = ln + sad_eta * la
                    loss = loss + sad_w * sadl
                    esad += float(sadl) * len(y)
            else:
                loss = bce_w * crit(out.squeeze(1), y)
            loss.backward()
            if hp.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step(); gstep += 1
            if warm is not None and gstep <= hp.warmup_steps:
                warm.step()
            eloss += loss.item() * len(y); eseen += len(y)
        tr_loss = eloss / max(eseen, 1)

        if bce_w == 0 and sad_on:
            # classification head is untrained noise -- select on Deep-SAD anomaly
            # val-AUROC instead (same trick as run_svdd_loco.py's pure-SVDD selection).
            vauprc = _sad_val_auroc(model, val_loader, device)
        else:
            yt, yp = run_inference(model, val_loader, device)
            vt, vp, _ = aggregate_by_position(yt, yp, val_keys)
            vauprc = (average_precision_score(vt, vp) if int(vt.sum()) > 0
                      else float(np.mean(1.0 - vp)))
        if warm is None or gstep > hp.warmup_steps:
            sched.step(vauprc)
        tr_hist.append(tr_loss); ap_hist.append(vauprc)
        va_hist.append(vauprc)
        adv_str = f"adv_ce={eadv/max(eseen,1):.4f}  " if eadv else ""
        sup_str = f"supcon={esup/max(eseen,1):.4f}  " if esup else ""
        sad_str = f"sad={esad/max(eseen,1):.4f}  " if esad else ""
        sel_label = "val_anomAUROC" if (bce_w == 0 and sad_on) else "val_AUPRC"
        print(f"  ep {ep:3d}/{hp.epochs}  tr_loss={tr_loss:.4f}  {adv_str}{sup_str}{sad_str}"
              f"{sel_label}={vauprc:.4f}  lr={opt.param_groups[0]['lr']:.2e}  "
              f"{time.time()-t0:.1f}s", flush=True)
        _wandb_log({f'{tag}/train_loss': tr_loss, f'{tag}/val_auprc': vauprc,
                    f'{tag}/supcon': esup / max(eseen, 1),
                    f'{tag}/lr': opt.param_groups[0]['lr'], f'{tag}/epoch': ep})

        if vauprc > best_auprc:
            best_auprc, best_ep, patience = vauprc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= hp.patience:
                print(f"  early stop @ ep {ep}", flush=True); break

    _wandb_log({f'{tag}/best_val_auprc': best_auprc, f'{tag}/best_epoch': best_ep})
    model.load_state_dict(best_state)
    torch.save({'model_state': best_state, 'in_channels': 11,
                'val_auprc': best_auprc, 'epoch': best_ep, 'tag': tag},
               run_dir / 'best_model.pt')
    np.savez(run_dir / 'history.npz', train_loss=tr_hist, val_auprc=ap_hist, best_epoch=best_ep)
    _plot_curve(tr_hist, ap_hist, best_ep, run_dir / 'training_curve.png', tag)
    print(f"  best val_AUPRC={best_auprc:.4f} @ ep {best_ep}  -> {run_dir/'best_model.pt'}",
          flush=True)
    return model


def evaluate(model, group, test_idx, device, hp):
    ds = make_ds(group, test_idx, False, hp)
    loader = DataLoader(ds, shuffle=False, **make_loader_kwargs(hp.batch, hp.num_workers, device, _wif))
    yt, yp = run_inference(model, loader, device)
    keys = group.source_keys(test_idx)
    t, p, _ = aggregate_by_position(yt, yp, keys)
    return compute_metrics(t, p)


def _wif(_):
    import model as _m
    _m._H5_HANDLES = {}


def _plot_curve(tr, ap, best, path, tag):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(range(1, len(tr)+1), tr, 'C0-', label='train loss'); ax1.set_ylabel('train loss', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(ap)+1), ap, 'C1-', label='val AUPRC'); ax2.set_ylabel('val AUPRC', color='C1')
    ax1.axvline(best, color='gray', ls='--', lw=0.7)
    ax1.set_xlabel('epoch'); ax1.set_title(f'{tag}  (best ep {best})')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ── split builders ────────────────────────────────────────────────────────────
def pick_ont_test_contigs(group, n, seed):
    contigs = sorted(set(group.contig.tolist()))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(contigs, size=min(n, len(contigs)), replace=False).tolist())


def ont_region_split(group, test_contigs):
    in_test = np.isin(group.contig, np.array(test_contigs))
    return np.nonzero(~in_test)[0].astype(np.int64), np.nonzero(in_test)[0].astype(np.int64)


def umces_in_region(group):
    """Boolean per-image: is this position in the held-out UMCES test region?"""
    lo, hi = UMCES_REGION
    inreg = np.zeros(group.N, dtype=bool)
    for nm in group.names:
        fi = group.name_index(nm)
        sel = group.file_of == fi
        if nm in UMCES_WGS:
            inreg |= sel & (group.ref_pos >= lo) & (group.ref_pos < hi)
        elif nm == 'bc01':
            inreg |= sel                      # bc01 amplicon == the test region
        # bc02-05: never in region
    return inreg


def umces_region_split(group):
    inreg = umces_in_region(group)
    return np.nonzero(~inreg)[0].astype(np.int64), np.nonzero(inreg)[0].astype(np.int64)


def umces_type_array(group, mod_map):
    """Per-image modification-type string ('' for unmodified/untyped)."""
    types = np.array([''] * group.N, dtype=object)
    modified = group.labels > 0
    for i in np.nonzero(modified)[0]:
        types[i] = mod_map.get((group.contig[i], int(group.ref_pos[i])), 'untyped')
    return types


def umces_lomo_split(group, mod_map, mod):
    """
    LOMO for UMCES modification `mod` (results6-style, region-clean negatives):
      test  = {modified & type==mod}  ∪  {unmodified & in held-out region}
      train = {modified & type!=mod}  ∪  {unmodified & outside region}
    """
    types = umces_type_array(group, mod_map)
    modified = group.labels > 0
    inreg = umces_in_region(group)
    is_mod_M = modified & (types == mod)
    test  = np.nonzero(is_mod_M | (~modified & inreg))[0].astype(np.int64)
    train = np.nonzero((modified & (types != mod)) | (~modified & ~inreg))[0].astype(np.int64)
    return train, test


# ── leakage guard ─────────────────────────────────────────────────────────────
def assert_disjoint(train_idx, test_idx, group, label):
    if np.intersect1d(train_idx, test_idx).size:
        raise SystemExit(f"LEAKAGE[{label}]: train/test image indices overlap")
    tr_keys = set(group.source_keys(train_idx)); te_keys = set(group.source_keys(test_idx))
    inter = tr_keys & te_keys
    if inter:
        raise SystemExit(f"LEAKAGE[{label}]: {len(inter)} (file,ref,pos) keys in both train and test")
    print(f"  leakage-guard OK [{label}]  train={len(train_idx):,} test={len(test_idx):,}", flush=True)


# ── reusable, model-agnostic experiment driver ───────────────────────────────
# model_factory: zero-arg callable returning an nn.Module (input (B,11,H,W) → logit).
# None uses the default PileupInceptionV3.  This is the single hook that lets each
# architecture live in its own thin driver file (run_mlp.py, etc.) while sharing
# the identical splits / LOMO / evaluation / metrics below.
def run_experiment(cli_model, out_dir, model_factory=None, epochs_override=None):
    hp = HP()
    if epochs_override:
        hp.epochs = epochs_override
    set_seed(hp.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(out_dir)
    (out / 'models').mkdir(parents=True, exist_ok=True)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}   model={cli_model}   out={out}", flush=True)

    # ── build groups + fixed test sets ────────────────────────────────────────
    ont = Group(ONT_ORDER, ONT_FILES)
    umc = Group(UMCES_ORDER, UMCES_FILES)
    if ont.file_sizes[0] and (ont.N, umc.N):
        pass
    # shape/attr consistency preflight
    shapes = set()
    for p in ont.paths + umc.paths:
        with h5py.File(p, 'r') as hf:
            shapes.add(tuple(hf['tensors'].shape[1:]))
    if len(shapes) != 1:
        raise SystemExit(f"PREFLIGHT: inconsistent tensor shapes across H5s: {shapes}")
    print(f"Preflight OK — tensor shape {shapes.pop()} across {len(ont.paths)+len(umc.paths)} files", flush=True)

    test_contigs = pick_ont_test_contigs(ont, N_ONT_TEST_CONTIGS, hp.seed)
    ont_train, ont_test = ont_region_split(ont, test_contigs)
    umc_train, umc_test = umces_region_split(umc)
    assert_disjoint(ont_train, ont_test, ont, 'ONT-region')
    assert_disjoint(umc_train, umc_test, umc, 'UMCES-region')

    mod_map = build_umces_mod_map(UMCES_PILEUPS, UMCES_REF)

    # write config once (from the ont_only job)
    if cli_model == 'ont_only':
        json.dump({
            'ont_test_contigs': test_contigs,
            'umces_test_region': list(UMCES_REGION),
            'ont_lomo_mods': ONT_LOMO_MODS, 'umces_lomo_mods': UMCES_LOMO_MODS,
            'hyperparams': {k: getattr(HP, k) for k in dir(HP) if not k.startswith('_')},
            'ont_train_n': int(len(ont_train)), 'ont_test_n': int(len(ont_test)),
            'umces_train_n': int(len(umc_train)), 'umces_test_n': int(len(umc_test)),
        }, open(out / 'config.json', 'w'), indent=2)
        print(f"Wrote {out/'config.json'}", flush=True)

    rows = []   # metric rows for this model

    def run_eval(model, run, eval_kind, held_out=''):
        for tname, g, tidx in [('ONT_test', ont, ont_test), ('UMCES_test', umc, umc_test)]:
            if eval_kind == 'lomo':
                continue  # base-only; LOMO evals its own held-out set below
            m = evaluate(model, g, tidx, device, hp)
            rows.append({'model': cli_model, 'run': run, 'eval_kind': eval_kind,
                         'test_set': tname, 'held_out': held_out, **m})
            print(f"  EVAL {run} on {tname}: micro_f1={m['micro_f1']:.3f} "
                  f"mod_f1={m['mod_f1']:.3f} auprc={m['auprc']:.3f}", flush=True)

    # ── BASE model (Mode A) ───────────────────────────────────────────────────
    if cli_model == 'ont_only':
        base = train_one_model(ont, ont_train, hp, device, out / 'models' / cli_model, 'base',
                               model_factory=model_factory)
    elif cli_model == 'umces_only':
        base = train_one_model(umc, umc_train, hp, device, out / 'models' / cli_model, 'base',
                               model_factory=model_factory)
    else:  # both — combined group
        both = Group(ONT_ORDER + UMCES_ORDER, {**ONT_FILES, **UMCES_FILES})
        off = int(ont.N)                       # UMCES indices shift by #ONT images
        both_train = np.concatenate([ont_train, umc_train + off]).astype(np.int64)
        base = train_one_model(both, both_train, hp, device, out / 'models' / cli_model, 'base',
                               model_factory=model_factory)
    run_eval(base, 'base', 'base')
    del base
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    # ── LOMO (Mode B, standard results6-style) ────────────────────────────────
    def lomo_eval_and_record(model, mod, test_group, test_idx):
        m = evaluate(model, test_group, test_idx, device, hp)
        rows.append({'model': cli_model, 'run': f'lomo_{mod}', 'eval_kind': 'lomo',
                     'test_set': f'heldout_{mod}', 'held_out': mod, **m})
        print(f"  LOMO[{mod}]: micro_f1={m['micro_f1']:.3f} mod_f1={m['mod_f1']:.3f} "
              f"auprc={m['auprc']:.3f} n_pos={m['n_pos']}", flush=True)

    if cli_model == 'ont_only':
        for mod in ONT_LOMO_MODS:
            train_names = [n for n in ONT_ORDER if n != mod]        # drop mod file, keep control
            g_tr = Group(train_names, ONT_FILES)
            g_te = Group([mod], ONT_FILES)
            model = train_one_model(g_tr, np.arange(g_tr.N, dtype=np.int64), hp, device,
                                    out / 'models' / cli_model, f'lomo_{mod}',
                                    model_factory=model_factory)
            lomo_eval_and_record(model, mod, g_te, np.arange(g_te.N, dtype=np.int64))
            del model; torch.cuda.empty_cache() if device.type == 'cuda' else None

    elif cli_model == 'umces_only':
        for mod in UMCES_LOMO_MODS:
            tr, te = umces_lomo_split(umc, mod_map, mod)
            assert_disjoint(tr, te, umc, f'UMCES-LOMO-{mod}')
            model = train_one_model(umc, tr, hp, device, out / 'models' / cli_model, f'lomo_{mod}',
                                    model_factory=model_factory)
            lomo_eval_and_record(model, mod, umc, te)
            del model; torch.cuda.empty_cache() if device.type == 'cuda' else None

    else:  # both — union of ONT-file-drop and UMCES-type-drop
        for mod in UMCES_LOMO_MODS:                   # 5mC,5hmC,6mA,5hmU
            umc_tr, umc_te = umces_lomo_split(umc, mod_map, mod)
            assert_disjoint(umc_tr, umc_te, umc, f'both-UMCES-LOMO-{mod}')
            ont_train_names = [n for n in ONT_ORDER if n != mod]   # 5hmU: keeps all ONT
            g_tr = Group(ont_train_names + UMCES_ORDER, {**ONT_FILES, **UMCES_FILES})
            off = sum(Group([n], ONT_FILES).N for n in ont_train_names)
            tr = np.concatenate([np.arange(off, dtype=np.int64), umc_tr + off]).astype(np.int64)
            model = train_one_model(g_tr, tr, hp, device, out / 'models' / cli_model, f'lomo_{mod}',
                                    model_factory=model_factory)
            # test on ONT held-out file (if exists) + UMCES held-out type — record separately
            if mod in ONT_FILES and mod != 'control':
                g_te_ont = Group([mod], ONT_FILES)
                m = evaluate(model, g_te_ont, np.arange(g_te_ont.N, dtype=np.int64), device, hp)
                rows.append({'model': cli_model, 'run': f'lomo_{mod}', 'eval_kind': 'lomo',
                             'test_set': f'ONT_heldout_{mod}', 'held_out': mod, **m})
                print(f"  LOMO[{mod}] ONT: micro_f1={m['micro_f1']:.3f} mod_f1={m['mod_f1']:.3f}", flush=True)
            m2 = evaluate(model, umc, umc_te, device, hp)
            rows.append({'model': cli_model, 'run': f'lomo_{mod}', 'eval_kind': 'lomo',
                         'test_set': f'UMCES_heldout_{mod}', 'held_out': mod, **m2})
            print(f"  LOMO[{mod}] UMCES: micro_f1={m2['micro_f1']:.3f} mod_f1={m2['mod_f1']:.3f}", flush=True)
            del model; torch.cuda.empty_cache() if device.type == 'cuda' else None

    # ── write per-model metrics tsv ───────────────────────────────────────────
    cols = ['model', 'run', 'eval_kind', 'test_set', 'held_out', 'micro_f1', 'mod_f1',
            'unmod_f1', 'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc',
            'threshold', 'n_pos', 'n_test']
    tsv = out / 'metrics' / f'{cli_model}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(
                f"{r[c]:.6f}" if isinstance(r.get(c), float) else str(r.get(c, ''))
                for c in cols) + '\n')
    print(f"\nWrote {tsv}  ({len(rows)} rows)\nDONE [{cli_model}]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=['ont_only', 'umces_only', 'both'])
    ap.add_argument('--out-dir', default=str(Path(__file__).resolve().parent / 'results1'))
    ap.add_argument('--epochs', type=int, default=None, help='override epochs (debug)')
    args = ap.parse_args()
    # default model_factory=None → PileupInceptionV3 (results1)
    run_experiment(args.model, args.out_dir, model_factory=None, epochs_override=args.epochs)


if __name__ == '__main__':
    main()
