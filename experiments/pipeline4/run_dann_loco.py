#!/usr/bin/env python3
"""
rawmod DANN two-head experiment (results7 / results8).

Same matched pool, chemistry typing, and curriculum as run_matched_loco.py
(imported and reused wholesale via `import run_matched_loco as ML`) -- ONLY the
model and loss differ. NO BCE head, NO SupCon, NO Deep SAD: exactly two
NLLLoss heads hang off the shared 96-d penultimate rep (ConvFormerV2DANN in
run_convformer_v2.py):

  presence_head (2-way): predicts modification PRESENCE (mod vs unmod) --
      the detector. Normal (non-reversed) gradient.
  adv_head (K-way): a gradient-reversed adversary predicting either
      --adv-target type    : modification TYPE, 5-way over the chemistries
                              {5hmU,4mC,6mA,5mC,5hmC}. Unmodified images are
                              EXCLUDED from this loss (they have no type).
      --adv-target dataset  : source DATASET/organism, 3-way {ONT,SPO1,HP}.
                              Every image has one, so nothing is excluded.
  Gradient reversal pushes the shared backbone to make `rep` UNINFORMATIVE
  about the adversary's target, while presence_head still needs `rep` to stay
  informative about mod/unmod. Purpose: test whether this directly attacks
  the organism-dominated embedding clustering found in prior experiments
  (see memory: embedding-not-chemistry-space, results6-bce-supcon-tradeoff).

Usage:
  python run_dann_loco.py --fold {mixed|loco_<CHEM>} --adv-target {type|dataset} \
      --out-dir <dir> [--epochs N]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / 'experiments' / 'pipeline1', REPO / 'deepmod'):
    sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline as R                            # noqa: E402
import run_matched_loco as ML                        # noqa: E402
from run_convformer_v2 import ConvFormerV2DANN        # noqa: E402
from model import make_loader_kwargs, _worker_init_fn, aggregate_by_position  # noqa: E402

WANDB_ENTITY = os.environ.get('WANDB_ENTITY', 'bds062-university-of-maryland')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'rawmod')

TYPE_CLASSES = list(ML.CHEMS)          # ('5hmU','4mC','6mA','5mC','5hmC') -- fixed class order
ORG_CLASSES = ['ONT', 'SPO1', 'HP']


class DANNDataset(Dataset):
    """Wraps a base PileupDataset, additionally returning the row's adversary
    target class id and a validity flag (default_collate handles the extra
    int/bool fields automatically -- no custom collate_fn needed)."""
    def __init__(self, base_ds, adv_y, adv_valid):
        self.base = base_ds
        self.adv_y = adv_y
        self.adv_valid = adv_valid

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        x, y = self.base[item]
        return x, y, int(self.adv_y[item]), bool(self.adv_valid[item])


def chem_to_org(pool):
    """Per-image organism id (0=ONT,1=SPO1,2=HP), for every image in pool."""
    member_of = np.array([pool.names[int(pool.file_of[i])] for i in range(pool.N)])
    org_str = np.array([ML.org_of(m).rstrip(':') for m in member_of])
    return np.array([ORG_CLASSES.index(o) for o in org_str], dtype=np.int64)


def adv_labels_for(idx, adv_target, chem, org_id):
    """(y, valid) integer adversary target + validity mask for global indices idx."""
    if adv_target == 'type':
        y = np.full(len(idx), -1, dtype=np.int64)
        valid = np.zeros(len(idx), dtype=bool)
        for j, i in enumerate(idx):
            c = chem[i]
            if c in TYPE_CLASSES:
                y[j] = TYPE_CLASSES.index(c)
                valid[j] = True
        return y, valid
    else:  # dataset
        y = org_id[idx].astype(np.int64)
        valid = np.ones(len(idx), dtype=bool)
        return y, valid


def class_weights(y_int, n_classes, device):
    """Balanced per-class weight: N / (n_classes * count_c); 0 for absent classes
    (guards div-by-zero for a LOCO fold's held-out chemistry, which has zero
    training examples for that class slot)."""
    counts = np.bincount(y_int, minlength=n_classes).astype(np.float64)
    N = counts.sum()
    w = np.zeros(n_classes, dtype=np.float32)
    nz = counts > 0
    if N > 0:
        w[nz] = N / (n_classes * counts[nz])
    else:
        w[:] = 1.0
    return torch.tensor(w, dtype=torch.float32, device=device)


def presence_probs(model, loader, device):
    """Softmax P(mod) from the 2-way presence head; model-agnostic downstream
    (aggregate_by_position/compute_metrics only need y_true/y_prob arrays)."""
    model.eval(); all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            logits = model(x.to(device, non_blocking=True))
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_y.append(y.numpy()); all_p.append(p)
    return np.concatenate(all_y), np.concatenate(all_p)


def eval_both_heads(model, loader, device):
    """One eval-mode forward pass; returns presence probs + adv predictions +
    the captured 96-d rep for every item. adv_head is applied directly to the
    captured rep (grad_reverse is IDENTITY on the forward pass -- the sign
    flip only happens in backward -- so this exactly matches what adv_head saw
    during training, without needing training mode)."""
    cap = {}
    hook = model.presence_head.register_forward_hook(
        lambda _m, inp, _o: cap.__setitem__('rep', inp[0].detach()))
    model.eval()
    all_y, all_p, all_advy, all_advvalid, all_advpred = [], [], [], [], []
    with torch.no_grad():
        for x, y, advy, advvalid in loader:
            logits = model(x.to(device, non_blocking=True))
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            adv_logits = model.adv_head(cap['rep'])
            adv_pred = adv_logits.argmax(dim=1).cpu().numpy()
            all_y.append(y.numpy()); all_p.append(p)
            all_advy.append(advy.numpy()); all_advvalid.append(advvalid.numpy())
            all_advpred.append(adv_pred)
    hook.remove()
    return (np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_advy),
           np.concatenate(all_advvalid).astype(bool), np.concatenate(all_advpred))


def adv_lambda_schedule(step, total_steps, lambda_max, gamma=10.0):
    """Ganin & Lempitsky (2016) ramp-up: lambda_p = lambda_max * (2/(1+exp(-gamma*p)) - 1),
    p = training progress in [0, 1]. Starts at 0, reaches ~87% of lambda_max by p=0.3,
    ~99% by p=0.6. Fixes the results7/8 instability (full-strength reversed gradient from
    step 1 destabilized training; see memory: results7-8-dann-backfire)."""
    p = min(step / max(total_steps - 1, 1), 1.0)
    return float(lambda_max * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))


def fit_one(pool, train_idx, hp, device, out_dir, tag, adv_target, n_adv,
           adv_lambda, chem, org_id, init_state=None, adv_lambda_gamma=10.0):
    print(f"\n{'='*66}\n  TRAIN [{tag}]  train_images={len(train_idx):,}", flush=True)
    sub_train, val_idx = R.carve_val(train_idx, pool, hp.val_frac, hp.seed)

    adv_y_tr, adv_valid_tr = adv_labels_for(sub_train, adv_target, chem, org_id)
    adv_y_va, adv_valid_va = adv_labels_for(val_idx, adv_target, chem, org_id)
    train_ds = DANNDataset(R.make_ds(pool, sub_train, True, hp), adv_y_tr, adv_valid_tr)
    val_ds = DANNDataset(R.make_ds(pool, val_idx, False, hp), adv_y_va, adv_valid_va)
    val_keys = pool.source_keys(val_idx)
    lk = make_loader_kwargs(hp.batch, hp.num_workers, device, _worker_init_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **lk)
    val_loader = DataLoader(val_ds, shuffle=False, **lk)

    model = ConvFormerV2DANN(dropout=hp.dropout, n_adv_classes=n_adv,
                             adv_lambda=adv_lambda, h=ML.HEIGHT).to(device)
    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
        print(f"  warm-started from prior stage ({len(init_state)} tensors)", flush=True)

    y_pres_tr = (pool.labels[sub_train] > 0).astype(np.int64)
    pres_w = class_weights(y_pres_tr, 2, device)
    adv_w = class_weights(adv_y_tr[adv_valid_tr], n_adv, device) if adv_valid_tr.any() \
        else torch.ones(n_adv, device=device)
    print(f"  presence class weights: {pres_w.tolist()}", flush=True)
    print(f"  [{adv_target}] adv class weights: {adv_w.tolist()}  "
          f"lambda={adv_lambda}  (n_valid={int(adv_valid_tr.sum()):,}/{len(sub_train):,})",
          flush=True)

    total_steps = hp.epochs * max(len(train_loader), 1)
    print(f"  adv_lambda ramp: 0 -> {adv_lambda} over {total_steps:,} steps "
          f"(gamma={adv_lambda_gamma})", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                       patience=7, min_lr=1e-6)
    warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0 / max(hp.warmup_steps, 1), end_factor=1.0,
        total_iters=hp.warmup_steps) if hp.warmup_steps > 0 else None

    best_auprc, best_state, best_ep, patience = -1.0, None, 0, 0
    gstep = 0
    tr_hist, va_hist = [], []
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, hp.epochs + 1):
        model.train()
        eloss = epres = eadv = eseen = 0
        for x, y, advy, advvalid in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            advy = advy.to(device, non_blocking=True)
            advvalid = advvalid.to(device, non_blocking=True)

            model.adv_lambda = adv_lambda_schedule(gstep, total_steps, adv_lambda, adv_lambda_gamma)
            opt.zero_grad()
            presence_logits, adv_logits = model(x)
            pres_loss = F.nll_loss(F.log_softmax(presence_logits, dim=1), y, weight=pres_w)
            loss = pres_loss
            if advvalid.any():
                adv_loss = F.nll_loss(F.log_softmax(adv_logits[advvalid], dim=1),
                                      advy[advvalid], weight=adv_w)
                loss = loss + adv_loss
                eadv += float(adv_loss) * int(advvalid.sum())
            loss.backward()
            if hp.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step(); gstep += 1
            if warm is not None and gstep <= hp.warmup_steps:
                warm.step()
            eloss += float(loss) * len(y); epres += float(pres_loss) * len(y); eseen += len(y)
        tr_loss = eloss / max(eseen, 1)

        yt, yp = presence_probs(model, val_loader, device)
        vt, vp, _ = aggregate_by_position(yt, yp, val_keys)
        from sklearn.metrics import average_precision_score
        vauprc = (average_precision_score(vt, vp) if int(vt.sum()) > 0
                  else float(np.mean(1.0 - vp)))
        if warm is None or gstep > hp.warmup_steps:
            sched.step(vauprc)
        tr_hist.append(tr_loss); va_hist.append(vauprc)
        print(f"  ep {ep:3d}/{hp.epochs}  tr_loss={tr_loss:.4f}  "
              f"presence={epres/max(eseen,1):.4f}  adv={eadv/max(eseen,1):.4f}  "
              f"lambda={model.adv_lambda:.3f}  "
              f"val_AUPRC={vauprc:.4f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)

        if vauprc > best_auprc:
            best_auprc, best_ep, patience = vauprc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= hp.patience:
                print(f"  early stop @ ep {ep}", flush=True); break

    model.load_state_dict(best_state)
    torch.save({'model_state': best_state, 'in_channels': 11, 'val_auprc': best_auprc,
               'epoch': best_ep, 'tag': tag, 'n_adv_classes': n_adv,
               'adv_target': adv_target, 'adv_lambda': adv_lambda,
               'adv_lambda_schedule': 'ganin_sigmoid', 'adv_lambda_gamma': adv_lambda_gamma},
              run_dir / 'best_model.pt')
    np.savez(run_dir / 'history.npz', train_loss=tr_hist, val_auprc=va_hist, best_epoch=best_ep)
    print(f"  best val_AUPRC={best_auprc:.4f} @ ep {best_ep}  -> {run_dir/'best_model.pt'}",
          flush=True)
    return model


def record(model, pool, idx, adv_target, device, hp, rows, fold, test_name, held=''):
    ds = DANNDataset(R.make_ds(pool, idx, False, hp), *adv_labels_for(
        idx, adv_target, record.chem, record.org_id))
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(hp.batch, hp.num_workers, device, _worker_init_fn))
    yt, yp, advy, advvalid, advpred = eval_both_heads(model, loader, device)
    keys = pool.source_keys(idx)
    t, p, _ = aggregate_by_position(yt, yp, keys)
    m = R.compute_metrics(t, p)
    if advvalid.any():
        m['adv_acc'] = float((advpred[advvalid] == advy[advvalid]).mean())
    else:
        m['adv_acc'] = float('nan')
    rows.append({'fold': fold, 'test_set': test_name, 'held_out': held, **m})
    print(f"  EVAL {test_name}: mod_f1={m['mod_f1']:.3f} mod_rec={m['mod_rec']:.3f} "
          f"mod_prec={m['mod_prec']:.3f} auprc={m['auprc']:.3f} auroc={m['auroc']:.3f} "
          f"adv_acc={m['adv_acc']:.3f} n_pos={m['n_pos']} n_test={m['n_test']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', required=True,
                    help="'mixed' or 'loco_<CHEM>' with CHEM in " + '/'.join(ML.CHEMS))
    ap.add_argument('--adv-target', required=True, choices=['type', 'dataset'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--epochs', type=int, default=None)
    a = ap.parse_args()

    valid = ['mixed'] + [f'loco_{c}' for c in ML.CHEMS]
    if a.fold not in valid:
        raise SystemExit(f"--fold must be one of {valid}, got {a.fold!r}")

    hp = R.HP()
    if a.epochs:
        hp.epochs = a.epochs
    R.set_seed(hp.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out_dir)
    (out / 'models').mkdir(parents=True, exist_ok=True)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    print(f"Device {device}  fold={a.fold}  adv_target={a.adv_target}  out={out}", flush=True)

    members = ML.build_members()
    pool = R.Group(list(members), members)
    print(f"Matched pool: {pool.N:,} images across {len(members)} files", flush=True)

    mod_map = ML.build_umces_mod_map(R.UMCES_PILEUPS, R.UMCES_REF)
    refbase = ML.ref_base_center(pool)
    chem = ML.chem_array(pool, mod_map, refbase)
    org_id = chem_to_org(pool)
    is_pos = pool.labels > 0
    kept_neg = ML.subsample_negatives(pool, seed=hp.seed)
    neg_mask = np.zeros(pool.N, dtype=bool); neg_mask[kept_neg] = True
    record.chem, record.org_id = chem, org_id     # stash for record()

    n_adv = len(TYPE_CLASSES) if a.adv_target == 'type' else len(ORG_CLASSES)
    adv_lambda = float(os.environ.get('ADV_LAMBDA', '1.0'))
    adv_lambda_gamma = float(os.environ.get('ADV_LAMBDA_GAMMA', '10.0'))

    from collections import Counter
    print("  positives per chemistry:",
          {k: int(v) for k, v in Counter(chem[is_pos]).items()}, flush=True)
    print(f"  controls kept (capped): {int(neg_mask.sum()):,} of {int((~is_pos).sum()):,}",
          flush=True)

    wandb_run = None
    if os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'):
        os.environ.setdefault('WANDB_MODE', 'online')
        os.environ.setdefault('WANDB_DIR', str(Path(__file__).resolve().parent))
        try:
            import wandb, secrets
            rid = secrets.token_hex(4)
            wandb_run = wandb.init(
                entity=WANDB_ENTITY, project=WANDB_PROJECT,
                name=f"dann-{a.adv_target}-{rid}-{a.fold}", group=f'dann_{a.adv_target}',
                job_type=a.fold.split('_')[0],
                config={'fold': a.fold, 'architecture': 'ConvFormerV2DANN',
                        'adv_target': a.adv_target, 'n_adv_classes': n_adv,
                        'adv_lambda': adv_lambda,
                        **{k: getattr(hp, k) for k in dir(hp) if not k.startswith('_')}},
                settings=wandb.Settings(init_timeout=180, start_method='thread'))
            print(f"  [wandb] {WANDB_ENTITY}/{WANDB_PROJECT}/dann-{a.adv_target}-{rid}-{a.fold}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  [wandb] disabled ({e})", file=sys.stderr)

    curriculum = os.environ.get('CURRICULUM', '0') == '1'
    cur_epochs = int(os.environ.get('CURRICULUM_EPOCHS', '15'))

    def fit(train_idx, fold_dir, runtag):
        mdir = out / 'models' / fold_dir
        if curriculum:
            anc = ML.anchor_idx_within(pool, train_idx, is_pos)
            npos = int(is_pos[anc].sum()) if len(anc) else 0
            print(f"  [curriculum] stage1 anchors={len(anc):,} (pos={npos:,} "
                  f"neg={len(anc)-npos:,})  epochs={cur_epochs}", flush=True)
            if len(anc) >= 500 and npos > 0 and (len(anc) - npos) > 0:
                hp1 = R.HP(); hp1.epochs = cur_epochs; hp1.patience = cur_epochs
                m1 = fit_one(pool, anc, hp1, device, mdir, runtag + '_cur1', a.adv_target,
                            n_adv, adv_lambda, chem, org_id, adv_lambda_gamma=adv_lambda_gamma)
                state = {k: v.detach().cpu().clone() for k, v in m1.state_dict().items()}
                del m1
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                return fit_one(pool, train_idx, hp, device, mdir, runtag, a.adv_target,
                              n_adv, adv_lambda, chem, org_id, init_state=state,
                              adv_lambda_gamma=adv_lambda_gamma)
            print("  [curriculum] too few paired anchors; single-stage fallback", flush=True)
        return fit_one(pool, train_idx, hp, device, mdir, runtag, a.adv_target,
                       n_adv, adv_lambda, chem, org_id, adv_lambda_gamma=adv_lambda_gamma)

    rows = []
    if a.fold == 'mixed':
        train_idx, test_idx, stats = ML.mixed_split(pool, is_pos, neg_mask, hp)
        print(f"  {stats}", flush=True)
        model = fit(train_idx, a.fold, 'mixed')
        record(model, pool, test_idx, a.adv_target, device, hp, rows, a.fold, 'held_out_test')
    else:
        chem_x = a.fold[len('loco_'):]
        ctrl_idx = np.nonzero(neg_mask)[0]
        tr_ctrl, te_ctrl = ML.pos_hash_split(pool, ctrl_idx, test_frac=0.15, seed=hp.seed)
        bases = ML.CHEM_BASES[chem_x]; orgs = ML.CHEM_ORGS[chem_x]
        te_ctrl = np.array([i for i in te_ctrl
                            if ML.org_of(pool.names[int(pool.file_of[i])]) in orgs
                            and refbase[i] in bases], dtype=np.int64)
        pos_x = np.nonzero(is_pos & (chem == chem_x))[0].astype(np.int64)
        pos_other = np.nonzero(is_pos & (chem != chem_x) & (chem != '')
                               & (chem != 'untyped'))[0].astype(np.int64)
        train_idx = np.sort(np.concatenate([pos_other, tr_ctrl]))
        test_idx = np.sort(np.concatenate([pos_x, te_ctrl]))
        R.assert_disjoint(train_idx, test_idx, pool, a.fold)
        print(f"  train={len(train_idx):,} (pos_other={len(pos_other):,} "
              f"neg={len(tr_ctrl):,})  test={len(test_idx):,} "
              f"(pos_{chem_x}={len(pos_x):,} neg={len(te_ctrl):,})", flush=True)
        if len(pos_x) == 0 or len(te_ctrl) == 0:
            raise SystemExit(f"empty test for {a.fold}: pos={len(pos_x)} neg={len(te_ctrl)}")
        model = fit(train_idx, a.fold, 'loco')
        record(model, pool, test_idx, a.adv_target, device, hp, rows, a.fold,
              f'zeroshot_{chem_x}', held=chem_x)

    cols = ['fold', 'test_set', 'held_out', 'micro_f1', 'mod_f1', 'unmod_f1', 'macro_f1',
            'mod_prec', 'mod_rec', 'auprc', 'auroc', 'adv_acc', 'threshold', 'n_pos', 'n_test']
    tsv = out / 'metrics' / f'{a.fold}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(f"{r[c]:.6f}" if isinstance(r.get(c), float)
                               else str(r.get(c, '')) for c in cols) + '\n')
    print(f"\nWrote {tsv}\nDONE [{a.fold}]", flush=True)
    if wandb_run is not None:
        try:
            wandb_run.save(str(tsv)); wandb_run.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
