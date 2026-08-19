#!/usr/bin/env python3
"""
rawmod results13: BCE + SupCon-on-presence (results6's backbone) + a
gradient-reversed ORGANISM/dataset adversary with lambda ramp, on the
strand-split 15-read data (RAWMOD_DATA_GEN=strand15).

MOTIVATION
----------
User wants an embedding with two well-separated balls: all MODIFIED sites in
one cluster, all UNMODIFIED sites in another, regardless of chemistry or
organism. Two findings from this session motivate this specific combination
(see memory: strand-split-partial-fix, results7-8-dann-backfire,
organism-identifiability-root-cause):

  1. results6's SupCon head was ALREADY trained on the BINARY presence label
     (not chemistry type) -- and a post-hoc check found it already achieves a
     2-cluster GMM ARI of 0.714 against mod/unmod (vs ~0 for every DANN
     variant that dropped SupCon/BCE/SAD entirely). SupCon-on-presence is a
     genuine ATTRACTIVE pull toward two balls; nothing in the DANN family
     (results7-12) has any pull at all, only erasure -- that's why they never
     showed 2-ball structure.
  2. Gradient-reversal (results8/10/12, "dataset" target) measurably reduced
     organism-ARI vs a no-adversary baseline once lambda was ramped properly,
     but organism substructure still isn't gone. An adversary supplies erasure
     with no pull; SupCon supplies pull with no anti-organism erasure. Neither
     alone is sufficient -- this experiment runs them together: SupCon's pull
     keeps the two presence-balls intact at the macro level while the
     organism-adversary's erasure fights whatever organism substructure
     survives WITHIN each ball.

Model: ConvFormerV2 (run_convformer_v2.py) with supcon_dim>0 (existing head,
trained on the binary presence label, as in results6) AND the NEW
org_adv_classes=3 gradient-reversed organism head (added this session,
reusing the same grad_reverse primitive as the model's existing sequence-
context adversary). NOT ConvFormerV2DANN (which has no BCE/SupCon/SAD at all).

Same matched pool / curriculum / LOCO fold logic as run_matched_loco.py and
run_dann_loco.py, reused wholesale via `import run_matched_loco as ML`.

Usage:
  RAWMOD_DATA_GEN=strand15 python run_supcon_orgadv_loco.py \
      --fold {mixed|loco_<CHEM>} --out-dir <dir> [--epochs N]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / 'experiments' / 'pipeline1', REPO / 'deepmod'):
    sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_pipeline as R                                       # noqa: E402
import run_matched_loco as ML                                  # noqa: E402
from run_convformer_v2 import ConvFormerV2                       # noqa: E402
from run_dann_loco import chem_to_org, adv_lambda_schedule, class_weights, ORG_CLASSES  # noqa: E402
from model import make_loader_kwargs, _worker_init_fn, aggregate_by_position, SupConLoss  # noqa: E402

WANDB_ENTITY = os.environ.get('WANDB_ENTITY', 'bds062-university-of-maryland')
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'rawmod')

SUPCON_DIM = int(os.environ.get('SUPCON_DIM', '128'))


class OrgDataset(Dataset):
    """Wraps a base PileupDataset, additionally returning the row's organism
    class id. Every image has an organism (no validity mask needed, unlike the
    type-adversary case in run_dann_loco.py)."""
    def __init__(self, base_ds, org_y):
        self.base = base_ds
        self.org_y = org_y

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        x, y = self.base[item]
        return x, y, int(self.org_y[item])


def presence_probs(model, loader, device):
    model.eval(); all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            logit = model(x.to(device, non_blocking=True))
            p = torch.sigmoid(logit.squeeze(1)).cpu().numpy()
            all_y.append(y.numpy()); all_p.append(p)
    return np.concatenate(all_y), np.concatenate(all_p)


def eval_with_org(model, loader, device):
    """One eval-mode forward pass; captures the penultimate rep via a forward
    hook on `head` (its INPUT) and applies org_adv_head directly -- valid since
    grad_reverse is identity on the forward pass."""
    cap = {}
    hook = model.head.register_forward_hook(
        lambda _m, inp, _o: cap.__setitem__('rep', inp[0].detach()))
    model.eval()
    all_y, all_p, all_orgy, all_orgpred = [], [], [], []
    with torch.no_grad():
        for x, y, orgy in loader:
            logit = model(x.to(device, non_blocking=True))
            p = torch.sigmoid(logit.squeeze(1)).cpu().numpy()
            org_logits = model.org_adv_head(cap['rep'])
            org_pred = org_logits.argmax(dim=1).cpu().numpy()
            all_y.append(y.numpy()); all_p.append(p)
            all_orgy.append(orgy.numpy()); all_orgpred.append(org_pred)
    hook.remove()
    return (np.concatenate(all_y), np.concatenate(all_p),
           np.concatenate(all_orgy), np.concatenate(all_orgpred))


def fit_one(pool, train_idx, hp, device, out_dir, tag, org_id, init_state=None):
    print(f"\n{'='*66}\n  TRAIN [{tag}]  train_images={len(train_idx):,}", flush=True)
    sub_train, val_idx = R.carve_val(train_idx, pool, hp.val_frac, hp.seed)

    train_ds = OrgDataset(R.make_ds(pool, sub_train, True, hp), org_id[sub_train])
    val_ds = OrgDataset(R.make_ds(pool, val_idx, False, hp), org_id[val_idx])
    val_keys = pool.source_keys(val_idx)
    lk = make_loader_kwargs(hp.batch, hp.num_workers, device, _worker_init_fn)
    train_loader = DataLoader(train_ds, shuffle=True, **lk)
    val_loader = DataLoader(val_ds, shuffle=False, **lk)

    adv_lambda_max = float(os.environ.get('ADV_LAMBDA', '1.0'))
    adv_lambda_gamma = float(os.environ.get('ADV_LAMBDA_GAMMA', '10.0'))
    model = ConvFormerV2(dropout=hp.dropout, h=ML.HEIGHT, supcon_dim=SUPCON_DIM,
                         org_adv_classes=len(ORG_CLASSES), org_adv_lambda=0.0).to(device)
    if init_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in init_state.items()})
        print(f"  warm-started from prior stage ({len(init_state)} tensors)", flush=True)

    y_tr = (pool.labels[sub_train] > 0).astype(np.int64)
    n_pos, n_neg = int(y_tr.sum()), len(y_tr) - int(y_tr.sum())
    pw = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
    bce_crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    bce_w = float(os.environ.get('BCE_WEIGHT', '1.0'))
    supcon_w = float(os.environ.get('SUPCON_WEIGHT', '1.0'))
    supcon_crit = SupConLoss(float(os.environ.get('SUPCON_TEMP', '0.07')))
    org_w = class_weights(org_id[sub_train], len(ORG_CLASSES), device)
    print(f"  pos_weight={pw.item():.2f}  bce_w={bce_w}  supcon_w={supcon_w}  "
          f"org class weights={org_w.tolist()}  adv_lambda(max)={adv_lambda_max}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                       patience=7, min_lr=1e-6)
    warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0 / max(hp.warmup_steps, 1), end_factor=1.0,
        total_iters=hp.warmup_steps) if hp.warmup_steps > 0 else None

    total_steps = hp.epochs * max(len(train_loader), 1)
    print(f"  adv_lambda ramp: 0 -> {adv_lambda_max} over {total_steps:,} steps "
          f"(gamma={adv_lambda_gamma})", flush=True)

    best_auprc, best_state, best_ep, patience = -1.0, None, 0, 0
    gstep = 0
    tr_hist, va_hist = [], []
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, hp.epochs + 1):
        model.train()
        eloss = ebce = esup = eorg = eseen = 0
        for x, y, orgy in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            orgy = orgy.to(device, non_blocking=True)

            model.org_adv_lambda = adv_lambda_schedule(gstep, total_steps,
                                                       adv_lambda_max, adv_lambda_gamma)
            opt.zero_grad()
            logit, aux = model(x)
            bce_loss = bce_crit(logit.squeeze(1), y)
            loss = bce_w * bce_loss
            ebce += float(bce_loss) * len(y)
            if 'proj' in aux and supcon_w > 0:
                sc = supcon_crit(aux['proj'], y.long())
                loss = loss + supcon_w * sc
                esup += float(sc) * len(y)
            if 'org_adv_logits' in aux:
                org_loss = F.nll_loss(F.log_softmax(aux['org_adv_logits'], dim=1),
                                      orgy, weight=org_w)
                loss = loss + org_loss
                eorg += float(org_loss) * len(y)
            loss.backward()
            if hp.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step(); gstep += 1
            if warm is not None and gstep <= hp.warmup_steps:
                warm.step()
            eloss += float(loss) * len(y); eseen += len(y)
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
              f"bce={ebce/max(eseen,1):.4f}  supcon={esup/max(eseen,1):.4f}  "
              f"org_adv={eorg/max(eseen,1):.4f}  lambda={model.org_adv_lambda:.3f}  "
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
               'epoch': best_ep, 'tag': tag, 'height': ML.HEIGHT,
               'supcon_dim': SUPCON_DIM, 'org_adv_classes': len(ORG_CLASSES),
               'adv_lambda_max': adv_lambda_max, 'adv_lambda_gamma': adv_lambda_gamma,
               'adv_lambda_schedule': 'ganin_sigmoid'},
              run_dir / 'best_model.pt')
    np.savez(run_dir / 'history.npz', train_loss=tr_hist, val_auprc=va_hist, best_epoch=best_ep)
    print(f"  best val_AUPRC={best_auprc:.4f} @ ep {best_ep}  -> {run_dir/'best_model.pt'}",
          flush=True)
    return model


def record(model, pool, idx, device, hp, rows, fold, test_name, held=''):
    ds = OrgDataset(R.make_ds(pool, idx, False, hp), record.org_id[idx])
    loader = DataLoader(ds, shuffle=False,
                        **make_loader_kwargs(hp.batch, hp.num_workers, device, _worker_init_fn))
    yt, yp, orgy, orgpred = eval_with_org(model, loader, device)
    keys = pool.source_keys(idx)
    t, p, _ = aggregate_by_position(yt, yp, keys)
    m = R.compute_metrics(t, p)
    m['adv_acc'] = float((orgpred == orgy).mean())
    rows.append({'fold': fold, 'test_set': test_name, 'held_out': held, **m})
    print(f"  EVAL {test_name}: mod_f1={m['mod_f1']:.3f} mod_rec={m['mod_rec']:.3f} "
          f"mod_prec={m['mod_prec']:.3f} auprc={m['auprc']:.3f} auroc={m['auroc']:.3f} "
          f"org_adv_acc={m['adv_acc']:.3f} n_pos={m['n_pos']} n_test={m['n_test']}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fold', required=True,
                    help="'mixed' or 'loco_<CHEM>' with CHEM in " + '/'.join(ML.CHEMS))
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
    print(f"Device {device}  fold={a.fold}  out={out}  height={ML.HEIGHT} "
          f"(RAWMOD_DATA_GEN={os.environ.get('RAWMOD_DATA_GEN','')!r})", flush=True)

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
    record.org_id = org_id     # stash for record()

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
                name=f"supcon-orgadv-{rid}-{a.fold}", group='supcon_orgadv',
                job_type=a.fold.split('_')[0],
                config={'fold': a.fold, 'architecture': 'ConvFormerV2+org_adv',
                        **{k: getattr(hp, k) for k in dir(hp) if not k.startswith('_')}},
                settings=wandb.Settings(init_timeout=180, start_method='thread'))
            print(f"  [wandb] {WANDB_ENTITY}/{WANDB_PROJECT}/supcon-orgadv-{rid}-{a.fold}",
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
                m1 = fit_one(pool, anc, hp1, device, mdir, runtag + '_cur1', org_id)
                state = {k: v.detach().cpu().clone() for k, v in m1.state_dict().items()}
                del m1
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                return fit_one(pool, train_idx, hp, device, mdir, runtag, org_id,
                              init_state=state)
            print("  [curriculum] too few paired anchors; single-stage fallback", flush=True)
        return fit_one(pool, train_idx, hp, device, mdir, runtag, org_id)

    rows = []
    if a.fold == 'mixed':
        train_idx, test_idx, stats = ML.mixed_split(pool, is_pos, neg_mask, hp)
        print(f"  {stats}", flush=True)
        model = fit(train_idx, a.fold, 'mixed')
        record(model, pool, test_idx, device, hp, rows, a.fold, 'held_out_test')
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
        record(model, pool, test_idx, device, hp, rows, a.fold,
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
