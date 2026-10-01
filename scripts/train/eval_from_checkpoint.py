#!/usr/bin/env python3
"""Score a saved fold checkpoint and write its metrics TSV.

run_matched_loco.py only writes metrics/<fold>.tsv at the very end, after
training finishes. A job that hits its SLURM wall clock therefore loses the
result even though models/<fold>/<runtag>/best_model.pt was saved at the best
epoch -- that is how loco_5hmC was lost at the 16 h limit on 2026-09-20. This
rebuilds the same pool and the same fold test indices, loads the checkpoint and
writes the identical TSV, so a timeout costs the remaining epochs and nothing
else.

The fold reconstruction is deterministic given (pool, SPLIT_SEED) and was
verified against a completed run's log (loco_5hmU: 161,813 test images, same
as the job reported).

Usage (same environment as the training run -- RAWMOD_DATA_GEN,
EXTRA_ORGANISMS, INCLUDE_HUMAN, RAWMOD_STRANDRES_ROOT, TF_LAYERS, SUPCON_DIM,
SAD_DIM must match, or the architecture/pool will not line up):

  python eval_from_checkpoint.py --fold loco_5hmU --out-dir .../results24_modkit
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rawmod'))

import run_matched_loco as M                                  # noqa: E402
from run_convformer_v2 import ConvFormerV2                    # noqa: E402

COLS = ['fold', 'test_set', 'held_out', 'micro_f1', 'mod_f1', 'unmod_f1',
        'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc', 'auroc_sad',
        'threshold', 'n_pos', 'n_test']


def fold_test_idx(pool, chem, refbase, is_pos, neg_mask, is_bench, fold, hp):
    if fold == 'mixed':
        _, test_idx, _ = M.mixed_split(pool, is_pos, neg_mask, hp,
                                       core_mask=~is_bench)
        return test_idx, ''
    chem_x = fold[len('loco_'):]
    ctrl_idx = np.nonzero(neg_mask)[0]
    _, te_ctrl = M.pos_hash_split(pool, ctrl_idx, test_frac=0.15, seed=M.SPLIT_SEED)
    bases, orgs = M.CHEM_BASES[chem_x], M.CHEM_ORGS[chem_x]
    te_ctrl = np.array([i for i in te_ctrl
                        if M.org_of(pool.names[int(pool.file_of[i])]) in orgs
                        and refbase[i] in bases], dtype=np.int64)
    pos_x = np.nonzero(is_pos & (chem == chem_x) & ~is_bench)[0].astype(np.int64)
    return np.sort(np.concatenate([pos_x, te_ctrl])), chem_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--checkpoint', default=None,
                    help='default: <out-dir>/models/<fold>/<runtag>/best_model.pt')
    a = ap.parse_args()
    out = Path(a.out_dir)
    runtag = 'mixed' if a.fold == 'mixed' else 'loco'
    ckpt = Path(a.checkpoint) if a.checkpoint else (
        out / 'models' / a.fold / runtag / 'best_model.pt')
    if not ckpt.exists():
        raise SystemExit(f'no checkpoint at {ckpt}')

    hp = M.R.HP()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = M.build_members()
    pool = M.R.Group(list(members), members)
    print(f'pool: {pool.N:,} images across {len(members)} files', flush=True)
    mod_map = M.build_umces_mod_map(M.R.UMCES_PILEUPS, M.R.UMCES_REF)
    refbase = M.ref_base_center(pool)
    chem = M.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    neg_mask = np.zeros(pool.N, dtype=bool)
    neg_mask[M.subsample_negatives(pool, seed=M.SPLIT_SEED)] = True
    is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::')
                         for i in range(pool.N)])

    test_idx, held = fold_test_idx(pool, chem, refbase, is_pos, neg_mask,
                                   is_bench, a.fold, hp)
    print(f'{a.fold}: test images={len(test_idx):,} '
          f'(pos={int(is_pos[test_idx].sum()):,})', flush=True)

    sad_dim = int(os.environ.get('SAD_DIM', '0'))
    in_ch = 10 if os.environ.get('RAWMOD_DROP_CH9', '0') == '1' else 11
    model = ConvFormerV2(in_ch=in_ch, dropout=hp.dropout,
                         supcon_dim=int(os.environ.get('SUPCON_DIM', '128')),
                         sad_dim=sad_dim, h=M.HEIGHT,
                         layers=int(os.environ.get('TF_LAYERS', '2')),
                         row_emb=os.environ.get('ROW_EMB', '1') == '1').to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get('model_state', state.get('model', state)))
    if 'sad_center' in state.get('model_state', {}):
        model.sad_center = state['model_state']['sad_center'].to(device)
    model.eval()
    print(f'loaded {ckpt} (epoch {state.get("epoch", "?")}, '
          f'val_auprc {state.get("val_auprc", float("nan")):.4f})', flush=True)

    m = M.R.evaluate(model, pool, test_idx, device, hp)
    name = 'held_out_test' if a.fold == 'mixed' else f'zeroshot_{held}'
    print(f'  EVAL {name}: auroc={m["auroc"]:.3f} auprc={m["auprc"]:.3f} '
          f'mod_f1={m["mod_f1"]:.3f} n_pos={m["n_pos"]} n_test={m["n_test"]}',
          flush=True)
    row = {'fold': a.fold, 'test_set': name, 'held_out': held, **m}
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    tsv = out / 'metrics' / f'{a.fold}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        fh.write('\t'.join(f'{row[c]:.6f}' if isinstance(row.get(c), float)
                           else str(row.get(c, '')) for c in COLS) + '\n')
    print(f'wrote {tsv}\nDONE [{a.fold}] (from checkpoint)', flush=True)


if __name__ == '__main__':
    main()
