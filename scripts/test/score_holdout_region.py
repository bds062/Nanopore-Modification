#!/usr/bin/env python3
"""Score a results26 checkpoint on the genomic region its training tree held out.

Every AUROC in results23/24 comes from a POSITION-grouped split
(run_matched_loco.mixed_split -> split_position_groups): train and test
coordinates are disjoint, but they are interleaved through the same genome and
one nanopore read spans hundreds of them, so a read behind a test image is
very often also behind training images at neighbouring positions. That is an
unseen-POSITION measurement.

This is the unseen-GENOME measurement. The checkpoint comes from
rawmod_strandres_merged_holdout_fast, where each organism's holdout region was
removed including every read that so much as touches it; the images scored here
come from rawmod_strandres_region_only, which is exactly that region. No read
in this test set contributed to training.

Fold semantics match run_matched_loco.py, but there is no 85/15 split: the
whole region is the test set.
  mixed    -- every core (non-BENCH) image in the region
  loco_X   -- every chemistry-X positive in the region, plus the region's
              negatives filtered by CHEM_ORGS[X]/CHEM_BASES[X]

Negatives are put through the same deterministic subsample_negatives() the
training folds use, so prevalence is constructed identically and the number is
comparable to the results24/26 tables. --all-negatives skips that and reports
the (harder, lower-prevalence) full-region number as well.

Usage:
  RAWMOD_STRANDRES_ROOT=$RAWMOD_STORE/rawmod_strandres_region_only/features \
  RAWMOD_DATA_GEN=strandres EXTRA_ORGANISMS=1 INCLUDE_HUMAN=1 SUPCON_DIM=128 \
  SAD_DIM=16 TF_LAYERS=2 \
  python score_holdout_region.py --fold loco_5mC \
      --ckpt-dir .../results26_modkit_holdout --out-dir .../results27_region_score
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts' / 'train'))
sys.path.insert(0, str(REPO / 'rawmod'))

import run_matched_loco as M                                  # noqa: E402
from run_convformer_v2 import ConvFormerV2                    # noqa: E402

COLS = ['fold', 'test_set', 'held_out', 'micro_f1', 'mod_f1', 'unmod_f1',
        'macro_f1', 'mod_prec', 'mod_rec', 'auprc', 'auroc', 'auroc_sad',
        'threshold', 'n_pos', 'n_test']


def region_test_idx(pool, chem, refbase, is_pos, neg_mask, is_bench, fold):
    core = ~is_bench
    if fold == 'mixed':
        return np.nonzero((is_pos | neg_mask) & core)[0].astype(np.int64), ''
    chem_x = fold[len('loco_'):]
    bases, orgs = M.CHEM_BASES[chem_x], M.CHEM_ORGS[chem_x]
    neg = np.array([i for i in np.nonzero(neg_mask & core)[0]
                    if M.org_of(pool.names[int(pool.file_of[i])]) in orgs
                    and refbase[i] in bases], dtype=np.int64)
    pos = np.nonzero(is_pos & (chem == chem_x) & core)[0].astype(np.int64)
    return np.sort(np.concatenate([pos, neg])), chem_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', required=True)
    ap.add_argument('--ckpt-dir', required=True,
                    help='a results<N> dir; the checkpoint is '
                         '<ckpt-dir>/models/<fold>/<runtag>/best_model.pt')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--all-negatives', action='store_true',
                    help='also score without subsample_negatives()')
    a = ap.parse_args()

    runtag = 'mixed' if a.fold == 'mixed' else 'loco'
    ckpt = Path(a.ckpt_dir) / 'models' / a.fold / runtag / 'best_model.pt'
    if not ckpt.exists():
        raise SystemExit(f'no checkpoint at {ckpt}')
    root = os.environ.get('RAWMOD_STRANDRES_ROOT', '')
    if 'region_only' not in root:
        raise SystemExit(f'RAWMOD_STRANDRES_ROOT must point at the region-only '
                         f'tree, got {root!r}')

    hp = M.R.HP()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = M.build_members()
    pool = M.R.Group(list(members), members)
    print(f'region pool: {pool.N:,} images across {len(members)} files', flush=True)
    mod_map = M.build_umces_mod_map(M.R.UMCES_PILEUPS, M.R.UMCES_REF)
    refbase = M.ref_base_center(pool)
    chem = M.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    is_bench = np.array([pool.names[int(pool.file_of[i])].startswith('BENCH::')
                         for i in range(pool.N)])

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
    print(f'loaded {ckpt} (epoch {state.get("epoch", "?")})', flush=True)

    variants = [('subsampled', False)] + ([('all_neg', True)] if a.all_negatives else [])
    rows = []
    for vname, allneg in variants:
        neg_mask = np.zeros(pool.N, dtype=bool)
        if allneg:
            neg_mask[~is_pos] = True
        else:
            neg_mask[M.subsample_negatives(pool, seed=M.SPLIT_SEED)] = True
        test_idx, held = region_test_idx(pool, chem, refbase, is_pos,
                                         neg_mask, is_bench, a.fold)
        if len(test_idx) == 0 or is_pos[test_idx].sum() == 0:
            print(f'  [{vname}] no positives in the held-out region -- skipped',
                  flush=True)
            continue
        m = M.R.evaluate(model, pool, test_idx, device, hp)
        name = f'region_{vname}' if a.fold == 'mixed' else f'region_{held}_{vname}'
        print(f'  EVAL {name}: auroc={m["auroc"]:.3f} auprc={m["auprc"]:.3f} '
              f'mod_f1={m["mod_f1"]:.3f} n_pos={m["n_pos"]} n_test={m["n_test"]}',
              flush=True)
        rows.append({'fold': a.fold, 'test_set': name, 'held_out': held, **m})

    out = Path(a.out_dir)
    (out / 'metrics').mkdir(parents=True, exist_ok=True)
    tsv = out / 'metrics' / f'{a.fold}.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        for row in rows:
            fh.write('\t'.join(f'{row[c]:.6f}' if isinstance(row.get(c), float)
                               else str(row.get(c, '')) for c in COLS) + '\n')
    print(f'wrote {tsv}\nDONE [{a.fold}] (region score)', flush=True)


if __name__ == '__main__':
    main()
