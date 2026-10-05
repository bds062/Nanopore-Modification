#!/usr/bin/env python3
"""Every genomic position the released RawMod model (checkpoints/rawmod_final/mixed.pt) was trained on.

Rebuilds the training pool exactly as scripts/train/run_matched_loco.py main() does for --fold mixed, with the training
job's settings (ENV below), and writes one row per pool image:

  file  contig  pos  strand  label  split

split = train  (in mixed_split's train_idx, or a BENCH:: stage-2 image that fit() unions into training)
        test   (mixed_split's held-out test_idx)
        unused (in the pool but in neither: controls beyond the cap, ambiguous BENCH positives, BGCTRL test-only)

The counts are checked against the training log (train=386,750 test=67,534 BENCH stage-2=524,306) and the script
stops if they differ. The result is shipped as test/data/holdout/rawmod_seen_results81_mixed.tsv.gz; rerunning this
script needs RawMod's training features (RAWMOD_STRANDRES_ROOT, see scripts/train). NEG_CAP_SPO1=200000 is part of the
training configuration: with the default cap the control set, and therefore the split, differs.

  RAWMOD_STRANDRES_ROOT=/path/to/features python -m bench.rawmod_seen [--out FILE]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from .settings import REPO

TRAIN = REPO / 'scripts' / 'train'
EXPECT = {'train': 386_750, 'test': 67_534, 'bench': 524_306, 'pool': 1_264_478}
ENV = dict(RAWMOD_DATA_GEN='strandres', EXTRA_ORGANISMS='1', INCLUDE_HUMAN='1', TF_LAYERS='2', ROW_EMB='1',
           RAWMOD_DROP_CH9='0', NEG_CAP_SPO1='200000', PILEUP_MASK_BASES='0', PILEUP_PRELOAD='0', CURRICULUM='1', SUPCON_DIM='128', SAD_DIM='16')


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(C.WORK / 'rawmod_seen_results81_mixed.tsv.gz'))
    a = ap.parse_args()
    if not os.environ.get('RAWMOD_STRANDRES_ROOT'):
        sys.exit('RAWMOD_STRANDRES_ROOT (RawMod training features) is not set')
    for k, v in ENV.items():
        if os.environ.get(k, v) != v:
            sys.exit(f'{k}={os.environ[k]} but the training job used {v}')
        os.environ[k] = v
    sys.path.insert(0, str(TRAIN)); sys.path.insert(0, str(TRAIN.parents[1] / 'rawmod'))
    import run_matched_loco as M

    members = M.build_members()
    names = list(members)
    pool = M.R.Group(names, members)
    mod_map = M.build_umces_mod_map(M.R.UMCES_PILEUPS, M.R.UMCES_REF)
    refbase = M.ref_base_center(pool)
    chem = M.chem_array(pool, mod_map, refbase)
    is_pos = pool.labels > 0
    kept_neg = M.subsample_negatives(pool, seed=M.SPLIT_SEED)
    neg_mask = np.zeros(pool.N, dtype=bool); neg_mask[kept_neg] = True
    fname = np.array(names)[pool.file_of]
    is_bench = np.char.startswith(fname.astype(str), 'BENCH::')
    is_bgctrl = np.char.startswith(fname.astype(str), 'BGCTRL::')
    bench_idx = np.nonzero(is_bench)[0]
    if os.environ.get('EXCLUDE_UNTYPED_POS', '1') == '1':
        amb = is_pos[bench_idx] & np.isin(chem[bench_idx], ['', 'untyped'])
        bench_idx = bench_idx[~amb]
    from collections import Counter
    print('positives per chemistry:', {k: int(v) for k, v in Counter(chem[is_pos]).items()}, flush=True)
    print(f'controls kept (capped): {int(neg_mask.sum()):,} of {int((~is_pos).sum()):,}', flush=True)
    core = (is_pos | neg_mask) & ~(is_bench | is_bgctrl)
    print('core images per file:', pd.Series(fname[core]).value_counts().to_dict(), flush=True)
    print('is_bgctrl', int(is_bgctrl.sum()), 'is_bench', int(is_bench.sum()), flush=True)
    tr, te, _ = M.mixed_split(pool, is_pos, neg_mask, M.HP() if hasattr(M, 'HP') else M.R.HP(),
                              core_mask=~(is_bench | is_bgctrl))
    got = {'train': len(tr), 'test': len(te), 'bench': len(bench_idx), 'pool': pool.N}
    print('counts', got, flush=True)
    if got != EXPECT:
        sys.exit(f'pool does not match the training log: expected {EXPECT}')
    split = np.full(pool.N, 'unused', dtype=object)
    split[te] = 'test'
    split[np.union1d(tr, bench_idx)] = 'train'
    strand = np.array(['-' if n.endswith('|-') else '+' if n.endswith('|+') else '.' for n in names])[pool.file_of]
    out = Path(a.out)
    pd.DataFrame({'file': fname, 'contig': np.asarray(pool.contig), 'pos': np.asarray(pool.ref_pos, dtype=np.int64),
                  'strand': strand, 'label': pool.labels.astype(int), 'split': split}).to_csv(out, sep='\t', index=False)
    print(f'wrote {out}')
    print(pd.crosstab(pd.Series(fname).str.replace(r'\|[+-]$', '', regex=True), split).to_string())


if __name__ == '__main__':
    main()
