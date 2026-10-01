#!/usr/bin/env python3
"""The complement of refeaturize_strand_resolved_{,modkit_}holdout.py: featurize
ONLY each organism's held-out genomic region.

Why: every number we have so far comes from a position-grouped split. Train and
test coordinates are disjoint, but they are interleaved across the same genome
and a single read spans many of them, so a read behind a test image can also sit
behind training images at neighbouring positions. That measures "unseen
position", not "unseen genome".

The holdout trees remove these regions (positions AND every read touching them).
This tree is the other half: the same regions, all reads kept, labels from the
same ground truth each dataset uses in the merged root -- modkit-thresholded for
the 12 matched datasets, motif for the rest. A checkpoint trained on
rawmod_strandres_merged_holdout_fast scored against this tree is the real
unseen-genome test.

Usage:
  python refeaturize_strand_resolved_region_only.py --dry-run
  python refeaturize_strand_resolved_region_only.py [--only NAME,NAME] [--dependency IDS]
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from rawmod.paths import RAWMOD_STORE  # noqa: E402  (site paths; see paths.env.example)

import argparse
import subprocess
from pathlib import Path

import refeaturize_strand_resolved as base
import refeaturize_strand_resolved_modkit as mk
from refeaturize_strand_resolved_holdout import EXCLUDE_BED

OUT = Path(f'{RAWMOD_STORE}/rawmod_strandres_region_only')

# Datasets whose labels come from modkit in the merged root; everything else
# keeps motif labels, exactly as in build_merged_root.sh.
MODKIT = sorted({p.name.rsplit('_', 1)[0]
                 for p in (Path(mk.OUT) / 'features_fast').glob('*.h5')})


def build(d, strand):
    builder = mk.build if d['name'] in MODKIT else base.build
    _, cmd = builder(d, strand)
    tag = 'plus' if strand == '+' else 'minus'
    out = OUT / 'features' / f"{d['name']}_{tag}.h5"
    i = cmd.index('--output') + 1
    cmd = list(cmd)
    cmd[i] = str(out)
    return out, cmd + ['--include-bed', EXCLUDE_BED[d['name']]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--strands', default='+,-')
    ap.add_argument('--dependency', default=None)
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    (OUT / 'features').mkdir(parents=True, exist_ok=True)
    n = 0
    for d in base.D:
        if d['name'] not in EXCLUDE_BED or (only and d['name'] not in only):
            continue
        for strand in a.strands.split(','):
            out, cmd = build(d, strand)
            if out.exists():
                print(f'[skip] {out.name} exists'); continue
            tag = 'plus' if strand == '+' else 'minus'
            sb = ['sbatch', '--parsable', '--partition=scavenger', '--account=scavenger',
                  '--qos=scavenger', '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8',
                  f"--mem={d['mem']}", '--time=16:00:00', '--exclude=legacy43',
                  f"--job-name=reg_{d['name']}_{tag}",
                  f"--output={OUT}/logs/%x_%j.out", f"--error={OUT}/logs/%x_%j.err"]
            if a.dependency:
                sb.append(f'--dependency=afterany:{a.dependency}')
            sb += ['--wrap', f'{base.CONDA} && ' + ' '.join(str(c) for c in cmd)]
            if a.dry_run:
                print(' '.join(str(c) for c in cmd), '\n')
            else:
                print(f"submitted {subprocess.run(sb, capture_output=True, text=True).stdout.strip()}"
                      f"  {d['name']}_{tag}")
            n += 1
    print(f'{n} jobs')


if __name__ == '__main__':
    main()
