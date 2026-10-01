#!/usr/bin/env python3
"""refeaturize_strand_resolved_modkit.py with each dataset's genomic holdout
region excluded -- the modkit-label counterpart of
refeaturize_strand_resolved_holdout.py.

Why this file exists: we now have two batteries that should differ in exactly
one thing, the train/test split.

  rawmod_strandres_merged_fast          position split   modkit labels
  rawmod_strand_resolved_holdout        genome split     MOTIF labels (stale)

The second tree predates modkit_gt.py, so its bacterial/SPO1 arms are still
~100% positive (barcode06_plus: 39,987/39,987) and a fold trained on it is not
comparable to the first. This regenerates the 12 modkit datasets with
--exclude-bed so a merged root can pair modkit labels with the genome split.

The other 20 datasets (ONT, human, arabidopsis, HPJ99, Anabaena, Tdenticola)
keep motif labels in BOTH batteries, so their holdout files are reused as-is
from rawmod_strand_resolved_holdout/features_fast.

Usage:
  python refeaturize_strand_resolved_modkit_holdout.py --dry-run
  python refeaturize_strand_resolved_modkit_holdout.py [--dependency IDS]
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

OUT = Path(f'{RAWMOD_STORE}/rawmod_strand_resolved_modkit_holdout')

# Only the datasets that actually have modkit ground truth featurized; the rest
# are reused from the motif holdout tree, unchanged.
DATASETS = sorted({p.name.rsplit('_', 1)[0]
                   for p in (Path(mk.OUT) / 'features_fast').glob('*.h5')})


def build(d, strand):
    out, cmd = mk.build(d, strand)
    tag = 'plus' if strand == '+' else 'minus'
    out = OUT / 'features' / f"{d['name']}_{tag}.h5"
    cmd = ['--output' if c == '--output' else c for c in cmd]
    i = cmd.index('--output') + 1
    cmd[i] = str(out)
    return out, cmd + ['--exclude-bed', EXCLUDE_BED[d['name']]]


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
    missing = sorted({b for n in DATASETS for b in [EXCLUDE_BED.get(n)] if not b or not Path(b).exists()})
    if missing:
        raise SystemExit('missing holdout BEDs:\n  ' + '\n  '.join(map(str, missing)))
    n = 0
    for d in base.D:
        if d['name'] not in DATASETS or (only and d['name'] not in only):
            continue
        for strand in a.strands.split(','):
            out, cmd = build(d, strand)
            if out.exists():
                print(f'[skip] {out.name} exists'); continue
            tag = 'plus' if strand == '+' else 'minus'
            sb = ['sbatch', '--parsable', '--partition=scavenger', '--account=scavenger',
                  '--qos=scavenger', '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8',
                  f"--mem={d['mem']}", '--time=16:00:00', '--exclude=legacy43',
                  f"--job-name=mkh_{d['name']}_{tag}",
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
