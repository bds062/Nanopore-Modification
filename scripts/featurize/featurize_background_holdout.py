#!/usr/bin/env python3
"""Re-run featurize_background.py's exact 6 datasets, unchanged in every flag,
except each gets --exclude-bed pointed at its organism's genomic holdout
region and output goes to the same rawmod_full_pipeline4_holdout/ tree as
refeaturize_{strand15,benchmark}_holdout.py.

Needed even though BGCTRL:: background images are only ever ADDED to a
fold's train/test split for logo_bacteria (see run_matched_loco.py) --
run_pipeline.Group's constructor unconditionally opens every member file
listed by build_members() up front, for every fold, so these files must
exist under RAWMOD_FEATURES_ROOT regardless of which fold is being trained.
(Found the hard way: mixed/loco_<CHEM> all failed with FileNotFoundError on
<org>_background/features.h5 before this script existed.)

Usage:
  python featurize_background_holdout.py --dry-run
  python featurize_background_holdout.py
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from rawmod.paths import RAWMOD_RESULTS  # noqa: E402  (site paths; see paths.env.example)

import argparse
import subprocess
from pathlib import Path

import featurize_background as base

GT_ROOT = base.GT_ROOT
OUT_ROOT = f'{RAWMOD_RESULTS}/rawmod_full_pipeline4_holdout/features/benchmark'


def build_cmd(name, pod5_sub, gt_name, min_reads, sample_n_sites):
    out_path = f'{OUT_ROOT}/{name}_background/features.h5'
    _, parts = base.build_cmd(name, pod5_sub, gt_name, min_reads, sample_n_sites)
    out_idx = parts.index('--output') + 1
    parts[out_idx] = out_path
    parts += ['--exclude-bed', f'{GT_ROOT}/{gt_name}/holdout_region.bed']
    return out_path, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None

    logdir = Path(OUT_ROOT) / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)

    for name, pod5_sub, gt_name, min_reads, sample_n_sites, mem in base.DATASETS:
        if only and name not in only:
            continue
        holdout_bed = Path(GT_ROOT) / gt_name / 'holdout_region.bed'
        assert holdout_bed.exists(), f"missing {holdout_bed}"
        out_path, cmd = build_cmd(name, pod5_sub, gt_name, min_reads, sample_n_sites)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        wrap = f"{base.CONDA_INIT} && " + ' '.join(cmd)
        sbatch = [
            'sbatch', '--parsable',
            '--partition=scavenger', '--account=scavenger', '--qos=scavenger',
            '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8', f'--mem={mem}',
            '--time=04:00:00',
            f'--job-name=featbgh_{name}',
            f'--output={logdir}/{name}_bg_%j.out', f'--error={logdir}/{name}_bg_%j.out',
            f'--wrap={wrap}',
        ]
        if a.dry_run:
            print(f"[dry-run] {' '.join(sbatch)}\n")
        else:
            r = subprocess.run(sbatch, capture_output=True, text=True, check=True)
            jid = r.stdout.strip()
            print(f"{name}: job {jid}  (mem={mem}, min_reads={min_reads}, "
                 f"sample_n_sites={sample_n_sites})")


if __name__ == '__main__':
    main()
