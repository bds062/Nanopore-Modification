#!/usr/bin/env python3
"""Re-run refeaturize_benchmark.py's exact 9 curriculum datasets, unchanged in
every flag, except each gets --exclude-bed pointed at its organism's genomic
holdout region (scripts/ground_truth/select_holdout_regions.py) and output
goes to a separate tree (rawmod_full_pipeline4_holdout/) so the production
features used by checkpoints/results20_sad_dim16/ are untouched.

Every dataset here already keys its ground truth off `gt_name`, and
select_holdout_regions.py wrote holdout_region.bed into that exact
data/gt/<gt_name>/ directory for all nine -- so no separate name-mapping
table is needed (contrast refeaturize_strand15_holdout.py, where three
datasets have no data/gt/ directory at all).

Point a training run at this tree with:
  RAWMOD_FEATURES_ROOT=/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4_holdout/features

Usage:
  python refeaturize_benchmark_holdout.py --dry-run
  python refeaturize_benchmark_holdout.py
  python refeaturize_benchmark_holdout.py --only Anabaena_WT_5kHz,hg001
"""
import argparse
import subprocess
from pathlib import Path

import refeaturize_benchmark as base

GT_ROOT = base.GT_ROOT
OUT_ROOT = '/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4_holdout/features/benchmark'


def build_cmd(name, pod5_root, pod5_sub, gt_name, min_reads, sample_n_sites):
    out_path = f'{OUT_ROOT}/{name}/features.h5'
    _, parts = base.build_cmd(name, pod5_root, pod5_sub, gt_name, min_reads, sample_n_sites)
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

    for name, pod5_root, pod5_sub, gt_name, min_reads, sample_n_sites, mem in base.DATASETS:
        if only and name not in only:
            continue
        holdout_bed = Path(GT_ROOT) / gt_name / 'holdout_region.bed'
        assert holdout_bed.exists(), f"missing {holdout_bed}"
        out_path, cmd = build_cmd(name, pod5_root, pod5_sub, gt_name, min_reads, sample_n_sites)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        wrap = f"{base.CONDA_INIT} && " + ' '.join(cmd)
        sbatch = [
            'sbatch', '--parsable',
            '--partition=scavenger', '--account=scavenger', '--qos=scavenger',
            '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8', f'--mem={mem}',
            '--time=08:00:00',
            f'--job-name=featbenchh_{name}',
            f'--output={logdir}/{name}_%j.out', f'--error={logdir}/{name}_%j.out',
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
