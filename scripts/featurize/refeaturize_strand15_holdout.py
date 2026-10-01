#!/usr/bin/env python3
"""Re-run refeaturize_strand15.py's exact 13 datasets, unchanged in every flag,
except each gets --exclude-bed pointed at its organism's genomic holdout
region (scripts/ground_truth/select_holdout_regions.py) and output goes to a
separate tree (rawmod_full_pipeline4_holdout/) so the production
rawmod_full_pipeline4/ features used by checkpoints/results20_sad_dim16/ are
untouched.

Point a training run at this tree with:
  RAWMOD_FEATURES_ROOT=/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4_holdout/features

Usage:
  python refeaturize_strand15_holdout.py --dry-run
  python refeaturize_strand15_holdout.py
  python refeaturize_strand15_holdout.py --only HP26695_WT_5kHz,barcode06
"""
import argparse
import subprocess
from pathlib import Path

import refeaturize_strand15 as base

GT_ROOT = '/fs/cbcb-scratch/bds062/data/gt'
OUT_ROOT = Path('/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4_holdout')

# dataset name (from refeaturize_strand15.DATASETS) -> holdout BED. HP26695 and
# SPO1/UMCES/ONT_all5mers each share ONE reference across all their datasets
# (see select_holdout_regions.py), so several names map to the same file.
_HP = f'{GT_ROOT}/hpylori_26695/holdout_region.bed'
_SPO1 = f'{GT_ROOT}/spo1_umces_holdout_region.bed'
_ONT = f'{GT_ROOT}/ont_all5mers_holdout_region.bed'
EXCLUDE_BED = {
    'HP26695_WT_5kHz': _HP, 'HP26695_WGA_5kHz': _HP,
    'barcode06': _SPO1, 'barcode07': _SPO1,
    'barcode02_train': _SPO1, 'barcode03_train': _SPO1,
    'barcode04_train': _SPO1, 'barcode05_train': _SPO1,
    'barcode01_test': _SPO1,
    'ONT_control': _ONT, 'ONT_5mC': _ONT, 'ONT_5hmC': _ONT, 'ONT_6mA': _ONT,
}


def build_cmd(d):
    out_path = OUT_ROOT / 'features' / d['out']
    _, parts = base.build_cmd(d)
    parts = [base.FEATURIZE if p == base.FEATURIZE else p for p in parts]
    # Rewrite --output to the holdout tree (base.build_cmd points at OUT_ROOT).
    out_idx = parts.index('--output') + 1
    parts[out_idx] = str(out_path)
    parts += ['--exclude-bed', EXCLUDE_BED[d['name']]]
    return out_path, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--mem', default='32G')
    a = ap.parse_args()

    only = set(a.only.split(',')) if a.only else None
    (OUT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    for d in base.DATASETS:
        if only and d['name'] not in only:
            continue
        assert d['name'] in EXCLUDE_BED, f"no holdout BED mapped for {d['name']!r}"
        out_path, cmd = build_cmd(d)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wrap = f"{base.CONDA_INIT} && " + ' '.join(
            f'"{c}"' if ' ' in c else c for c in cmd)
        sbatch = [
            'sbatch', '--parsable',
            '--partition=scavenger', '--account=scavenger', '--qos=scavenger',
            '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=4', f'--mem={a.mem}',
            '--time=04:00:00',
            f'--job-name=refeat15h_{d["name"]}',
            f'--output={OUT_ROOT}/logs/{d["name"]}_%j.out',
            f'--error={OUT_ROOT}/logs/{d["name"]}_%j.out',
            f'--wrap={wrap}',
        ]
        if a.dry_run:
            print(' '.join(sbatch))
            print()
        else:
            r = subprocess.run(sbatch, capture_output=True, text=True)
            jid = r.stdout.strip()
            print(f"{d['name']}: job {jid}")


if __name__ == '__main__':
    main()
