#!/usr/bin/env python3
"""Featurize the non-motif background candidate sites (candidate_background.bed,
built by pipeline/generate_background_sites.py) for the 6 motif-saturated
bacterial benchmark datasets. No --gt is passed, so featurization.py sets
every label to 0 -- these are genuine unmodified-context negatives, same
base chemistry as the real (motif) candidates, just outside the
recognition motif. Reuses the SAME reads_refined.bam / peaks_refined.tsv
already produced for the positive-only run (refeaturize_benchmark.py) --
only the candidate set and --gt differ.

Usage:
  python featurize_background.py --dry-run
  python featurize_background.py
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from rawmod.paths import CONDA_SH, PYTHON, RAWHASH2_DIR, RAWMOD_DATA, RAWMOD_ENV, RAWMOD_RESULTS, RAWMOD_STORE, REPO_DIR  # noqa: E402  (site paths; see paths.env.example)

import argparse
import subprocess
from pathlib import Path

PYTHON = f'{PYTHON}'
FEATURIZE = f'{REPO_DIR}/rawmod/featurization.py'
CONDA_INIT = (f'source {CONDA_SH} && '
             f'conda activate {RAWMOD_ENV}')

POD5_ROOT = f'{RAWMOD_STORE}/data/benchmark'
OLD_RESULTS = f'{RAWMOD_RESULTS}/benchmark_results'
GT_ROOT = f'{RAWMOD_DATA}/gt'
LEVEL_TABLE = (f'{RAWHASH2_DIR}/extern/'
              'local_kmer_models/uncalled_r1041_model_only_means.txt')
OUT_ROOT = f'{RAWMOD_RESULTS}/rawmod_full_pipeline4/features/benchmark'

COMMON = ('--half-window 10 --L 10 --max-reads 15 --min-mapq 0 '
         '--strand + --uniform-sampling --max-images-per-base 1')

# (name, pod5_subdir, gt_name, min_reads, sample_n_sites, mem)
DATASETS = [
    ('Anabaena_WT_5kHz', 'bacteria/Anabaena_WT_5kHz/pod5', 'anabaena', 12, 3000, '48G'),
    ('Ecoli_DM_5kHz', 'bacteria/Ecoli_DM_5kHz/pod5', 'Ecoli_DM', 12, 3000, '48G'),
    ('Ecoli_DM_MSssI_5kHz', 'bacteria/Ecoli_DM_MSssI_5kHz/pod5', 'Ecoli_DM_MSssI', 12, 3000, '48G'),
    ('Ecoli_WT_5kHz', 'bacteria/Ecoli_WT_5kHz/pod5', 'Ecoli_WT', 12, 3000, '48G'),
    ('Tdenticola_WT_5kHz', 'bacteria/Tdenticola_WT_5kHz/pod5', 'tdenticola', 12, 3000, '48G'),
    ('HPJ99_WT_5kHz', 'bacteria/HPJ99_WT_5kHz/pod5', 'hpylori_j99', 12, 3000, '48G'),
]


def build_cmd(name, pod5_sub, gt_name, min_reads, sample_n_sites):
    out_path = f'{OUT_ROOT}/{name}_background/features.h5'
    parts = [
        PYTHON, FEATURIZE,
        '--pod5', f'{POD5_ROOT}/{pod5_sub}',
        '--bam', f'{OLD_RESULTS}/{name}/reads_refined.bam',
        '--peaks', f'{OLD_RESULTS}/{name}/peaks_refined.tsv',
        '--output', out_path,
        '--level-table', LEVEL_TABLE,
        '--candidate-bed', f'{GT_ROOT}/{gt_name}/candidate_background.bed',
        '--min-reads', str(min_reads),
    ] + COMMON.split()
    if sample_n_sites:
        parts += ['--sample-n-sites', str(sample_n_sites)]
    return out_path, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None

    logdir = Path(OUT_ROOT) / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)

    for name, pod5_sub, gt_name, min_reads, sample_n_sites, mem in DATASETS:
        if only and name not in only:
            continue
        out_path, cmd = build_cmd(name, pod5_sub, gt_name, min_reads, sample_n_sites)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        wrap = f"{CONDA_INIT} && " + ' '.join(cmd)
        sbatch = [
            'sbatch', '--parsable',
            '--partition=scavenger', '--account=scavenger', '--qos=scavenger',
            '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8', f'--mem={mem}',
            '--time=04:00:00',
            f'--job-name=featbg_{name}',
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
