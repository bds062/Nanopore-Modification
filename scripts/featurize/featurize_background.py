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
import argparse
import subprocess
from pathlib import Path

PYTHON = '/fs/nexus-scratch/bds062/envs/mod/bin/python'
FEATURIZE = '/fs/nexus-scratch/bds062/Nanopore-Modification/rawmod/featurization.py'
CONDA_INIT = ('source /nfshomes/bds062/miniconda3/etc/profile.d/conda.sh && '
             'conda activate /fs/nexus-scratch/bds062/envs/mod')

POD5_ROOT = '/fs/cbcb-lab/storm/bds062/data/benchmark'
OLD_RESULTS = '/fs/cbcb-scratch/bds062/results/benchmark_results'
GT_ROOT = '/fs/cbcb-scratch/bds062/data/gt'
LEVEL_TABLE = ('/fs/nexus-scratch/bds062/rawhash2-env/rawhash2-storm/extern/'
              'local_kmer_models/uncalled_r1041_model_only_means.txt')
OUT_ROOT = '/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4/features/benchmark'

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
