#!/usr/bin/env python3
"""
Featurize the 7 ONT-basemod-benchmark (Kulkarni et al. 2024) datasets that
already have Dorado+Remora alignment AND ground truth ready, using the
project's current single-strand convention (--strand + --max-reads 15,
matching rawmod_full_pipeline4's strand15 data): Anabaena_WT, Ecoli_DM,
Ecoli_DM_MSssI, Ecoli_WT, Tdenticola_WT, HPJ99_WT, arabidopsis.

NOT included:
  - HP26695_WT/WGA: already the source of the HP data in the existing
    matched pool (see refeaturize_strand15.py) -- would be pure duplication.
  - osativa: ground truth ready, but Dorado/Remora never run on its 34GB of
    pod5 (would need a large highmem GPU basecalling job first) -- deferred.
  - mouse: no ground truth yet (emseq_bam download never succeeded, dir is
    empty) -- blocked until that's resolved.

Reuses EXISTING reads_refined.bam / peaks_refined.tsv / gt_modified.bed /
candidate.bed (produced by the older benchmark pipeline, pipeline/pipeline.sh
+ submit_all.sh) -- only the pod5 path changed (moved to
/fs/cbcb-lab/storm/bds062/data/benchmark) and the featurization flags are new
(--strand + --min-mapq 0 per this round's explicit request, vs the old
pipeline's --min-mapq 60 and both-strand pooling).

Usage:
  python refeaturize_benchmark.py --dry-run
  python refeaturize_benchmark.py
  python refeaturize_benchmark.py --only Anabaena_WT_5kHz,arabidopsis
"""
import argparse
import subprocess
from pathlib import Path

PYTHON = '/fs/nexus-scratch/bds062/envs/mod/bin/python'
FEATURIZE = '/fs/nexus-scratch/bds062/Nanopore-Modification/deepmod/featurization.py'
CONDA_INIT = ('source /nfshomes/bds062/miniconda3/etc/profile.d/conda.sh && '
             'conda activate /fs/nexus-scratch/bds062/envs/mod')

POD5_ROOT = '/fs/cbcb-lab/storm/bds062/data/benchmark'
OLD_RESULTS = '/fs/cbcb-scratch/bds062/results/benchmark_results'
GT_ROOT = '/fs/cbcb-scratch/bds062/data/gt'
LEVEL_TABLE = ('/fs/nexus-scratch/bds062/rawhash2-env/rawhash2-storm/extern/'
              'local_kmer_models/uncalled_r1041_model_only_means.txt')
OUT_ROOT = '/fs/cbcb-scratch/bds062/results/rawmod_full_pipeline4/features/benchmark'

# common flags: single-strand convention (matches rawmod_full_pipeline4's
# strand15 data -- height 16, 15-read pileups), min-mapq=0 per this round's
# explicit request (overrides the old pipeline's --min-mapq 60), no
# --target-bases restriction and --uniform-sampling (no site/nucleotide bias,
# same precedent as refeaturize_strand15.py). No --normalize: matches the
# original benchmark pipeline.sh convention for this same data (HP26695 in
# the existing pool also skips --normalize).
COMMON = ('--half-window 10 --L 10 --max-reads 15 --min-mapq 0 '
         '--strand + --uniform-sampling --max-images-per-base 1')

# (name, pod5_subdir, gt_name, min_reads, sample_n_sites, mem)
# min_reads=12 matches the strand15 HP convention (~80% fill of 15).
# sample_n_sites caps the two datasets with very large candidate pools
# (Ecoli_DM_MSssI: 731,834 candidate lines; arabidopsis: 8.3M) to avoid the
# presample_cap=sample_n_sites*3 OOM already hit once this session for HP WGA.
DATASETS = [
    ('Anabaena_WT_5kHz', 'bacteria/Anabaena_WT_5kHz/pod5', 'anabaena', 12, None, '48G'),
    ('Ecoli_DM_5kHz', 'bacteria/Ecoli_DM_5kHz/pod5', 'Ecoli_DM', 12, None, '48G'),
    ('Ecoli_DM_MSssI_5kHz', 'bacteria/Ecoli_DM_MSssI_5kHz/pod5', 'Ecoli_DM_MSssI', 12, 80000, '96G'),
    ('Ecoli_WT_5kHz', 'bacteria/Ecoli_WT_5kHz/pod5', 'Ecoli_WT', 12, None, '48G'),
    ('Tdenticola_WT_5kHz', 'bacteria/Tdenticola_WT_5kHz/pod5', 'tdenticola', 12, None, '48G'),
    ('HPJ99_WT_5kHz', 'bacteria/HPJ99_WT_5kHz/pod5', 'hpylori_j99', 12, None, '48G'),
    ('arabidopsis', 'arabidopsis/pod5', 'arabidopsis', 12, 100000, '128G'),
]


def build_cmd(name, pod5_sub, gt_name, min_reads, sample_n_sites):
    out_path = f'{OUT_ROOT}/{name}/features.h5'
    parts = [
        PYTHON, FEATURIZE,
        '--pod5', f'{POD5_ROOT}/{pod5_sub}',
        '--bam', f'{OLD_RESULTS}/{name}/reads_refined.bam',
        '--peaks', f'{OLD_RESULTS}/{name}/peaks_refined.tsv',
        '--output', out_path,
        '--level-table', LEVEL_TABLE,
        '--gt', f'{GT_ROOT}/{gt_name}/gt_modified.bed',
        '--candidate-bed', f'{GT_ROOT}/{gt_name}/candidate.bed',
        '--min-reads', str(min_reads),
    ] + COMMON.split()
    if sample_n_sites:
        parts += ['--sample-n-sites', str(sample_n_sites)]
    return out_path, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None, help='comma-separated dataset names to run')
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
            '--time=08:00:00',
            f'--job-name=featbench_{name}',
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
