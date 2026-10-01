#!/usr/bin/env python3
"""
Refeaturize all 13 dataset files used by run_matched_loco.py's matched pool
(build_members()) with two changes vs the originals:
  --max-reads 30 -> 15
  --strand both  -> +   (forward-strand reads only; see rawmod/featurization.py
                         --strand, added because raw nanopore signal is strand-
                         specific -- pooling both strands in one pileup mixes two
                         physically different measurements. See memory:
                         organism-identifiability-root-cause -- strand-pooling
                         behavior was also found to be a major dataset/organism
                         batch-effect fingerprint.)

Everything else (pod5/bam/peaks/gt/candidate-bed/level-table/half-window/L/
min-reads/min-mapq/normalize/max-images-per-base) is reproduced EXACTLY from
the original invocation for each dataset. Original commands were recovered
from pipeline.sh, featurize_barcodes.sh, featurize.sh, gen_features.sh and the
corresponding SLURM logs (see chat history / RESULTS for full provenance per
dataset).

Two further changes (second revamp pass, per user request "no site or
nucleotide bias, double the sites"):
  - HP's original --target-bases AC (restricted candidates to A/C reference
    positions only) is DROPPED. HP is uncapped (uses every candidate.bed
    position), so this is also how its site count grows.
  - barcode06/07 originally omitted --uniform-sampling, so default weighted
    A/C-biased sampling applied (3x G/T); --uniform-sampling is now added to
    match barcode02-05's already-unbiased treatment.
  - --sample-n-sites doubled for every dataset that had an explicit cap
    (barcode06/07, barcode02-05: 4000->8000).
  - HP, barcode01 (test), and the 4 ONT files were already uncapped (using
    100% of eligible sites at current --min-reads/--min-mapq thresholds) --
    left as-is rather than loosening quality thresholds to force more sites.

All 13 datasets reuse already-computed reads_refined.bam/peaks_refined.tsv (or
reads.bam+peaks_refined.tsv for the SPO1/UMCES lineage, which never used a
refined BAM) -- Dorado basecalling and Remora refinement are NOT rerun.

Usage:
  python refeaturize_strand15.py --dry-run      # print sbatch commands only
  python refeaturize_strand15.py                # submit all 13 jobs
  python refeaturize_strand15.py --only HP26695_WT_5kHz,barcode06   # subset
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from rawmod.paths import CONDA_SH, PYTHON, RAWHASH2_DIR, RAWMOD_DATA, RAWMOD_ENV, RAWMOD_LEGACY_DATA, RAWMOD_LEGACY_RESULTS, RAWMOD_RESULTS, RAWMOD_SHARED, RAWMOD_STORE, REPO_DIR  # noqa: E402  (site paths; see paths.env.example)

import argparse
import subprocess
from pathlib import Path

RAWHASH2_LEVEL_TABLE = (f'{RAWHASH2_DIR}/'
                        'extern/local_kmer_models/uncalled_r1041_model_only_means.txt')
ONT_LEVEL_TABLE = f'{RAWMOD_LEGACY_RESULTS}/uncalled_r1041_model_only_means.txt'

FEATURIZE = f'{REPO_DIR}/rawmod/featurization.py'
PYTHON = f'{PYTHON}'
CONDA_INIT = (f'source {CONDA_SH} && '
             f'conda activate {RAWMOD_ENV}')
OUT_ROOT = Path(f'{RAWMOD_RESULTS}/rawmod_full_pipeline4')

UMBC = f'{RAWMOD_SHARED}/umbc-ont-data'
BC_POD5 = f'{UMBC}/pod5_by_barcode/run1_jan31/single_end/high_quality'
BC_BAM = f'{UMBC}/basecalled/run1_jan31/single_end/high_quality'

# ── HP (H. pylori 26695): rawmod/featurization.py, pipeline.sh convention ────
# Original min-reads=25 was set against max-reads=30 (~83% fill). With
# max-reads=15 (this revamp) and --max-images-per-base 1 unchanged, 25 is
# structurally unreachable (15*1 < 25 -- featurization.py now validates this).
# Scale to the same ~80% fill ratio instead: 15 * 0.8 = 12.
# --target-bases AC (original) restricted candidates to A/C reference bases
# only -- a real nucleotide bias. Dropped here per user request (no site or
# nucleotide bias this round); HP is uncapped (all candidate.bed positions
# used), so removing this restriction is also how HP's site count grows.
HP_COMMON = ('--half-window 10 --L 10 --min-reads 12 --min-mapq 30 '
            '--uniform-sampling --max-images-per-base 1')
# HP WGA has NO candidate-bed (genome-wide, all-unmodified control) and was
# previously self-limited to a manageable site count only by strict min-mapq
# 60/min-reads 25. At min-mapq 30/min-reads 12 the eligible pool explodes to
# ~1.65M positions (~186GB tensor, OOMs even at 100G) -- and the downstream
# matched-pool builder (run_matched_loco.py NEG_CAP) subsamples HP negatives
# to 40,000 anyway, so anything past that is wasted compute/storage. Capped
# explicitly, generously above the 40k downstream cap.
HP_WGA_COMMON = HP_COMMON + ' --sample-n-sites 80000'

# ── SPO1/UMCES 2a (deepmod_ont+umces, barcode06/07 -- real WGS positives) ─────
# Original omitted --uniform-sampling -- default weighted sampling gives A/C
# sites 3x the probability of G/T. Added here to match barcode02-05's
# treatment (no base bias). --sample-n-sites doubled 4000->8000 per user
# request.
BC0607_COMMON = ('--min-reads 5 --max-images-per-base 5 --min-mapq 30 --normalize '
                 '--half-window 10 --L 10 --sample-n-sites 8000 --uniform-sampling')

# ── SPO1/UMCES 2b (deepmod_umces, barcode01-05 -- all-unmodified controls) ───
BC_TRAIN_COMMON = ('--min-reads 5 --min-mapq 30 --normalize --max-images-per-base 5 '
                   '--sample-n-sites 8000 --uniform-sampling')
BC_TEST_COMMON = '--min-reads 5 --min-mapq 30 --normalize --max-images-per-base 5'

# ── ONT (deep_modification/results9 lineage) ──────────────────────────────────
# Original used the script default min-mapq (60); standardized to 30 per user
# request (min-mapq 30 for everything, matching the SPO1/UMCES lineage).
ONT_COMMON = '--normalize --max-images-per-base 5 --min-mapq 30'
ONT_ROOT = f'{RAWMOD_LEGACY_RESULTS}'

DATASETS = [
    # -- HP --
    dict(name='HP26695_WT_5kHz',
        # Original path ($RAWMOD_DATA/benchmark/...) no longer
        # exists -- this data now lives under $RAWMOD_STORE (matching
        # refeaturize_benchmark.py's POD5_ROOT for every other curriculum
        # organism); found when a from-scratch rerun of this script failed
        # with "0 reads in pod5" for both HP26695 datasets.
        pod5=f'{RAWMOD_STORE}/data/benchmark/bacteria/HP26695_WT_5kHz/pod5',
        bam=f'{RAWMOD_RESULTS}/benchmark_results/HP26695_WT_5kHz/reads_refined.bam',
        peaks=f'{RAWMOD_RESULTS}/benchmark_results/HP26695_WT_5kHz/peaks_refined.tsv',
        level_table=RAWHASH2_LEVEL_TABLE,
        gt=f'{RAWMOD_DATA}/gt/hpylori_26695/gt_modified.bed',
        candidate=f'{RAWMOD_DATA}/gt/hpylori_26695/candidate.bed',
        extra=HP_COMMON, out='HP26695_WT_5kHz/features.h5'),
    dict(name='HP26695_WGA_5kHz',
        pod5=f'{RAWMOD_STORE}/data/benchmark/bacteria/HP26695_WGA_5kHz/pod5',
        bam=f'{RAWMOD_RESULTS}/benchmark_results/HP26695_WGA_5kHz/reads_refined.bam',
        peaks=f'{RAWMOD_RESULTS}/benchmark_results/HP26695_WGA_5kHz/peaks_refined.tsv',
        level_table=RAWHASH2_LEVEL_TABLE,
        gt=f'{RAWMOD_DATA}/gt/empty.bed',
        candidate=None,
        extra=HP_WGA_COMMON, out='HP26695_WGA_5kHz/features.h5'),
]

for bc in ('06', '07'):
    DATASETS.append(dict(
        name=f'barcode{bc}',
        pod5=f'{BC_POD5}/barcode{bc}.pod5',
        bam=f'{BC_BAM}/barcode{bc}/reads.bam',
        peaks=f'{BC_BAM}/barcode{bc}/peaks_refined.tsv',
        level_table=RAWHASH2_LEVEL_TABLE,
        gt=f'{RAWMOD_RESULTS}/deepmod_ont+umces/gt/barcode{bc}/gt_combined.bed',
        candidate=f'{RAWMOD_RESULTS}/deepmod_ont+umces/gt/barcode{bc}/candidate.bed',
        extra=BC0607_COMMON, out=f'deepmod_ont+umces/barcode{bc}.h5'))

for bc in ('02', '03', '04', '05'):
    DATASETS.append(dict(
        name=f'barcode{bc}_train',
        pod5=f'{BC_POD5}/barcode{bc}.pod5',
        bam=f'{BC_BAM}/barcode{bc}/reads.bam',
        peaks=f'{BC_BAM}/barcode{bc}/peaks_refined.tsv',
        level_table=RAWHASH2_LEVEL_TABLE,
        gt='BARE', candidate=None,
        extra=BC_TRAIN_COMMON, out=f'deepmod_umces/train/barcode{bc}.h5'))

DATASETS.append(dict(
    name='barcode01_test',
    pod5=f'{BC_POD5}/barcode01.pod5',
    bam=f'{BC_BAM}/barcode01/reads.bam',
    peaks=f'{BC_BAM}/barcode01/peaks_refined.tsv',
    level_table=RAWHASH2_LEVEL_TABLE,
    gt='BARE', candidate=None,
    extra=BC_TEST_COMMON, out='deepmod_umces/test/barcode01_test.h5'))

for mod in ('control', '5mC', '5hmC', '6mA'):
    gt_path = ('BARE' if mod == 'control' else
              f'{RAWMOD_LEGACY_DATA}/ont-os/references/all_5mers_{mod}_sites.bed')
    DATASETS.append(dict(
        name=f'ONT_{mod}',
        pod5=f'{RAWMOD_LEGACY_DATA}/ont-os/subset_{mod}/{mod}_rep1.pod5',
        bam=f'{RAWMOD_LEGACY_RESULTS}/event_clustering_{mod}/basecalled/reads_refined.bam',
        peaks=f'{RAWMOD_LEGACY_RESULTS}/event_clustering_{mod}/basecalled/peaks_refined.tsv',
        level_table=ONT_LEVEL_TABLE,
        gt=gt_path, candidate=None,
        extra=ONT_COMMON, out=f'ONT/{mod}.h5'))


def build_cmd(d):
    out_path = OUT_ROOT / 'features' / d['out']
    parts = [
        PYTHON, FEATURIZE,
        '--pod5', d['pod5'],
        '--bam', d['bam'],
        '--peaks', d['peaks'],
        '--output', str(out_path),
        '--level-table', d['level_table'],
        '--max-reads', '15',
        '--strand', '+',
    ]
    if d['gt'] == 'BARE':
        parts += ['--gt']
    elif d['gt'] is not None:
        parts += ['--gt', d['gt']]
    if d['candidate']:
        parts += ['--candidate-bed', d['candidate']]
    parts += d['extra'].split()
    return out_path, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None, help='comma-separated dataset names to run')
    ap.add_argument('--mem', default='32G',
                    help='sbatch --mem (default 32G; whole-genome no-candidate-bed '
                         'datasets like HP WGA can OOM at 32G -- bump for those)')
    a = ap.parse_args()

    only = set(a.only.split(',')) if a.only else None
    (OUT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    for d in DATASETS:
        if only and d['name'] not in only:
            continue
        out_path, cmd = build_cmd(d)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wrap = f"{CONDA_INIT} && " + ' '.join(
            f'"{c}"' if ' ' in c else c for c in cmd)
        sbatch = [
            'sbatch', '--parsable',
            '--partition=scavenger', '--account=scavenger', '--qos=scavenger',
            '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=4', f'--mem={a.mem}',
            '--time=04:00:00',
            f'--job-name=refeat15_{d["name"]}',
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
            print(f"{d['name']:20s} -> job {jid}  out={out_path}")
            if r.returncode != 0:
                print(f"  ERROR: {r.stderr}")


if __name__ == '__main__':
    main()
