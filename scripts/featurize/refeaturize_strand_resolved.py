#!/usr/bin/env python3
"""Strand-resolved refeaturization: one image per (position, strand).

Why this exists
---------------
The ground-truth BEDs list only (contig, pos).  motif_gt.py searches both
strands for palindromic motifs and records a minus-strand modification at its
COMPLEMENTARY + strand coordinate, and Bismark's XM tag does the same for
bisulfite data.  Nanopore sequences one strand at a time, so a minus-strand
modification does not change the current of a forward read -- yet the previous
(forward-only) featurization attached those labels to forward images anyway.
On the Dam organisms that is half of all positives.

Each dataset is therefore featurized twice:
  plus  : --strand +                (candidate base A/C, labels from *_plus.bed)
  minus : --strand - --orient read  (candidate base T/G, labels from *_minus.bed)

--orient read mirrors the minus-strand image into the sequenced read's own
direction (columns reversed, bases complemented, expected level from the
reverse-complement k-mer), so both files are directly comparable to each other
and to the old forward-only data.

Everything else (window, L, min-reads, min-mapq, sampling, max-reads 15) is
reproduced from refeaturize_strand15.py / refeaturize_benchmark.py.
"""
import argparse
import subprocess
from pathlib import Path

PYTHON = '/fs/nexus-scratch/bds062/envs/mod/bin/python'
FEATURIZE = '/fs/nexus-scratch/bds062/Nanopore-Modification/rawmod/featurization.py'
CONDA = ('source /nfshomes/bds062/miniconda3/etc/profile.d/conda.sh && '
         'conda activate /fs/nexus-scratch/bds062/envs/mod')
OUT = Path('/fs/cbcb-lab/storm/bds062/rawmod_strand_resolved')  # cbcb-scratch is 93% full
GS = '/fs/cbcb-scratch/bds062/data/gt_strand'
RAWHASH2_LT = ('/fs/nexus-scratch/bds062/rawhash2-env/rawhash2-storm/extern/'
               'local_kmer_models/uncalled_r1041_model_only_means.txt')
ONT_LT = '/fs/nexus-scratch/bds062/results/uncalled_r1041_model_only_means.txt'
UMBC = '/fs/cbcb-lab/storm/shared/umbc-ont-data'
BC_POD5 = f'{UMBC}/pod5_by_barcode/run1_jan31/single_end/high_quality'
BC_BAM = f'{UMBC}/basecalled/run1_jan31/single_end/high_quality'
BENCH_POD5 = '/fs/cbcb-lab/storm/bds062/data/benchmark'
HUMAN_POD5 = '/fs/cbcb-scratch/bds062/data/human'
OLD = '/fs/cbcb-scratch/bds062/results/benchmark_results'

HP = '--half-window 10 --L 10 --min-reads 12 --min-mapq 30 --uniform-sampling --max-images-per-base 1'
BC0607 = ('--min-reads 5 --max-images-per-base 5 --min-mapq 30 --normalize '
          '--half-window 10 --L 10 --sample-n-sites 8000 --uniform-sampling')
BCTRAIN = ('--min-reads 5 --min-mapq 30 --normalize --max-images-per-base 5 '
           '--sample-n-sites 8000 --uniform-sampling')
BCTEST = '--min-reads 5 --min-mapq 30 --normalize --max-images-per-base 5'
ONTC = '--normalize --max-images-per-base 5 --min-mapq 30'
BENCH = '--half-window 10 --L 10 --min-mapq 0 --normalize --max-images-per-base 1 --uniform-sampling'

# name, pod5, bam, peaks, level table, gt dir (None = BARE control), extra, mem
D = []
D.append(dict(name='HP26695_WT_5kHz',
              pod5=f'{BENCH_POD5}/bacteria/HP26695_WT_5kHz/pod5',
              bam=f'{OLD}/HP26695_WT_5kHz/reads_refined.bam',
              peaks=f'{OLD}/HP26695_WT_5kHz/peaks_refined.tsv',
              lt=RAWHASH2_LT, gt=f'{GS}/hpylori_26695', cand=True, extra=HP, mem='48G'))
D.append(dict(name='HP26695_WGA_5kHz',
              pod5=f'{BENCH_POD5}/bacteria/HP26695_WGA_5kHz/pod5',
              bam=f'{OLD}/HP26695_WGA_5kHz/reads_refined.bam',
              peaks=f'{OLD}/HP26695_WGA_5kHz/peaks_refined.tsv',
              lt=RAWHASH2_LT, gt=None, cand=False, extra=HP + ' --sample-n-sites 80000', mem='160G'))
for bc in ('06', '07'):
    D.append(dict(name=f'barcode{bc}', pod5=f'{BC_POD5}/barcode{bc}.pod5',
                  bam=f'{BC_BAM}/barcode{bc}/reads.bam',
                  peaks=f'{BC_BAM}/barcode{bc}/peaks_refined.tsv',
                  lt=RAWHASH2_LT, gt=f'{GS}/spo1', cand=True, extra=BC0607, mem='64G'))
for bc in ('02', '03', '04', '05'):
    D.append(dict(name=f'barcode{bc}_train', pod5=f'{BC_POD5}/barcode{bc}.pod5',
                  bam=f'{BC_BAM}/barcode{bc}/reads.bam',
                  peaks=f'{BC_BAM}/barcode{bc}/peaks_refined.tsv',
                  lt=RAWHASH2_LT, gt=None, cand=False, extra=BCTRAIN, mem='48G'))
D.append(dict(name='barcode01_test', pod5=f'{BC_POD5}/barcode01.pod5',
              bam=f'{BC_BAM}/barcode01/reads.bam',
              peaks=f'{BC_BAM}/barcode01/peaks_refined.tsv',
              lt=RAWHASH2_LT, gt=None, cand=False, extra=BCTEST, mem='48G'))
for mod in ('control', '5mC', '5hmC', '6mA'):
    # ONT constructs carry the modification at 256 designed positions on the +
    # strand only, so the minus strand of those sites is genuinely unmodified.
    D.append(dict(name=f'ONT_{mod}',
                  pod5=f'/fs/nexus-scratch/bds062/data/ont-os/subset_{mod}/{mod}_rep1.pod5',
                  bam=f'/fs/nexus-scratch/bds062/results/event_clustering_{mod}/basecalled/reads_refined.bam',
                  peaks=f'/fs/nexus-scratch/bds062/results/event_clustering_{mod}/basecalled/peaks_refined.tsv',
                  lt=ONT_LT,
                  gt=(None if mod == 'control' else
                      f'/fs/nexus-scratch/bds062/data/ont-os/references|all_5mers_{mod}_sites.bed'),
                  cand=False, extra=ONTC, mem='32G'))
for nm, sub, gtd, mem, cap in (
        ('Anabaena_WT_5kHz', f'{BENCH_POD5}/bacteria/Anabaena_WT_5kHz/pod5', 'anabaena', '48G', None),
        ('Ecoli_DM_5kHz', f'{BENCH_POD5}/bacteria/Ecoli_DM_5kHz/pod5', 'Ecoli_DM', '48G', None),
        ('Ecoli_DM_MSssI_5kHz', f'{BENCH_POD5}/bacteria/Ecoli_DM_MSssI_5kHz/pod5', 'Ecoli_DM_MSssI', '96G', 80000),
        ('Ecoli_WT_5kHz', f'{BENCH_POD5}/bacteria/Ecoli_WT_5kHz/pod5', 'Ecoli_WT', '48G', None),
        ('Tdenticola_WT_5kHz', f'{BENCH_POD5}/bacteria/Tdenticola_WT_5kHz/pod5', 'tdenticola', '48G', None),
        ('HPJ99_WT_5kHz', f'{BENCH_POD5}/bacteria/HPJ99_WT_5kHz/pod5', 'hpylori_j99', '48G', None),
        ('arabidopsis', f'{BENCH_POD5}/arabidopsis/pod5', 'arabidopsis', '128G', 100000),
        ('hg001', f'{HUMAN_POD5}/hg001/pod5', 'hg001', '192G', 300000),
        ('hg002', f'{HUMAN_POD5}/hg002/pod5', 'hg002', '192G', 300000)):
    D.append(dict(name=nm, pod5=sub, bam=f'{OLD}/{nm}/reads_refined.bam',
                  peaks=f'{OLD}/{nm}/peaks_refined.tsv', lt=RAWHASH2_LT,
                  gt=f'{GS}/{gtd}', cand=True, min_reads=12,
                  cand_split=gtd in ('arabidopsis', 'hg001', 'hg002'),
                  extra=BENCH + (f' --sample-n-sites {cap}' if cap else ''), mem=mem))


def build(d, strand):
    tag = 'plus' if strand == '+' else 'minus'
    out = OUT / 'features' / f"{d['name']}_{tag}.h5"
    p = [PYTHON, FEATURIZE, '--pod5', d['pod5'], '--bam', d['bam'], '--peaks', d['peaks'],
         '--output', str(out), '--level-table', d['lt'], '--max-reads', '15', '--strand', strand]
    if strand == '-':
        p += ['--orient', 'read']
    if d['gt'] is None:
        p += ['--gt']                                   # control: all labels 0
    elif '|' in d['gt']:                                # ONT: + strand bed only
        root, bed = d['gt'].split('|')
        p += ['--gt'] + ([f'{root}/{bed}'] if strand == '+' else [])
    else:
        p += ['--gt', f"{d['gt']}/gt_{tag}.bed"]
        if d.get('cand'):
            # bisulfite/EM-seq sets have a candidate bed holding high-confidence
            # negatives as well as positives; motif sets have positives only
            cand = (f"{d['gt']}/cand_{tag}.bed" if d.get('cand_split')
                    else f"{d['gt']}/gt_{tag}.bed")
            p += ['--candidate-bed', cand]
    if 'min_reads' in d:
        p += ['--min-reads', str(d['min_reads'])]
    return out, p + d['extra'].split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--strands', default='+,-')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    (OUT / 'features').mkdir(parents=True, exist_ok=True)
    n = 0
    for d in D:
        if only and d['name'] not in only:
            continue
        for strand in a.strands.split(','):
            out, cmd = build(d, strand)
            if out.exists():
                print(f'[skip] {out.name} exists')
                continue
            tag = 'plus' if strand == '+' else 'minus'
            wrap = f'{CONDA} && ' + ' '.join(cmd)
            sb = ['sbatch', '--parsable', '--partition=scavenger', '--account=scavenger',
                  '--qos=scavenger', '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8',
                  f"--mem={d['mem']}", '--time=12:00:00',
                  f"--job-name=sr_{d['name']}_{tag}",
                  f"--output={OUT}/logs/%x_%j.out", f"--error={OUT}/logs/%x_%j.err",
                  '--wrap', wrap]
            if a.dry_run:
                print(' '.join(cmd), '\n')
            else:
                jid = subprocess.run(sb, capture_output=True, text=True).stdout.strip()
                print(f"submitted {jid}  {d['name']}_{tag}")
            n += 1
    print(f'{n} jobs')


if __name__ == '__main__':
    main()
