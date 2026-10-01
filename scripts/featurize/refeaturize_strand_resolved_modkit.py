#!/usr/bin/env python3
"""refeaturize_strand_resolved.py with modkit-derived, confidence-thresholded
ground truth instead of REBASE motif hits.

Two problems this addresses:

1. Motif ground truth makes the reference k-mer partly predictive of the label
   by construction, and it has no negatives at all -- the bacterial sets are
   ~100% positive, which is why they can only ever be curriculum data.
2. The previous modkit typing (mod_types.parse_pileup_dominant) applied NO
   percent-modified threshold, so in SPO1 essentially every A position became a
   "6mA" positive at a median 3.1% modified.

modkit_gt.py therefore keeps only calls that clear >=80% modified (positive) or
fall at <=5% (negative), both at cov>=10 and only where the reference base on
that strand matches the modification. Featurizing the union gives each dataset
real within-library negatives.

Output goes to a separate tree so the motif-based production features are
untouched:
  /fs/cbcb-lab/storm/bds062/rawmod_strand_resolved_modkit/features

Usage:
  python refeaturize_strand_resolved_modkit.py --dry-run
  python refeaturize_strand_resolved_modkit.py [--only NAME,NAME] [--dependency IDS]
"""
import argparse
import subprocess
from pathlib import Path

import refeaturize_strand_resolved as base

GT_MODKIT = Path('/fs/cbcb-scratch/bds062/data/gt_modkit')
OUT = Path('/fs/cbcb-lab/storm/bds062/rawmod_strand_resolved_modkit')

# dataset name -> directory under GT_MODKIT holding gt_/cand_<tag>.bed
GT_DIR = {name: name for name in (
    'HP26695_WT_5kHz', 'HP26695_WGA_5kHz', 'HPJ99_WT_5kHz', 'Ecoli_WT_5kHz',
    'Ecoli_DM_5kHz', 'Ecoli_DM_MSssI_5kHz', 'Anabaena_WT_5kHz',
    'Tdenticola_WT_5kHz')}
GT_DIR.update({'barcode06': 'spo1_bc06', 'barcode07': 'spo1_bc07',
               'barcode01_test': 'spo1_bc01', 'barcode02_train': 'spo1_bc02',
               'barcode03_train': 'spo1_bc03', 'barcode04_train': 'spo1_bc04',
               'barcode05_train': 'spo1_bc05'})

# Families where a treatment and its control share ONE site list
# (make_matched_sites.py), so every confident positive has a same-coordinate
# control image and the curriculum's stage-1 anchors are real matched pairs
# rather than accidental collisions between independently sampled site sets.
MATCHED = {'HP26695_WT_5kHz', 'HP26695_WGA_5kHz', 'Ecoli_WT_5kHz', 'Ecoli_DM_5kHz',
           'Ecoli_DM_MSssI_5kHz', 'barcode06', 'barcode07', 'barcode01_test',
           'barcode02_train', 'barcode03_train', 'barcode04_train', 'barcode05_train'}

# Arms treated as pure controls: every image is label 0. Whole-genome
# amplification and PCR erase base modifications, so the handful of calls that
# still clear 80% there (216 of 797,853 positions in HP26695 WGA) are
# basecaller noise, not biology. Ecoli_DM is NOT in this set: it is a dam-/dcm-
# genetic mutant, and its 5,102 remaining confident positives are real.
CONTROL_ONLY = {'HP26695_WGA_5kHz', 'barcode01_test', 'barcode02_train',
                'barcode03_train', 'barcode04_train', 'barcode05_train'}


def build(d, strand):
    """base.build() with the GT/candidate beds and the output tree swapped."""
    _, cmd = base.build(d, strand)
    tag = 'plus' if strand == '+' else 'minus'
    gtd = GT_MODKIT / GT_DIR[d['name']]
    matched = d['name'] in MATCHED
    cand = gtd / (f'cand_matched_{tag}.bed' if matched else f'cand_{tag}.bed')
    gt = None if d['name'] in CONTROL_ONLY else gtd / f'gt_{tag}.bed'
    out = OUT / 'features' / f"{d['name']}_{tag}.h5"

    new, i = [], 0
    while i < len(cmd):
        c = str(cmd[i])
        nxt = str(cmd[i + 1]) if i + 1 < len(cmd) else ''
        if c == '--output':
            new += ['--output', str(out)]; i += 2; continue
        if c == '--gt':
            new += ['--gt'] + ([str(gt)] if gt else [])
            i += 2 if (nxt and not nxt.startswith('--')) else 1
            continue
        if c == '--candidate-bed':
            i += 2; continue                       # re-added once below
        if c == '--sample-n-sites' and matched:
            i += 2; continue                       # the shared list IS the sample
        new.append(c); i += 1
    if '--gt' not in new:
        new += ['--gt'] + ([str(gt)] if gt else [])
    new += ['--candidate-bed', str(cand)]
    return out, new


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
        if d['name'] not in GT_DIR or (only and d['name'] not in only):
            continue
        for strand in a.strands.split(','):
            out, cmd = build(d, strand)
            if out.exists():
                print(f'[skip] {out.name} exists'); continue
            tag = 'plus' if strand == '+' else 'minus'
            sb = ['sbatch', '--parsable', '--partition=scavenger', '--account=scavenger',
                  '--qos=scavenger', '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8',
                  f"--mem={d['mem']}", '--time=16:00:00', '--exclude=legacy43',
                  f"--job-name=mk_{d['name']}_{tag}",
                  f"--output={OUT}/logs/%x_%j.out", f"--error={OUT}/logs/%x_%j.err"]
            if a.dependency:
                sb.append(f'--dependency=afterok:{a.dependency}')
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
