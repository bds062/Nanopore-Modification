#!/usr/bin/env python3
"""refeaturize_strand_resolved.py, unchanged in every flag, except each dataset
gets --exclude-bed pointed at its organism's genomic holdout region and the
output goes to a separate tree.

Why: the mixed fold's train/test split is position-grouped, so train and test
positions sit in the same genomic neighbourhoods and a mixed AUROC measures
"unseen position", not "unseen genome". A model trained on this tree has
provably never seen the held-out region -- reads included, since a read that
overlaps the region is dropped entirely -- so scoring that region afterwards is
a real zero-shot genomic test for organisms that are otherwise in the pool.
Holdout regions come from scripts/ground_truth/select_holdout_regions.py; this
is the strand-resolved counterpart of refeaturize_strand15_holdout.py.

Point a training run at the result with:
  RAWMOD_STRANDRES_ROOT=$RAWMOD_STORE/rawmod_strand_resolved_holdout/features

Usage:
  python refeaturize_strand_resolved_holdout.py --dry-run
  python refeaturize_strand_resolved_holdout.py [--dependency JOBIDS] [--only NAME,NAME]
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from rawmod.paths import RAWMOD_DATA, RAWMOD_STORE  # noqa: E402  (site paths; see paths.env.example)

import argparse
import subprocess
from pathlib import Path

import refeaturize_strand_resolved as base

GT_ROOT = f'{RAWMOD_DATA}/gt'
OUT = Path(f'{RAWMOD_STORE}/rawmod_strand_resolved_holdout')

# One holdout region per REFERENCE, so datasets sharing a genome share a BED
# (see select_holdout_regions.py).
_HP = f'{GT_ROOT}/hpylori_26695/holdout_region.bed'
_SPO1 = f'{GT_ROOT}/spo1_umces_holdout_region.bed'
_ONT = f'{GT_ROOT}/ont_all5mers_holdout_region.bed'
EXCLUDE_BED = {
    'HP26695_WT_5kHz': _HP, 'HP26695_WGA_5kHz': _HP,
    'HPJ99_WT_5kHz': f'{GT_ROOT}/hpylori_j99/holdout_region.bed',
    'Anabaena_WT_5kHz': f'{GT_ROOT}/anabaena/holdout_region.bed',
    'Ecoli_DM_5kHz': f'{GT_ROOT}/Ecoli_DM/holdout_region.bed',
    'Ecoli_DM_MSssI_5kHz': f'{GT_ROOT}/Ecoli_DM_MSssI/holdout_region.bed',
    'Ecoli_WT_5kHz': f'{GT_ROOT}/Ecoli_WT/holdout_region.bed',
    'Tdenticola_WT_5kHz': f'{GT_ROOT}/tdenticola/holdout_region.bed',
    'arabidopsis': f'{GT_ROOT}/arabidopsis/holdout_region.bed',
    'hg001': f'{GT_ROOT}/hg001/holdout_region.bed',
    'hg002': f'{GT_ROOT}/hg002/holdout_region.bed',
    'ONT_control': _ONT, 'ONT_5mC': _ONT, 'ONT_5hmC': _ONT, 'ONT_6mA': _ONT,
    'barcode06': _SPO1, 'barcode07': _SPO1, 'barcode01_test': _SPO1,
    'barcode02_train': _SPO1, 'barcode03_train': _SPO1,
    'barcode04_train': _SPO1, 'barcode05_train': _SPO1,
}


def build(d, strand):
    """base.build() with the output redirected and the holdout region excluded."""
    _, cmd = base.build(d, strand)
    tag = 'plus' if strand == '+' else 'minus'
    out = OUT / 'features' / f"{d['name']}_{tag}.h5"
    cmd = [str(out) if str(c).endswith(f"{d['name']}_{tag}.h5") else c for c in cmd]
    return out, cmd + ['--exclude-bed', EXCLUDE_BED[d['name']]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None)
    ap.add_argument('--strands', default='+,-')
    ap.add_argument('--dependency', default=None,
                    help='colon-separated job ids to wait for (afterany), so this '
                         'does not add NFS load while a training battery is reading '
                         'the same mount')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    (OUT / 'features').mkdir(parents=True, exist_ok=True)
    missing = sorted({b for b in EXCLUDE_BED.values() if not Path(b).exists()})
    if missing:
        raise SystemExit('missing holdout BEDs:\n  ' + '\n  '.join(missing))
    n = 0
    for d in base.D:
        if only and d['name'] not in only:
            continue
        if d['name'] not in EXCLUDE_BED:
            print(f"[skip] {d['name']}: no holdout region mapped")
            continue
        for strand in a.strands.split(','):
            out, cmd = build(d, strand)
            if out.exists():
                print(f'[skip] {out.name} exists')
                continue
            tag = 'plus' if strand == '+' else 'minus'
            sb = ['sbatch', '--parsable', '--partition=scavenger', '--account=scavenger',
                  '--qos=scavenger', '--gres=gpu:0', '--ntasks=1', '--cpus-per-task=8',
                  f"--mem={d['mem']}", '--time=16:00:00',
                  f"--job-name=srh_{d['name']}_{tag}",
                  f"--output={OUT}/logs/%x_%j.out", f"--error={OUT}/logs/%x_%j.err"]
            if a.dependency:
                sb.append(f'--dependency=afterany:{a.dependency}')
            sb += ['--wrap', f'{base.CONDA} && ' + ' '.join(str(c) for c in cmd)]
            if a.dry_run:
                print(' '.join(str(c) for c in cmd), '\n')
            else:
                jid = subprocess.run(sb, capture_output=True, text=True).stdout.strip()
                print(f"submitted {jid}  {d['name']}_{tag}")
            n += 1
    print(f'{n} jobs')


if __name__ == '__main__':
    main()
