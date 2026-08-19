#!/usr/bin/env python3
"""
Score a saved RawMod checkpoint against an externally-supplied list of
genomic sites (e.g. a collaborator's own scoring run), with ground truth
DERIVED FROM THIS REPO'S OWN authoritative GT bed -- only the (contig, pos)
columns of --sites are used; any label/score/n_reads column already present
in that file is ignored. This is an independent check, not a re-report of
someone else's labels.

Reuses the exact same machinery as every other test set in this repo instead
of reimplementing anything:
  1. deepmod/featurization.py builds a correctly-labeled features.h5 for
     EXACTLY the requested sites (--candidate-bed = the sites, --gt = the
     ground-truth bed), one image per site by default
     (--max-images-per-base 1).
  2. run_pipeline's own inference/position-aggregation/metric code (the same
     evaluate() every loco_<CHEM>/logo_<group> fold in run_matched_loco.py
     uses) scores it.

Usage:
  python test_external_sites.py \
    --sites /fs/cbcb-lab/storm/ernzhang/rawmod_data/ecoli_scores/Ecoli_WT_5kHz_scores.tsv.gz \
    --pod5 /fs/cbcb-lab/storm/bds062/data/benchmark/bacteria/Ecoli_WT_5kHz/pod5 \
    --bam /fs/cbcb-scratch/bds062/results/benchmark_results/Ecoli_WT_5kHz/reads_refined.bam \
    --peaks /fs/cbcb-scratch/bds062/results/benchmark_results/Ecoli_WT_5kHz/peaks_refined.tsv \
    --gt /fs/cbcb-scratch/bds062/data/gt/Ecoli_WT/gt_modified.bed \
    --checkpoint /fs/cbcb-scratch/bds062/results/rawmod_matched_loco/<results_dir>/models/logo_bacteria/logo/best_model.pt \
    --out-dir /fs/cbcb-scratch/bds062/results/rawmod_matched_loco/<results_dir>/external_tests/Ecoli_WT_5kHz

Writes <out-dir>/features.h5 (the freshly featurized+labeled sites),
<out-dir>/metrics.tsv (one row: auroc/f1/etc., same columns as
run_matched_loco.py's metrics/<fold>.tsv), and <out-dir>/scores.tsv
(per-position: contig, pos, predicted score, GT label).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / 'scripts' / 'train', REPO / 'scripts' / 'test', REPO / 'rawmod'):
    sys.path.insert(0, str(_p))

import run_pipeline as R                        # noqa: E402
from score_genome import load_model              # noqa: E402

PYTHON = '/fs/nexus-scratch/bds062/envs/mod/bin/python'
FEATURIZE = str(REPO / 'rawmod' / 'featurization.py')
DEFAULT_LEVEL_TABLE = ('/fs/nexus-scratch/bds062/rawhash2-env/rawhash2-storm/'
                       'extern/local_kmer_models/uncalled_r1041_model_only_means.txt')


def load_sites(path):
    """Read (contig, pos) from an arbitrary tsv/tsv.gz. Uses columns named
    contig/chrom(osome)/ref_name and pos/position/ref_pos if present
    (case-insensitive), else the first two columns positionally. Every other
    column (score, label, n_reads, ...) is dropped -- ground truth comes from
    --gt, never from this file."""
    df = pd.read_csv(path, sep='\t')
    cols = {c.lower(): c for c in df.columns}
    contig_col = next((cols[c] for c in ('contig', 'chrom', 'chromosome', 'ref_name')
                       if c in cols), df.columns[0])
    pos_col = next((cols[c] for c in ('pos', 'position', 'ref_pos')
                    if c in cols), df.columns[1])
    sites = df[[contig_col, pos_col]].drop_duplicates()
    sites.columns = ['contig', 'pos']
    return sites.astype({'pos': np.int64}).sort_values(['contig', 'pos']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sites', required=True,
                    help='tsv/tsv.gz with contig+pos columns (other columns ignored)')
    ap.add_argument('--pod5', required=True)
    ap.add_argument('--bam', required=True)
    ap.add_argument('--peaks', required=True)
    ap.add_argument('--gt', required=True,
                    help="ground-truth BED (ref_name, ref_pos) -- authoritative; "
                         "NOT --sites' own label/score column")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--level-table', default=DEFAULT_LEVEL_TABLE)
    ap.add_argument('--min-reads', type=int, default=12)
    ap.add_argument('--max-reads', type=int, default=15,
                    help="must match the checkpoint's expected height-1 "
                         "(15 for every current strand15 model)")
    ap.add_argument('--max-images-per-base', type=int, default=1,
                    help='1 (default) = exactly one image per site; raise to let '
                         'high-coverage sites emit multiple images (still '
                         'aggregated back to one score per site at eval time)')
    ap.add_argument('--strand', choices=['both', '+', '-'], default='+')
    ap.add_argument('--min-mapq', type=int, default=0)
    ap.add_argument('--half-window', type=int, default=10)
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--skip-featurize', action='store_true',
                    help='reuse an existing --out-dir/features.h5 instead of rebuilding it')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the featurization command and exit without running it')
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / 'features.h5'

    if not a.skip_featurize:
        sites = load_sites(a.sites)
        bed_path = out / 'candidate_sites.bed'
        sites.to_csv(bed_path, sep='\t', header=False, index=False)
        print(f"[test_external_sites] {len(sites):,} unique sites -> {bed_path}", flush=True)

        cmd = [
            PYTHON, FEATURIZE,
            '--pod5', a.pod5, '--bam', a.bam, '--peaks', a.peaks,
            '--output', str(h5_path),
            '--level-table', a.level_table,
            '--gt', a.gt,
            '--candidate-bed', str(bed_path),
            '--min-reads', str(a.min_reads),
            '--max-reads', str(a.max_reads),
            '--min-mapq', str(a.min_mapq),
            '--strand', a.strand,
            '--half-window', str(a.half_window),
            '--L', str(a.L),
        ]
        if a.max_images_per_base:
            cmd += ['--max-images-per-base', str(a.max_images_per_base)]
        print('[test_external_sites] featurizing: ' + ' '.join(cmd), flush=True)
        if a.dry_run:
            return
        subprocess.run(cmd, check=True)
    elif a.dry_run:
        print(f"[test_external_sites] --dry-run + --skip-featurize: would score "
              f"existing {h5_path}", flush=True)
        return
    else:
        print(f"[test_external_sites] --skip-featurize: reusing {h5_path}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    group = R.Group(['ext'], {'ext': str(h5_path)})
    print(f"[test_external_sites] featurized pool: {group.N:,} images "
          f"(pos={int((group.labels > 0).sum()):,} neg={int((group.labels <= 0).sum()):,})",
          flush=True)

    model, arch = load_model(a.checkpoint, device)
    hp = R.HP()
    idx = np.arange(group.N, dtype=np.int64)

    ds = R.make_ds(group, idx, False, hp)
    loader = DataLoader(ds, shuffle=False,
                        **R.make_loader_kwargs(hp.batch, hp.num_workers, device, R._wif))
    yt, yp = R.run_inference(model, loader, device)
    keys = group.source_keys(idx)
    t, p, pos_keys = R.aggregate_by_position(yt, yp, keys)
    metrics = R.compute_metrics(t, p)

    print(f"\n[test_external_sites] RESULTS  n_test={metrics['n_test']:,} "
          f"n_pos={metrics['n_pos']:,}", flush=True)
    print(f"  auroc={metrics['auroc']:.4f}  auprc={metrics['auprc']:.4f}  "
          f"mod_f1={metrics['mod_f1']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
          f"mod_prec={metrics['mod_prec']:.4f}  mod_rec={metrics['mod_rec']:.4f}  "
          f"threshold={metrics['threshold']:.4f}", flush=True)

    cols = ['checkpoint', 'sites_file', 'micro_f1', 'mod_f1', 'unmod_f1', 'macro_f1',
           'mod_prec', 'mod_rec', 'auprc', 'auroc', 'threshold', 'n_pos', 'n_test']
    row = {'checkpoint': a.checkpoint, 'sites_file': a.sites, **metrics}
    metrics_path = out / 'metrics.tsv'
    with open(metrics_path, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        fh.write('\t'.join(f"{row[c]:.6f}" if isinstance(row.get(c), float)
                           else str(row.get(c, '')) for c in cols) + '\n')
    print(f"wrote {metrics_path}", flush=True)

    scores_path = out / 'scores.tsv'
    with open(scores_path, 'w') as fh:
        fh.write('contig\tpos\tscore\tlabel\n')
        for (_fi, contig, pos), label, score in zip(pos_keys, t, p):
            fh.write(f"{contig}\t{pos}\t{score:.6f}\t{int(label)}\n")
    print(f"wrote {scores_path}", flush=True)


if __name__ == '__main__':
    main()
