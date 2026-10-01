#!/usr/bin/env python3
"""Pick a genomic holdout region per organism/reference genome, for a final
retrain that tests generalization to genuinely unseen genome -- not just
unseen chemistry (loco_<CHEM>) or unseen organism group (logo_<group>).

Several curriculum organisms (Ecoli_WT, Anabaena, HPJ99, ...) are also the
organisms external validation later scores against (see the
rawmod-loco-circularity-gap memory: mixed's 0.996 vs loco_5mC's 0.583 on the
SAME E. coli sites is entirely a training-membership effect). A per-organism
genomic holdout gives every one of those organisms a region the model
provably never trained on -- reads included -- so a later `score_genome.py`
/ `test_external_sites.py` run restricted to that region is a real zero-shot
test on an organism otherwise in the training pool.

Policy (target_frac=0.15 of genome length by default), computed from each
reference's own contig list (read from a representative reads_refined.bam
header -- no separate reference FASTA needed):

  1. If a single contig's fraction of the genome falls in
     [0.5, 2.0] x target_frac, hold out that whole contig (the one closest
     to target_frac, if several qualify). Clean: an entire replicon/
     chromosome the model never sees, no interior boundary.
  2. Else, if the genome has one dominant contig (>=50% of total length)
     and its remaining ("minor") contigs sum to at least 5% of the genome,
     hold out ALL of the minor contigs whole (e.g. a bacterium's plasmids)
     without touching the dominant contig.
  3. Else, if there is no dominant contig (several comparably-sized contigs,
     e.g. a small multi-contig synthetic reference), accumulate whole
     contigs in a seeded-random order until the cumulative fraction reaches
     target_frac.
  4. Else (a single-contig genome with no usable minor contigs), carve one
     contiguous window of target_frac * length out of that contig, at a
     seeded-random offset.

Every random choice is seeded from a stable hash of the organism name (not
Python's salted hash()), so this is reproducible independent of run order.

Output:
  <gt-root>/<gt_name>/holdout_region.bed   -- one per organism, BED3
  <out>/genomic_holdout_regions.tsv        -- consolidated report (all organisms)

Usage:
  python select_holdout_regions.py --dry-run
  python select_holdout_regions.py
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pysam

GT_ROOT = '/fs/cbcb-scratch/bds062/data/gt'
BENCH = '/fs/cbcb-scratch/bds062/results/benchmark_results'
UMBC_BAM = ('/fs/cbcb-lab/storm/shared/umbc-ont-data/basecalled/'
           'run1_jan31/single_end/high_quality')
EVENT = '/fs/nexus-scratch/bds062/results'

TARGET_FRAC = 0.15
MIN_MINOR_FRAC = 0.05       # floor for policy 2 (hold out minor contigs as-is)
DOMINANT_FRAC = 0.50        # a contig this big or bigger is "the chromosome"

# (organism, gt_name, representative BAM). One entry per distinct reference
# genome -- e.g. Ecoli_DM/Ecoli_DM_MSssI/Ecoli_WT/HP26695_WT/HP26695_WGA all
# map to the SAME organism/reference and so get the SAME region, computed
# once here and applied to every dataset built from that reference.
ORGANISMS = [
    ('HP26695',      'hpylori_26695', f'{BENCH}/HP26695_WT_5kHz/reads_refined.bam'),
    ('SPO1_UMCES',   'spo1_umces',    f'{UMBC_BAM}/barcode06/reads.bam'),
    ('ONT_all5mers', 'ont_all5mers',  f'{EVENT}/event_clustering_control/basecalled/reads_refined.bam'),
    ('Anabaena',     'anabaena',      f'{BENCH}/Anabaena_WT_5kHz/reads_refined.bam'),
    ('Ecoli',        'Ecoli',         f'{BENCH}/Ecoli_WT_5kHz/reads_refined.bam'),
    ('Tdenticola',   'tdenticola',    f'{BENCH}/Tdenticola_WT_5kHz/reads_refined.bam'),
    ('HPJ99',        'hpylori_j99',   f'{BENCH}/HPJ99_WT_5kHz/reads_refined.bam'),
    ('Arabidopsis',  'arabidopsis',   f'{BENCH}/arabidopsis/reads_refined.bam'),
    ('Human',        'human',         f'{BENCH}/hg001/reads_refined.bam'),
]

# gt-root directories that should receive a copy of the region (every dataset
# built off that reference has its own data/gt/<name>/ directory, and
# featurization is invoked once per dataset, so each needs its own BED path).
GT_ALIASES = {
    'hpylori_26695': ['hpylori_26695'],
    'spo1_umces':    [],  # SPO1/UMCES barcodes have no data/gt/ entry (--gt BARE); see NOTE below
    'ont_all5mers':  [],  # ditto -- ONT all_5mers GT beds live outside data/gt/, see NOTE below
    'anabaena':      ['anabaena'],
    'Ecoli':         ['Ecoli_DM', 'Ecoli_DM_MSssI', 'Ecoli_WT'],
    'tdenticola':    ['tdenticola'],
    'hpylori_j99':   ['hpylori_j99'],
    'arabidopsis':   ['arabidopsis'],
    'human':         ['hg001', 'hg002'],
}


def stable_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def get_contigs(bam_path: str):
    bf = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    return list(zip(bf.references, bf.lengths))


def choose_holdout(name: str, contigs: list, target_frac=TARGET_FRAC):
    total = sum(l for _, l in contigs)
    rng = np.random.default_rng(stable_seed(name))

    # Policy 1: a single contig close to target_frac alone.
    lo, hi = 0.5 * target_frac, 2.0 * target_frac
    band = [(c, l) for c, l in contigs if lo <= l / total <= hi]
    if band:
        c, l = min(band, key=lambda cl: abs(cl[1] / total - target_frac))
        return [(c, 0, l)], f"whole contig {c} ({l/total:.1%} of genome)"

    largest_name, largest_len = max(contigs, key=lambda cl: cl[1])
    dominant = largest_len / total >= DOMINANT_FRAC

    if dominant:
        minor = [(c, l) for c, l in contigs if c != largest_name]
        minor_sum = sum(l for _, l in minor)
        if minor_sum / total >= MIN_MINOR_FRAC:
            regions = [(c, 0, l) for c, l in minor]
            return regions, (f"all {len(minor)} minor contig(s) "
                            f"({minor_sum/total:.1%} of genome), "
                            f"dominant contig {largest_name} untouched")
        # No usable minor contigs -- carve a window from the dominant contig.
        win = int(round(target_frac * total))
        start = int(rng.integers(0, max(1, largest_len - win)))
        return [(largest_name, start, start + win)], (
            f"window in {largest_name} [{start:,}-{start+win:,}) "
            f"({win/total:.1%} of genome)")

    # No dominant contig: accumulate whole contigs in a seeded-random order
    # until the cumulative fraction reaches target_frac.
    order = list(contigs)
    rng.shuffle(order)
    picked, cum = [], 0
    for c, l in order:
        if cum / total >= target_frac:
            break
        picked.append((c, 0, l))
        cum += l
    return picked, f"{len(picked)}/{len(contigs)} contig(s) ({cum/total:.1%} of genome)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help="compute and print, don't write any files")
    ap.add_argument('--target-frac', type=float, default=TARGET_FRAC)
    ap.add_argument('--report', default=str(Path(GT_ROOT) / 'genomic_holdout_regions.tsv'))
    a = ap.parse_args()

    report_rows = []
    for name, gt_name, bam in ORGANISMS:
        contigs = get_contigs(bam)
        regions, desc = choose_holdout(name, contigs, a.target_frac)
        total = sum(l for _, l in contigs)
        held = sum(e - s for _, s, e in regions)
        print(f"{name} ({gt_name}): {desc}  [{held:,}/{total:,} bp = {held/total:.1%}]")
        for c, s, e in regions:
            print(f"    {c}\t{s}\t{e}")
            report_rows.append((name, gt_name, c, s, e, e - s, total))

        if a.dry_run:
            continue

        bed_text = ''.join(f"{c}\t{s}\t{e}\n" for c, s, e in regions)
        for alias in GT_ALIASES.get(gt_name, []):
            out = Path(GT_ROOT) / alias / 'holdout_region.bed'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(bed_text)
            print(f"  wrote {out}")
        if not GT_ALIASES.get(gt_name):
            # No data/gt/<name>/ directory for this reference (BARE-GT
            # datasets); write next to the report instead so the region is
            # still on record and reusable via an explicit --exclude-bed path.
            out = Path(GT_ROOT) / f'{gt_name}_holdout_region.bed'
            out.write_text(bed_text)
            print(f"  wrote {out} (no data/gt/{gt_name}/ dir; pass this path explicitly)")

    if not a.dry_run:
        with open(a.report, 'w') as f:
            f.write("organism\tgt_name\tcontig\tstart\tend\tlength_bp\tgenome_total_bp\n")
            for row in report_rows:
                f.write('\t'.join(str(x) for x in row) + '\n')
        print(f"\nwrote {a.report}")


if __name__ == '__main__':
    main()
