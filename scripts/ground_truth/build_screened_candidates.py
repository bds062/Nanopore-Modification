#!/usr/bin/env python3
"""
Build a balanced candidate BED (positives + Dorado-screened negatives) for a
motif-methylated dataset, so featurization emits BOTH classes instead of only
the all-modified motif positions.

Positives  : the motif ground truth (gt_modified.bed), label 1 downstream.
Negatives  : forward reference A/C positions NOT in the GT, that Dorado calls
             CONFIDENTLY UNMODIFIED — i.e. every modification code (6mA 'a',
             5mC 'm', 5hmC 'h') is below --neg-max-frac at >= --min-cov valid
             reads on that position's + strand. A non-motif base that Dorado
             flags as modified (a possible uncharacterised MTase site) is
             therefore EXCLUDED from the negative set rather than mislabelled 0.
             Optionally also hard-exclude any position inside a known methylation
             motif (--exclude-motifs), belt-and-suspenders against imperfect
             Dorado recall (e.g. EcoKI in E. coli).

Featurization keys candidates by (contig, forward-pos) and filters to
--target-bases AC, so we only enumerate FORWARD A/C positions here; a forward-A
position's 6mA status is the + strand 'a' fraction, a forward-C's 5mC/5hmC
status is the + strand 'm'/'h' fractions.

Negatives are sampled uniformly across the genome (no base/positional bias) to
--neg-per-pos x |positives|, and the whole candidate set is capped at
--max-total (positives are downsampled first if they alone exceed the cap).

Usage:
  build_screened_candidates.py --ref REF.fa --gt gt_modified.bed \
      --pileup modkit_pileup.bed --out candidate.bed \
      [--neg-per-pos 1.0] [--max-total 500000] [--neg-max-frac 10.0] \
      [--min-cov 5] [--exclude-motifs GATC,CCWGG] [--seed 42]
"""
import argparse
import re
import sys

import numpy as np

_IUPAC = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'R': '[AG]', 'Y': '[CT]',
          'S': '[GC]', 'W': '[AT]', 'K': '[GT]', 'M': '[AC]', 'B': '[CGT]',
          'D': '[AGT]', 'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'}
_COMP = str.maketrans('ACGTN', 'TGCAN')


def read_fasta(path):
    name, parts = None, []
    seqs = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(parts).upper()
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
    if name is not None:
        seqs[name] = ''.join(parts).upper()
    return seqs


def load_bed_positions(path):
    s = set()
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            c = line.split('\t')
            s.add((c[0], int(c[1])))
    return s


def load_pileup(path):
    """(contig,pos) -> {'cov': max valid cov on +, 'frac': max mod %% on +}.

    Only + strand rows are used (forward A/C candidate typing). We take the max
    modified fraction across all codes at a position: if ANY code fires, the
    site is not a confident negative.
    """
    d = {}
    with open(path) as fh:
        for line in fh:
            c = line.rstrip('\n').split('\t')
            if len(c) < 11 or c[5] != '+':
                continue
            try:
                pos = int(c[1]); cov = int(c[9]); frac = float(c[10])
            except (ValueError, IndexError):
                continue
            key = (c[0], pos)
            e = d.get(key)
            if e is None:
                d[key] = {'cov': cov, 'frac': frac}
            else:
                e['cov'] = max(e['cov'], cov)
                e['frac'] = max(e['frac'], frac)
    return d


def motif_positions(seqs, motifs):
    """Set of (contig,pos) covered by any motif occurrence on either strand."""
    excl = set()
    for contig, seq in seqs.items():
        for m in motifs:
            rx = re.compile(''.join(_IUPAC[b] for b in m))
            rxc = re.compile(''.join(_IUPAC[b] for b in m.translate(_COMP)[::-1]))
            for r in (rx, rxc):
                for mo in r.finditer(seq):
                    for p in range(mo.start(), mo.end()):
                        excl.add((contig, p))
    return excl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--pileup', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--neg-per-pos', type=float, default=1.0)
    ap.add_argument('--max-total', type=int, default=500000)
    ap.add_argument('--neg-max-frac', type=float, default=10.0)
    ap.add_argument('--min-cov', type=int, default=5)
    ap.add_argument('--exclude-motifs', default='')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    seqs = read_fasta(a.ref)
    gt = load_bed_positions(a.gt)
    pile = load_pileup(a.pileup)
    motifs = [m.strip().upper() for m in a.exclude_motifs.split(',') if m.strip()]
    excl = motif_positions(seqs, motifs) if motifs else set()
    print(f"ref contigs={len(seqs)}  gt_positives={len(gt):,}  "
          f"pileup_sites={len(pile):,}  motif_excluded={len(excl):,}", file=sys.stderr)

    # enumerate forward A/C candidate negatives that Dorado confirms unmodified
    negs = []
    n_nocov = n_flagged = n_inmotif = 0
    for contig, seq in seqs.items():
        for pos, base in enumerate(seq):
            if base not in ('A', 'C'):
                continue
            key = (contig, pos)
            if key in gt:
                continue
            if key in excl:
                n_inmotif += 1; continue
            e = pile.get(key)
            if e is None or e['cov'] < a.min_cov:
                n_nocov += 1; continue
            if e['frac'] > a.neg_max_frac:
                n_flagged += 1; continue          # Dorado thinks it's modified
            negs.append(key)
    negs = np.array(negs, dtype=object)
    print(f"eligible negatives={len(negs):,}  (skipped: no/low cov={n_nocov:,}, "
          f"Dorado-flagged-modified={n_flagged:,}, in-motif={n_inmotif:,})",
          file=sys.stderr)

    # balance + cap while PRESERVING the pos:neg ratio. Start from all positives
    # and neg_per_pos x that many negatives, then scale both down together if the
    # total exceeds max_total (so a dataset with more positives than the cap still
    # keeps its negatives instead of collapsing back to 100% positive).
    pos_all = sorted(gt)
    n_pos = len(pos_all)
    n_neg = min(int(round(a.neg_per_pos * n_pos)), len(negs))
    total = n_pos + n_neg
    if total > a.max_total and total > 0:
        scale = a.max_total / total
        n_pos = int(n_pos * scale)
        n_neg = int(n_neg * scale)
    pos_list = ([pos_all[i] for i in sorted(rng.choice(len(pos_all), n_pos, replace=False))]
                if n_pos < len(pos_all) else pos_all)
    neg_sel = ([tuple(negs[i]) for i in sorted(rng.choice(len(negs), n_neg, replace=False))]
               if n_neg > 0 else [])

    all_sites = sorted(set(pos_list) | set(neg_sel))
    with open(a.out, 'w') as fh:
        for contig, pos in all_sites:
            fh.write(f"{contig}\t{pos}\n")
    print(f"WROTE {a.out}: {len(all_sites):,} candidates "
          f"({n_pos:,} positives + {len(neg_sel):,} screened negatives)",
          file=sys.stderr)


if __name__ == '__main__':
    main()
