#!/usr/bin/env python3
"""
The foreign-motif negative control.

Premise: every supervised bacterial modification caller is trained on motif-derived
labels, where the label is a near-deterministic function of the sequence (Dam methylates
~100% of GATC, so an unmethylated GATC does not exist in E. coli). A model can therefore
score well by learning the MOTIF rather than the SIGNAL, and standard evaluation never
catches it because it never includes a genome where the motif is present but unmethylated.

The control: score whole-genome-amplified (unmodified) DNA from an organism whose motif
repertoire differs from the training organism's. Every genome contains GATC; only some
methylate it. H. pylori does not, and amplification removes everything anyway — so every
GATC call on H. pylori WGA is a false positive attributable to sequence context.

The statistic is deliberately scorer-relative:

    enrichment = P(called modified | foreign motif) / P(called modified | background)

so a scorer with a high overall false-positive rate is not penalised for calibration,
only for *context bias*. enrichment ~= 1 means no memorisation.

Handles both our per-site score tables and modkit bedMethyl (Dorado), so every tool is
measured the same way.

Usage:
  foreign_motif_control.py --ref hpylori.fa.gz --out-dir out/ \
      --score  "RawMod pipeline1 (signal labels)=p1_scores.tsv.gz" \
      --score  "RawMod pipeline2 (motif labels)=p2_scores.tsv.gz" \
      --pileup "Dorado=dorado_modkit_pileup.bed" \
      --foreign-motif GATC --foreign-offset 1 \
      --native-motifs GCATG:1,TCTTC:3
"""
import argparse
import gzip
from pathlib import Path

import numpy as np


def read_fasta(path):
    s = []
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        for line in fh:
            if not line.startswith('>'):
                s.append(line.strip())
    return ''.join(s).upper()


def load_score_table(path):
    """our per-site table -> {pos: P(modified)} (single-contig genomes)"""
    d = {}
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        fh.readline()
        for line in fh:
            c = line.rstrip('\n').split('\t')
            d[int(c[1])] = float(c[2])
    return d


def load_modkit_pileup(path, codes=('a',), min_cov=5):
    """modkit bedMethyl -> {pos: max percent-modified across `codes` on the + strand}.

    Column layout (0-indexed): 0 contig, 1 start, 3 code, 5 strand, 9 valid cov,
    10 percent modified.
    """
    d = {}
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        for line in fh:
            c = line.rstrip('\n').split('\t')
            if len(c) < 11 or c[5] != '+' or c[3] not in codes:
                continue
            try:
                pos = int(c[1]); cov = int(c[9]); frac = float(c[10])
            except ValueError:
                continue
            if cov < min_cov:
                continue
            d[pos] = max(d.get(pos, 0.0), frac / 100.0)
    return d


def motif_positions(seq, motif, offset):
    """0-based positions of the modified base for every occurrence, both strands."""
    comp = str.maketrans('ACGT', 'TGCA')
    rc = motif.translate(comp)[::-1]
    out = set()
    for m, off in ((motif, offset), (rc, len(motif) - 1 - offset)):
        start = 0
        while True:
            i = seq.find(m, start)
            if i < 0:
                break
            out.add(i + off)
            start = i + 1
    return out


def analyse(name, scores, foreign, native_sets, thresh=0.5):
    keys = np.fromiter(scores.keys(), dtype=np.int64)
    vals = np.fromiter(scores.values(), dtype=np.float64)
    inf = np.isin(keys, list(foreign))
    nat = np.zeros(len(keys), dtype=bool)
    for s in native_sets.values():
        nat |= np.isin(keys, list(s))
    bg = ~(inf | nat)

    def rate(mask):
        return float(np.mean(vals[mask] > thresh)) if mask.sum() else float('nan')

    r_inf, r_bg = rate(inf), rate(bg)
    enr = r_inf / r_bg if (r_bg and np.isfinite(r_bg) and r_bg > 0) else float('inf')
    row = {'scorer': name, 'n_foreign': int(inf.sum()), 'n_bg': int(bg.sum()),
           'foreign_rate': r_inf, 'bg_rate': r_bg, 'enrichment': enr,
           'foreign_mean': float(np.mean(vals[inf])) if inf.sum() else float('nan'),
           'bg_mean': float(np.mean(vals[bg])) if bg.sum() else float('nan')}
    for mname, s in native_sets.items():
        m = np.isin(keys, list(s))
        row[f'native_{mname}_rate'] = rate(m)
        row[f'native_{mname}_n'] = int(m.sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--score', action='append', default=[], help='"Label=path.tsv.gz"')
    ap.add_argument('--pileup', action='append', default=[], help='"Label=pileup.bed"')
    ap.add_argument('--pileup-codes', default='a')
    ap.add_argument('--foreign-motif', default='GATC')
    ap.add_argument('--foreign-offset', type=int, default=1)
    ap.add_argument('--native-motifs', default='',
                    help='comma list MOTIF:OFFSET of the organism\'s REAL motifs')
    ap.add_argument('--threshold', type=float, default=0.5)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    seq = read_fasta(a.ref)
    foreign = motif_positions(seq, a.foreign_motif, a.foreign_offset)
    native = {}
    for spec in [s for s in a.native_motifs.split(',') if s.strip()]:
        m, off = spec.split(':')
        native[m] = motif_positions(seq, m, int(off))
    print(f"genome {len(seq):,} bp   foreign {a.foreign_motif}: {len(foreign):,} sites   "
          f"native: {[(k, len(v)) for k, v in native.items()]}\n")

    rows = []
    for spec in a.score:
        label, path = spec.split('=', 1)
        rows.append(analyse(label, load_score_table(path), foreign, native, a.threshold))
    for spec in a.pileup:
        label, path = spec.split('=', 1)
        sc = load_modkit_pileup(path, codes=tuple(a.pileup_codes.split(',')))
        rows.append(analyse(label, sc, foreign, native, a.threshold))

    hdr = (f"{'scorer':44} {'foreign':>9} {'background':>11} {'ENRICH':>8}  "
           f"{'n_foreign':>9}")
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f"{r['scorer']:44} {r['foreign_rate']:>8.3%} {r['bg_rate']:>10.3%} "
              f"{r['enrichment']:>7.1f}x {r['n_foreign']:>9,}")

    cols = sorted({k for r in rows for k in r})
    tsv = out / 'foreign_motif_control.tsv'
    with open(tsv, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
    print(f"\nwrote {tsv}")

    # figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        names = [r['scorer'] for r in rows]
        enr = [r['enrichment'] for r in rows]
        cols_ = ['#2ca25f' if e < 2 else '#de2d26' for e in enr]
        y = np.arange(len(names))
        ax.barh(y, enr, color=cols_, height=0.6)
        ax.axvline(1.0, color='#111827', ls='--', lw=1.2)
        ax.text(1.05, -0.75, 'no context bias', fontsize=9, color='#6b7280')
        for yi, e in zip(y, enr):
            ax.text(e, yi, f"  {e:.1f}x", va='center', fontsize=10, fontweight='bold')
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(f'false-positive enrichment at {a.foreign_motif} '
                      f'(relative to each scorer\'s own background)')
        ax.set_title('Foreign-motif negative control — unmodified (WGA) DNA',
                     loc='left', fontweight='bold')
        ax.grid(axis='x', color='#eef2f7'); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(out / 'foreign_motif_control.png', dpi=300)
        print(f"wrote {out / 'foreign_motif_control.png'}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == '__main__':
    main()
