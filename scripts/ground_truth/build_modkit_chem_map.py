#!/usr/bin/env python3
"""Per-site chemistry assignment from the per-modification modkit pileups.

Replaces run_matched_loco.chem_array()'s reference-base heuristic, which types
every C-site positive as whatever single mark BENCH_ORG_CHEMS lists for the
organism. That is wrong for H. pylori (26695 and J99 carry 4mC AND 5mC AND
6mA) and it silently discards A-site positives in organisms whose entry omits
6mA.

THE HARD LIMIT: the depositor shipped one BAM per modification, each basecalled
with a single-mod model that only knows canonical-C vs its own mark. A
methylated C is therefore claimed by whichever model is asked, and 4mC/5mC are
NOT separable from this data:

    organism        4mC-only  5mC-only  both>=80%
    HP26695_WT         3,465       232     15,977
    HPJ99_WT           7,132       130     13,769
    Ecoli_WT           8,278       180     23,105   <- has NO 4mC MTase
    Anabaena_WT       16,625       397      6,482
    Tdenticola_WT      7,212        59         49

E. coli K-12 has dcm (5mC at CCWGG) and no 4mC methyltransferase, yet the 4mC
model reports 8,278 sites the 5mC model does not -- so "one model fired and the
other did not" is evidence only in an organism that actually carries the mark.
Separating 4mC from 5mC properly needs a re-basecall with dorado's joint
4mC_5mC model, where the two compete on one denominator.

Until then this module is deliberately conservative:
  A site, 6mA >= POS_PCT                      -> '6mA'   (no competing mark;
                                                 WGA control: 41,004 vs 1)
  C site, exactly one C-model fires, and the
    organism is documented to carry that mark -> '4mC' / '5mC'
  C site, both fire (or neither is credible)  -> 'untyped'  (dropped from
                                                 training AND test)

Output: one TSV per dataset, 'contig  pos  strand  chem', consumed by
run_matched_loco.chem_array().

Usage:
  python build_modkit_chem_map.py [--out DIR] [--pos-pct 80] [--min-cov 10]
"""
import argparse
import os
from pathlib import Path

GT = Path('/fs/cbcb-scratch/bds062/data/gt_modkit')

# Which C marks each organism is documented to carry. A mark absent here is
# treated as a model artefact no matter how confidently it is called.
#   HP26695 / HPJ99 : 4mC (M.HpyAIV family) and 5mC, plus 6mA
#   Ecoli K-12      : dcm -> 5mC only; dam -> 6mA. No 4mC MTase.
#   Ecoli DM        : dam-/dcm- double mutant -- neither mark
#   Ecoli DM+M.SssI : M.SssI -> 5mC at CpG
#   Anabaena        : 4mC and 5mC both reported
#   Tdenticola      : 4mC; its 5mC model fires at 108 sites total (vs 7,261
#                     4mC), i.e. essentially not at all
C_MARKS = {
    'HP26695_WT_5kHz':     {'4mC', '5mC'},
    'HP26695_WGA_5kHz':    set(),          # whole-genome amplified: no marks
    'HPJ99_WT_5kHz':       {'4mC', '5mC'},
    'Anabaena_WT_5kHz':    {'4mC', '5mC'},
    'Tdenticola_WT_5kHz':  {'4mC'},
    'Ecoli_WT_5kHz':       {'5mC'},
    'Ecoli_DM_5kHz':       set(),
    'Ecoli_DM_MSssI_5kHz': {'5mC'},
}
HAS_6MA = {'HP26695_WT_5kHz', 'HPJ99_WT_5kHz', 'Anabaena_WT_5kHz',
           'Tdenticola_WT_5kHz', 'Ecoli_WT_5kHz'}


def load(path, pos_pct, min_cov):
    """Confident-positive keys (contig, pos, strand) from a modkit bedMethyl."""
    out = set()
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            w = line.rstrip('\n').split('\t')
            if len(w) < 11:
                continue
            try:
                cov, pct = int(w[4]), float(w[10])
            except ValueError:
                continue
            if cov >= min_cov and pct >= pos_pct:
                out.add((w[0], int(w[1]), w[5]))
    return out


def build(ds, pos_pct, min_cov):
    d = GT / ds / 'pileups'
    a = load(d / '4mC.bed', pos_pct, min_cov)
    m = load(d / '5mC.bed', pos_pct, min_cov)
    g = load(d / '6mA.bed', pos_pct, min_cov)
    marks = C_MARKS.get(ds, set())
    rows, stats = [], {'6mA': 0, '4mC': 0, '5mC': 0, 'untyped': 0}
    if ds in HAS_6MA:
        for k in g:
            rows.append((*k, '6mA')); stats['6mA'] += 1
    for k in a | m:
        in4, in5 = k in a, k in m
        if in4 and not in5 and '4mC' in marks:
            c = '4mC'
        elif in5 and not in4 and '5mC' in marks:
            c = '5mC'
        elif in4 and in5 and len(marks) == 1:
            # only one mark is credible for this organism, so the other
            # model firing too is the artefact, not an ambiguity
            c = next(iter(marks))
        else:
            c = 'untyped'
        rows.append((*k, c)); stats[c] += 1
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(GT / 'chem_map'))
    ap.add_argument('--pos-pct', type=float, default=80.0)
    ap.add_argument('--min-cov', type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tot = {}
    for ds in sorted(C_MARKS):
        if not (GT / ds / 'pileups').is_dir():
            print(f'  {ds:<22} no pileups -- skipped'); continue
        rows, stats = build(ds, a.pos_pct, a.min_cov)
        rows.sort()
        with open(out / f'{ds}.tsv', 'w') as fh:
            fh.write('contig\tpos\tstrand\tchem\n')
            for r in rows:
                fh.write('\t'.join(map(str, r)) + '\n')
        tot[ds] = stats
        print(f"  {ds:<22} " + "  ".join(f"{k}={v:,}" for k, v in stats.items()))
    print(f'\nwrote {len(tot)} maps to {out}')


if __name__ == '__main__':
    main()
