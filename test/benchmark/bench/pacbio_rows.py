#!/usr/bin/env python3
"""Rows with PacBio ground truth (H. pylori J99, T. denticola): per-site ipdSummary calls on public HiFi-kinetics data
(PacBio 2021-11 Microbial 96-plex; Kulkarni et al. 2026 used the same data and flags). Neither organism has a WGA
control, so these are type-one rows (positives and negatives on the same native library).

  positives  m4C (or m6A) calls inside a CONFIRMED motif: a k-mer (k = 4..7, mod at a fixed offset) called at >= 90% of
             its genome occurrences, found greedily (each motif must explain >= 50 not-yet-explained calls). Calls outside
             every confirmed motif (low QV, partial fractions) are neither positive nor negative.
  excluded   motifs listed in SPILLOVER: a C one or two bases from a confirmed 6mA motif, i.e. the same 6mA-induced
             kinetic signal that makes the C of H. pylori 26695 GCATG look modified (its mark is 6mA on the A).
  negatives  the row's base (either strand) with PacBio coverage >= 20 on that strand and no ipdSummary call of ANY type
             (m4C, m6A, modified_base) within +-3 bp on either strand.

Input: $BENCH_DATA/<dataset>/pacbio/ipd/{ref.fa,modbase.gff,kinetics.csv.gz} from test/data/pacbio/run_ipdsummary.sh.
Writes $BENCH_WORK/rows/<row>/{gt.bed,cand.bed,motifs.tsv,gt_motif.tsv.gz} for rows <org>_{4mC,6mA}_pb.
motifs.tsv lists every confirmed motif with its genome occurrences and call rate; gt_motif.tsv.gz maps each positive to its
motif, so the scorer can split AUROC into motifs shared with RawMod's training organisms and novel ones.

  python -m bench.pacbio_rows [--orgs hpj99,tden]
"""
import argparse
import collections
import gzip
import re
from pathlib import Path

from . import common as C
from . import settings as ST

ORGS = {'hpj99': 'd10_hpylori_j99', 'tden': 'd11_tdenticola'}     # row prefix -> dataset folder under $BENCH_DATA
SPILLOVER = {('hpj99', 'm4C'): {'[C]CATGG'}}             # C next to the C[A]TG 6mA motif
RATE, MIN_EXPLAIN, FLANK, MIN_COV = 0.9, 50, 3, 20
RC = str.maketrans('ACGTN', 'TGCAN')


def ref(p):
    s, k = {}, None
    for l in open(p):
        if l[0] == '>':
            k = l[1:].split()[0]; s[k] = []
        else:
            s[k].append(l.strip().upper())
    return {k: ''.join(v) for k, v in s.items()}


def gff(p):
    for l in open(p):
        if l[0] == '#':
            continue
        c = l.rstrip('\n').split('\t')
        a = dict(x.split('=', 1) for x in c[8].split(';'))
        yield c[0], int(c[3]) - 1, c[6], c[2], a


def occurrences(R, km, off):
    out = set()
    for c, s in R.items():
        L = len(s)
        for st, ss in (('+', s), ('-', s.translate(RC)[::-1])):
            for m in re.finditer(f'(?={km})', ss):
                p = m.start() + off
                out.add((c, p if st == '+' else L - 1 - p, st))
    return out


def window(R, c, p, st, f=7):
    s = R[c]
    if p - f < 0 or p + f + 1 > len(s):
        return None
    w = s[p - f:p + f + 1]
    return w if st == '+' else w.translate(RC)[::-1]


def motifs(R, calls, base):
    """Greedy decomposition -> list of (label, kmer, offset, occurrence set, call rate)."""
    left, out = set(calls), []
    W = {k: window(R, *k) for k in calls}
    while True:
        cnt = collections.Counter()
        for key in left:
            w = W[key]
            if w and w[7] == base:
                for k in range(4, 8):
                    for off in range(k):
                        cnt[(w[7 - off:7 - off + k], off)] += 1
        best = None
        for (km, off), n in cnt.most_common(60):
            if n < MIN_EXPLAIN:
                break
            occ = occurrences(R, km, off)
            rate = len(occ & calls) / len(occ)
            if rate >= RATE and (best is None or (n, -len(km)) > best[0]):
                best = ((n, -len(km)), km, off, occ, rate)
        if best is None:
            return out
        _, km, off, occ, rate = best
        out.append((f'{km[:off]}[{km[off]}]{km[off + 1:]}', km, off, occ, rate, len(left & occ)))
        left -= occ


def ipd(org):
    return Path(ST.need('BENCH_DATA', 'PacBio rows')) / ORGS[org] / 'pacbio' / 'ipd'


def coverage(org):
    """(contig, pos0, strand) -> PacBio coverage from ipdSummary's per-base csv (strand 0 = +, 1 = -)."""
    cov = {}
    with gzip.open(ipd(org) / 'kinetics.csv.gz', 'rt') as f:
        next(f)
        for l in f:
            c = l.split(',')
            cov[(c[0].strip('"'), int(c[1]) - 1, '+' if c[2] == '0' else '-')] = int(c[9])
    return cov


def build(org):
    R = ref(ipd(org) / 'ref.fa')
    allc = list(gff(ipd(org) / 'modbase.gff'))
    near = set()                                    # (contig, pos) within FLANK of any call, either strand
    for c, p, _, _, _ in allc:
        near.update((c, q) for q in range(p - FLANK, p + FLANK + 1))
    cov = coverage(org)
    for mod, base, chem in (('m4C', 'C', '4mC'), ('m6A', 'A', '6mA')):
        calls = {(c, p, st) for c, p, st, t, _ in allc if t == mod}
        M = motifs(R, calls, base)
        skip = SPILLOVER.get((org, mod), set())
        pos = {}
        for lab, km, off, occ, rate, n in M:
            if lab in skip:
                continue
            for k in occ & calls:
                pos.setdefault(k, lab)
        neg = set()
        for (c, p, st), n in cov.items():
            if n < MIN_COV or (c, p) in near:
                continue
            b = R[c][p]
            if (st == '+' and b == base) or (st == '-' and b == base.translate(RC)):
                neg.add((c, p, st))
        row = f'{org}_{chem}_pb'
        out = C.WORK / 'rows' / row
        out.mkdir(parents=True, exist_ok=True)
        gt = sorted({(c, p) for c, p, _ in pos})
        cand = sorted(set(gt) | {(c, p) for c, p, _ in neg})
        (out / 'gt.bed').write_text(''.join(f'{c}\t{p}\t{p + 1}\n' for c, p in gt))
        (out / 'cand.bed').write_text(''.join(f'{c}\t{p}\t{p + 1}\n' for c, p in cand))
        with gzip.open(out / 'gt_motif.tsv.gz', 'wt') as f:
            f.write('contig\tpos\tstrand\tmotif\n' + ''.join(f'{c}\t{p}\t{st}\t{m}\n' for (c, p, st), m in sorted(pos.items())))
        with open(out / 'motifs.tsv', 'w') as f:
            f.write('motif\tkmer\toffset\tgenome_occurrences\tcall_rate\tcalls_explained\tused\n')
            for lab, km, off, occ, rate, n in M:
                f.write(f'{lab}\t{km}\t{off}\t{len(occ)}\t{rate:.3f}\t{n}\t{"no (spillover)" if lab in skip else "yes"}\n')
        (out / 'PROVENANCE.txt').write_text(
            f'bench.pacbio_rows {org} {mod}: ipdSummary calls {ORGS[org]}/pacbio/ipd/modbase.gff (SMRT Link 13.1); confirmed motif '
            f'rate>={RATE}, explain>={MIN_EXPLAIN}; negatives cov>={MIN_COV}, no call within {FLANK} bp; spillover {sorted(skip)}\n')
        print(f'{row}: {len(calls):,} {mod} calls, {len(M)} motifs, {len(gt):,} positives, {len(neg):,} negatives '
              f'({len(cand):,} cand)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--orgs', default=','.join(ORGS))
    for org in ap.parse_args().orgs.split(','):
        build(org)


if __name__ == '__main__':
    main()
