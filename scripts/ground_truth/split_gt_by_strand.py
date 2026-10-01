#!/usr/bin/env python3
"""Split a ground-truth BED into strand-resolved plus/minus files.

Existing GT beds list only (contig, pos), pooling both strands: a minus-strand
modification is recorded at its complementary + strand coordinate.  Nanopore
reads one strand at a time, so those two cases are different measurements and
must carry different labels.

The strand is recoverable from the reference base, because a modification sits
on its own base:
    reference A or C at the site -> the modified base is on the + strand
    reference T or G at the site -> the modified base is on the - strand
(6mA/5hmU are A/T; 5mC/5hmC/4mC are C/G.)
"""
import argparse, gzip, os, sys, collections

PLUS = {'A', 'C'}
MINUS = {'T', 'G'}


def seqs(path):
    d, name, buf = {}, None, []
    op = gzip.open(path, 'rt') if path.endswith(('.gz', '.bgz')) else open(path)
    for line in op:
        if line.startswith('>'):
            if name: d[name] = ''.join(buf).upper()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name: d[name] = ''.join(buf).upper()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--bed', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--prefix', default='gt')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    S = seqs(a.ref)
    c = collections.Counter()
    pp = os.path.join(a.outdir, f'{a.prefix}_plus.bed')
    mp = os.path.join(a.outdir, f'{a.prefix}_minus.bed')
    with open(a.bed) as f, open(pp, 'w') as fp, open(mp, 'w') as fm:
        for line in f:
            p = line.split()
            if len(p) < 2: continue
            s = S.get(p[0])
            if s is None:
                c['no_contig'] += 1; continue
            i = int(p[1])
            if i >= len(s):
                c['out_of_range'] += 1; continue
            b = s[i]
            if b in PLUS:
                fp.write(f'{p[0]}\t{i}\n'); c['plus'] += 1
            elif b in MINUS:
                fm.write(f'{p[0]}\t{i}\n'); c['minus'] += 1
            else:
                c['ambiguous_base'] += 1
    print(f'{a.bed}\n  plus={c["plus"]:,}  minus={c["minus"]:,}  '
          f'skipped={c["no_contig"]+c["out_of_range"]+c["ambiguous_base"]:,}', file=sys.stderr)


if __name__ == '__main__':
    main()
