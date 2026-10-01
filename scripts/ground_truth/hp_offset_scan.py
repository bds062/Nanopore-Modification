#!/usr/bin/env python3
"""Which base in each H. pylori 26695 motif is actually modified?

Three motif_gt presets point at a base the chemistry forbids (CTTCAAG offset 6 is
a G but labelled 6mA; TCTTC offset 3 is a T but labelled 4mC).  HP26695 WT vs WGA
is a matched pair over the same coordinates, so the modified base is simply the
offset whose read-median current differs most between the two libraries -- the
measurement behind Fig. 1b, reused here per motif offset.

Both libraries' images span +-10 bases, so every candidate offset is already
covered by the existing features; nothing needs re-featurizing.
"""
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from rawmod.paths import RAWMOD_RESULTS, RAWMOD_STORE  # noqa: E402  (site paths; see paths.env.example)

import gzip, re, sys
import numpy as np, h5py

F = f'{RAWMOD_RESULTS}/rawmod_full_pipeline4/features'
REF = f'{RAWMOD_STORE}/data/benchmark/references/hpylori_26695.fa.gz'
HALF, L, W = 10, 10, 21
MOTIFS = [('GCATG', '4mC', 1), ('CTTCAAG', '6mA', 6), ('TCTTC', '4mC', 3)]
COMP = str.maketrans('ACGT', 'TGCA')


def genome():
    seq, name = [], None
    with gzip.open(REF, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                name = line[1:].split()[0]
            else:
                seq.append(line.strip())
    return name, ''.join(seq).upper()


def profile(path, n_genome, cap=None):
    """mean read-median (observed - expected) per genomic position."""
    s = np.zeros(n_genome, np.float64); c = np.zeros(n_genome, np.int64)
    with h5py.File(path, 'r') as f:
        n = f['tensors'].shape[0] if cap is None else min(cap, f['tensors'].shape[0])
        rp = f['ref_pos'][:n]
        B = 2000
        for a in range(0, n, B):
            b = min(a + B, n)
            t = f['tensors'][a:b, :, :, 0].astype(np.float32)      # (B,16,210)
            ref = t[:, 0, :]                                        # expected level
            reads = t[:, 1:, :]
            cov = np.abs(reads).sum(-1) > 0                         # (B,15)
            d = reads - ref[:, None, :]                             # observed-expected
            d = d.reshape(d.shape[0], d.shape[1], W, L).mean(-1)    # per base
            for i in range(d.shape[0]):
                k = cov[i]
                if not k.any():
                    continue
                med = np.median(d[i][k], axis=0)                    # (21,)
                lo = int(rp[a + i]) - HALF
                for w in range(W):
                    p = lo + w
                    if 0 <= p < n_genome:
                        s[p] += med[w]; c[p] += 1
            print(f'  {b}/{n}', end='\r', flush=True)
    out = np.full(n_genome, np.nan)
    ok = c > 0
    out[ok] = s[ok] / c[ok]
    return out, c


if __name__ == '__main__':
    name, seq = genome()
    N = len(seq)
    print(f'genome {name} {N:,} bp', flush=True)
    wt, cwt = profile(f'{F}/HP26695_WT_5kHz/features.h5', N)
    print(f'\nWT positions with data: {(cwt>0).sum():,}', flush=True)
    wga, cwga = profile(f'{F}/HP26695_WGA_5kHz/features.h5', N)
    print(f'\nWGA positions with data: {(cwga>0).sum():,}', flush=True)
    # one image already carries a 15-read median, so >=1 observation per library is
    # enough once thousands of motif occurrences are averaged together
    both = (cwt >= 1) & (cwga >= 1)
    signed = np.where(both, wt - wga, np.nan)
    print(f'positions comparable (>=1 obs each): {both.sum():,}')
    bg_mean = np.nanmean(signed); bg_sd = np.nanstd(signed)
    print(f'background: mean(WT-WGA) = {bg_mean:+.4f}  sd = {bg_sd:.4f}  '
          f'mean|WT-WGA| = {np.nanmean(np.abs(signed)):.4f}\n')
    diff = signed

    for motif, mod, cur in MOTIFS:
        rx = re.compile(motif.replace('N', '[ACGT]'))
        for strand in ('+', '-'):
            pat = motif if strand == '+' else motif.translate(COMP)[::-1]
            starts = [m.start() for m in re.finditer(pat.replace('N', '[ACGT]'), seq)]
            if not starts:
                continue
            starts = np.array(starts)
            print(f'{motif} [{mod}] {strand} strand: {len(starts):,} occurrences')
            for j in range(len(motif)):
                p = starts + j
                p = p[(p >= 0) & (p < N)]
                v = diff[p]
                base = pat[j]
                n = int(np.isfinite(v).sum())
                if n < 20:
                    print(f'   offset {j} ({base}): too few comparable positions (n={n})')
                    continue
                m = np.nanmean(v)
                # z of the offset mean against the genome-wide background
                z = (m - bg_mean) / (bg_sd / np.sqrt(n))
                flag = '  <-- current preset offset' if (strand == '+' and j == cur) else ''
                mark = '  ***' if abs(z) > 8 else ''
                print(f'   offset {j} ({base}): mean(WT-WGA) = {m:+.4f}  z = {z:+7.1f}  '
                      f'(n={n:,}){flag}{mark}')
            print()
