#!/usr/bin/env python3
"""Recover REAL modification-chemistry types for the 7 EXTRA_ORGANISMS
(BENCH::) datasets in the results15 all-organism embedding, replacing the
organism-name placeholder used in results15_embedding_allorg.py.

Why this is possible: motif_gt.py (the GT generator for the 6 bacterial
datasets) computes which motif/mod-base matched at generation time, but
gt_modified.bed/candidate.bed only ever write out chrom+position -- the
motif/type identity is discarded. It's recoverable by re-running the same
motif search against the same reference FASTA and re-attaching the type to
each candidate coordinate. arabidopsis needs no motif search -- its GT is
bisulfite/EM-seq derived, so every modified position is 5mC by definition
and every unmodified position is genuinely unmod.

Per-organism chemistry (from submit_all.sh comments + motif_gt.py presets):
  Anabaena_WT_5kHz     Dam-like 6mA @ GATC only                  -> pure 6mA
  Ecoli_DM_5kHz        Dam 6mA @ GATC only                       -> pure 6mA
  Tdenticola_WT_5kHz   Dam-like 6mA @ GATC + TdeI 6mA @ TATAC     -> pure 6mA
  Ecoli_DM_MSssI_5kHz  Dam 6mA @ GATC + MSssI 5mC @ CG            -> mixed, resolved per-position
  Ecoli_WT_5kHz        Dam 6mA @ GATC + Dcm 5mC @ CCWGG           -> mixed, resolved per-position
  HPJ99_WT_5kHz        HpyAIII 6mA + HpyAIV 4mC (REBASE)          -> mixed, resolved per-position
  arabidopsis          bisulfite/EM-seq 5mC                       -> 5mC (mod) / unmod

Usage: python recompute_bench_types.py
Rewrites embeddings_allorg.npz's `types` array in place (backs up the
original once as embeddings_allorg.orig_types.npy) and regenerates the
bytype PCA/t-SNE figures.
"""
import gzip
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
for _p in (REPO / 'analysis' / 'orca_remake', REPO / 'experiments' / 'pipeline4',
           REPO / 'experiments' / 'pipeline1', REPO / 'deepmod', REPO / 'experiments' / 'pipeline3'):
    sys.path.insert(0, str(_p))

import os
os.environ.setdefault('RAWMOD_DATA_GEN', 'strand15')
os.environ.setdefault('EXTRA_ORGANISMS', '1')
os.environ.setdefault('INCLUDE_HUMAN', '1')

import run_matched_loco as ML                                   # noqa: E402
from run_matched_loco import R                                  # noqa: E402

BENCH_REF = '/fs/cbcb-lab/storm/bds062/data/benchmark/references'

# gt_name -> (reference fasta, [(motif_iupac, offset, strand, type), ...])
TYPED_MOTIFS = {
    'anabaena':       (f'{BENCH_REF}/anabaena_sp_PCC7120_ATCC27893.fa.gz',
                       [('GATC', 1, 'both', '6mA')]),
    'Ecoli_DM':       (f'{BENCH_REF}/ecoli.fa.gz',
                       [('GATC', 1, 'both', '6mA')]),
    'Ecoli_DM_MSssI': (f'{BENCH_REF}/ecoli.fa.gz',
                       [('GATC', 1, 'both', '6mA'), ('CG', 0, 'both', '5mC')]),
    'Ecoli_WT':       (f'{BENCH_REF}/ecoli.fa.gz',
                       [('GATC', 1, 'both', '6mA'), ('CCWGG', 1, 'both', '5mC')]),
    'tdenticola':     (f'{BENCH_REF}/treponema_denticola_ATCC35405.fa.gz',
                       [('GATC', 1, 'both', '6mA'), ('TATAC', 1, '+', '6mA'),
                        ('GTATA', 3, '+', '6mA')]),
    'hpylori_j99':    (f'{BENCH_REF}/hpylori_J99_ATCC700824.fa.gz',
                       [('GTNNNNNNAC', 1, '+', '6mA'), ('TCNNNNNNNGC', 1, '+', '4mC')]),
}

# BENCH:: org name (as used in pool/orgs) -> gt_name key above (arabidopsis handled separately)
ORG_TO_GT = {
    'Anabaena_WT_5kHz':    'anabaena',
    'Ecoli_DM_5kHz':       'Ecoli_DM',
    'Ecoli_DM_MSssI_5kHz': 'Ecoli_DM_MSssI',
    'Ecoli_WT_5kHz':       'Ecoli_WT',
    'Tdenticola_WT_5kHz':  'tdenticola',
    'HPJ99_WT_5kHz':       'hpylori_j99',
}

_IUPAC = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'R': '[AG]', 'Y': '[CT]',
         'S': '[GC]', 'W': '[AT]', 'K': '[GT]', 'M': '[AC]', 'B': '[CGT]',
         'D': '[AGT]', 'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'}
_COMP = str.maketrans('ACGT', 'TGCA')


def iupac_to_regex(motif):
    return ''.join(_IUPAC.get(c.upper(), c) for c in motif)


def revcomp(seq):
    return seq.translate(_COMP)[::-1]


def open_fasta(path):
    return gzip.open(path, 'rt') if path.endswith(('.gz', '.bgz')) else open(path)


def build_type_lookup(gt_name):
    ref_path, motifs = TYPED_MOTIFS[gt_name]
    lut = {}
    n_collisions = 0
    chrom, seq_parts = None, []

    def flush():
        nonlocal n_collisions
        if chrom is None:
            return
        seq = ''.join(seq_parts).upper()
        for motif_str, offset, strand, mtype in motifs:
            rx = re.compile(iupac_to_regex(motif_str), re.IGNORECASE)
            for m in rx.finditer(seq):
                key = (chrom, m.start() + offset)
                if key in lut and lut[key] != mtype:
                    n_collisions += 1
                lut[key] = mtype
            if strand == 'both':
                rc = revcomp(seq)
                rc_len = len(rc)
                for m in rx.finditer(rc):
                    fwd_pos = rc_len - (m.start() + offset) - 1
                    key = (chrom, fwd_pos)
                    if key in lut and lut[key] != mtype:
                        n_collisions += 1
                    lut[key] = mtype

    with open_fasta(ref_path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                chrom = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        flush()
    print(f"  {gt_name}: {len(lut):,} typed positions from {ref_path.split('/')[-1]}"
         f"{f' ({n_collisions} collisions)' if n_collisions else ''}", flush=True)
    return lut


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/'
                    'results15/embedding_clustering/embeddings_allorg.npz')
    a = ap.parse_args()
    npz_path = Path(a.npz)
    d = np.load(npz_path, allow_pickle=True)
    rep, orgs, types, mods, idx = d['rep'], d['orgs'], d['types'].copy(), d['mods'], d['idx']

    backup = npz_path.with_name('embeddings_allorg.orig_types.npy')
    if not backup.exists():
        np.save(backup, d['types'])
        print(f"backed up original types -> {backup}")

    print("Building pool (for contig/ref_pos lookup on the saved idx)...", flush=True)
    members = ML.build_members()
    pool = R.Group(list(members), members)
    contig = pool.contig[idx]
    refpos = pool.ref_pos[idx]

    print("Building motif->type lookups per bacterial dataset...", flush=True)
    luts = {gt: build_type_lookup(gt) for gt in set(ORG_TO_GT.values())}

    n_resolved, n_unresolved = 0, 0
    unresolved_orgs = {}
    for i in range(len(idx)):
        org = orgs[i]
        if org in ('arabidopsis', 'hg001', 'hg002'):
            # Real bisulfite/EM-seq GT, not motif-derived -- mod is 5mC by
            # definition (that's what bisulfite/EM-seq measures), no motif
            # search needed.
            types[i] = '5mC' if mods[i] == 'mod' else 'unmod'
            n_resolved += 1
        elif org in ORG_TO_GT:
            key = (contig[i], int(refpos[i]))
            t = luts[ORG_TO_GT[org]].get(key)
            if t is not None:
                types[i] = t
                n_resolved += 1
            else:
                n_unresolved += 1
                unresolved_orgs[org] = unresolved_orgs.get(org, 0) + 1
        # else: ONT/SPO1/HP -- already has a real chem type, leave untouched

    print(f"\nResolved {n_resolved:,} bench-organism points to real types; "
         f"{n_unresolved:,} unresolved (motif didn't match at that coordinate)")
    if unresolved_orgs:
        print(f"  unresolved by org: {unresolved_orgs}")
    print("\nFinal type distribution:")
    vals, counts = np.unique(types, return_counts=True)
    for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
        print(f"  {v:22} {c:,}")

    np.savez_compressed(npz_path, rep=rep, orgs=orgs, types=types, mods=mods, idx=idx)
    print(f"\noverwrote {npz_path} with real types")


if __name__ == '__main__':
    main()
