#!/usr/bin/env python3
"""
UMCES per-modification typing for leave-one-modification-out (LOMO).

The UMCES featurized H5s carry only binary labels (modified/unmodified), and
barcode06/07 mix all modifications in one file. To hold out a single
modification for LOMO we must attribute each *modified* position to one of:

    5hmU   — every forward reference-T position (SPO1 hypermodifies all T)
    5mC    — modkit code 'm' dominant at a C position
    5hmC   — modkit code 'h' dominant at a C position
    6mA    — modkit code 'a' dominant at an A position

This mirrors how the ground truth was built (`scripts/extract_gt_barcode.py`:
gt_combined = all T + modkit-modified C/A). We type each position by its
*dominant* modkit code (largest modified fraction among m/h/a), which is robust
to the exact pileup fraction threshold. T takes precedence (matches the GT).

Modkit bedMethyl columns used (0-indexed): col0=contig, col1=start(0-based),
col3=mod code, col9=valid coverage, col10=percent modified (0-100).

Public API
----------
  build_umces_mod_map(pileup_paths, ref_fasta, min_cov=10) -> {(contig,pos): type}
  positions_of_type(ref_names, ref_pos, mod_map, mod_type) -> np.ndarray[bool]
"""

from pathlib import Path

import numpy as np

# SAM/modkit modification codes → our modification-type names
CODE_TO_TYPE = {'m': '5mC', 'h': '5hmC', 'a': '6mA'}
UMCES_MODS = ('5mC', '5hmC', '6mA', '5hmU')


def scan_t_positions(fasta_path: str) -> set:
    """Return {(contig, 0-based pos)} for every forward reference T (=5hmU)."""
    t_sites = set()
    with open(fasta_path) as fh:
        contig, offset = None, 0
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                contig = line[1:].split()[0]
                offset = 0
            elif contig is not None:
                for i, base in enumerate(line):
                    if base.upper() == 'T':
                        t_sites.add((contig, offset + i))
                offset += len(line)
    return t_sites


def parse_pileup_dominant(pileup_paths, min_cov: int = 10) -> dict:
    """
    For each (contig,pos), return the dominant modkit modification code among
    m/h/a (the one with the largest percent-modified), requiring valid coverage
    >= min_cov. Returns {(contig,pos): 'm'|'h'|'a'}.
    """
    best: dict = {}       # (contig,pos) -> (best_frac, code)
    for path in pileup_paths:
        with open(path) as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                cols = line.rstrip('\n').split('\t')
                if len(cols) < 11:
                    continue
                code = cols[3]
                if code not in CODE_TO_TYPE:
                    continue
                try:
                    cov  = int(cols[9])
                    frac = float(cols[10])       # percent 0-100
                except ValueError:
                    continue
                if cov < min_cov:
                    continue
                key = (cols[0], int(cols[1]))
                cur = best.get(key)
                if cur is None or frac > cur[0]:
                    best[key] = (frac, code)
    return {k: v[1] for k, v in best.items()}


def build_umces_mod_map(pileup_paths, ref_fasta: str,
                        min_cov: int = 10) -> dict:
    """
    Build {(contig, 0-based pos): modification_type} for UMCES.

    T positions -> '5hmU' (precedence). Other positions -> dominant modkit code
    mapped through CODE_TO_TYPE. Positions with neither are absent from the map
    (treated as untyped by positions_of_type).
    """
    if isinstance(pileup_paths, (str, Path)):
        pileup_paths = [pileup_paths]

    t_sites  = scan_t_positions(ref_fasta)
    dominant = parse_pileup_dominant(pileup_paths, min_cov=min_cov)

    mod_map: dict = {}
    for key, code in dominant.items():
        mod_map[key] = CODE_TO_TYPE[code]
    # T precedence — overwrite any modkit call at a T position with 5hmU
    for key in t_sites:
        mod_map[key] = '5hmU'
    return mod_map


def _decode(names):
    out = []
    for v in names:
        out.append(v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v))
    return out


def positions_of_type(ref_names, ref_pos, mod_map: dict,
                      mod_type: str) -> np.ndarray:
    """
    Boolean mask over featurized positions selecting those typed `mod_type`.

    ref_names / ref_pos are the per-image arrays from the H5 (image-level;
    the mask is therefore image-level and aligns with labels/tensors).
    """
    names = _decode(np.asarray(ref_names))
    pos   = np.asarray(ref_pos).astype(np.int64)
    mask  = np.zeros(len(pos), dtype=bool)
    for i, (nm, p) in enumerate(zip(names, pos)):
        if mod_map.get((nm, int(p))) == mod_type:
            mask[i] = True
    return mask


def type_counts(ref_names, ref_pos, labels, mod_map: dict) -> dict:
    """Count modified (label>0) featurized positions per modification type."""
    names = _decode(np.asarray(ref_names))
    pos   = np.asarray(ref_pos).astype(np.int64)
    lab   = np.asarray(labels) > 0
    counts = {m: 0 for m in UMCES_MODS}
    counts['untyped_mod'] = 0
    for nm, p, is_mod in zip(names, pos, lab):
        if not is_mod:
            continue
        t = mod_map.get((nm, int(p)))
        if t in counts:
            counts[t] += 1
        else:
            counts['untyped_mod'] += 1
    return counts


if __name__ == '__main__':
    # Smoke/inspection: build the map and print per-type modified counts for bc06/07.
    import h5py
    REF = '/fs/cbcb-lab/storm/shared/umbc-ont-data/ref/SPO1_FJ230960.1.fasta'
    PILEUPS = [
        '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/modbam/barcode06_pileup.bed',
        '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/modbam/barcode07_pileup.bed',
    ]
    H5S = [
        '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/features/barcode06.h5',
        '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/features/barcode07.h5',
    ]
    mm = build_umces_mod_map(PILEUPS, REF)
    print(f"mod_map entries: {len(mm):,}")
    from collections import Counter
    print("map type distribution:", dict(Counter(mm.values())))
    for h in H5S:
        with h5py.File(h, 'r') as hf:
            c = type_counts(hf['ref_names'][:], hf['ref_pos'][:], hf['labels'][:], mm)
        print(f"{Path(h).name}: {c}")
