#!/usr/bin/env python3
"""
orca_remake — assign a per-site modification-TYPE string to every image in
barcode06.h5 / barcode07.h5 (SPO1 native WGS).

The h5 files carry only a binary modified/unmodified label (label_semantics=
'binary_modified_vs_unmodified'); which of 5mC/5hmC/6mA/5hmU a modified site
actually is has to be reconstructed by joining (contig, pos) against modkit's
bedMethyl pileup (dominant code) plus a T-scan of the reference (every T is
5hmU, SPO1 hypermodifies all T genome-wide). This is exactly the logic
experiments/pipeline1/mod_types.py already implements for LOMO -- reused here
verbatim rather than reimplemented, so typing stays identical to what the
model's own training/eval already relies on.

Writes one .npy of dtype '<U6' per input h5, parallel to h['labels'], with
values in {'5mC','5hmC','6mA','5hmU','unmod','untyped'}:
  - label==0                     -> 'unmod'
  - label==1, in mod_map          -> the mapped type
  - label==1, not in mod_map      -> 'untyped' (excluded downstream)
"""
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path('/fs/nexus-scratch/bds062/Nanopore-Modification')
sys.path.insert(0, str(REPO / 'experiments' / 'pipeline1'))
from mod_types import build_umces_mod_map, _decode  # noqa: E402

UMCES_REF = '/fs/cbcb-lab/storm/shared/umbc-ont-data/ref/SPO1_FJ230960.1.fasta'
UMCES_PILEUPS = [
    '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/modbam/barcode06_pileup.bed',
    '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/modbam/barcode07_pileup.bed',
]
H5_FILES = {
    'SPO1_bc06': '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/features/barcode06.h5',
    'SPO1_bc07': '/fs/cbcb-scratch/bds062/results/deepmod_ont+umces/features/barcode07.h5',
}
OUT_DIR = Path('/fs/cbcb-scratch/bds062/results/orca_remake/data/type_labels')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building UMCES modification-type map from modkit pileups + T-scan...")
    mod_map = build_umces_mod_map(UMCES_PILEUPS, UMCES_REF, min_cov=10)
    print(f"  {len(mod_map):,} typed (contig,pos) keys")

    for name, h5_path in H5_FILES.items():
        with h5py.File(h5_path, 'r') as h:
            ref_names = _decode(np.asarray(h['ref_names'][:]))
            ref_pos = np.asarray(h['ref_pos'][:]).astype(np.int64)
            labels = np.asarray(h['labels'][:])

        types = np.full(len(labels), 'untyped', dtype='<U7')
        types[labels == 0] = 'unmod'
        n_mod = int((labels > 0).sum())
        n_found = 0
        for i in np.nonzero(labels > 0)[0]:
            key = (ref_names[i], int(ref_pos[i]))
            t = mod_map.get(key)
            if t is not None:
                types[i] = t
                n_found += 1

        out = OUT_DIR / f'{name}_types.npy'
        np.save(out, types)
        vals, counts = np.unique(types, return_counts=True)
        dist = dict(zip(vals, counts))
        print(f"{name}: n={len(labels):,}  modified={n_mod:,}  typed={n_found:,}  "
              f"({100*n_found/max(n_mod,1):.1f}% of modified sites typed)")
        print(f"  distribution: {dist}")
        print(f"  wrote {out}")


if __name__ == '__main__':
    main()
