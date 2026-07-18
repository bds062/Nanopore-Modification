#!/usr/bin/env python
"""Re-chunk a featurized HDF5 so the `tensors` dataset uses 1 image / chunk.

The featurizer writes `tensors` with chunks=(64, 31, 210, 9) + gzip. Reading a
single random image therefore decompresses all 64 images in its chunk (~62
img/s random vs ~3861 img/s contiguous), which is why training preloads the
whole split into RAM (~390 GB -> mem=460G, only fits scarce qos=highmem nodes).

Re-chunking to (1, 31, 210, 9) makes a random read decompress exactly one image,
so DataLoader streaming (preload=False) is fast and RAM drops to a few GB
(mem=48G -> qos=high, plentiful nodes). Values are copied verbatim (still
float32, still gzip) so results are identical to the preloaded runs.

Usage:  rechunk_features.py <features.h5> [--level N] [--verify-n K]
Writes <features.h5>.tmp, verifies, then atomically renames over the original.
The original is left untouched if anything fails.
"""
import argparse
import os
import sys

import h5py
import numpy as np


def rechunk(path: str, level: int, verify_n: int) -> None:
    tmp = path + '.tmp'
    with h5py.File(path, 'r') as s:
        tshape = s['tensors'].shape
        tchunks = s['tensors'].chunks
        if tchunks is not None and tchunks[0] == 1:
            print(f"  [skip] {path} already chunked {tchunks}")
            return
        print(f"  {path}\n    tensors {tshape} chunks {tchunks} -> (1, ...) gzip{level}")
        with h5py.File(tmp, 'w') as d:
            for k, v in s.attrs.items():
                d.attrs[k] = v
            # copy every non-tensor dataset verbatim
            for name in s:
                if name != 'tensors':
                    s.copy(name, d)
            out = d.create_dataset(
                'tensors', shape=tshape, dtype=s['tensors'].dtype,
                chunks=(1,) + tshape[1:], compression='gzip',
                compression_opts=level)
            n = tshape[0]
            B = 4000
            for i in range(0, n, B):
                out[i:i + B] = s['tensors'][i:i + B]
                if i % 40000 == 0:
                    print(f"    {i}/{n}", flush=True)

    # ---- verify tmp against original before replacing ----
    with h5py.File(path, 'r') as s, h5py.File(tmp, 'r') as d:
        assert s['tensors'].shape == d['tensors'].shape, "shape mismatch"
        assert s['tensors'].dtype == d['tensors'].dtype, "dtype mismatch"
        assert d['tensors'].chunks[0] == 1, "chunk not applied"
        # spot-check labels + a random set of full tensors are byte-identical
        assert np.array_equal(s['labels'][:], d['labels'][:]), "labels differ"
        rng = np.random.default_rng(0)
        n = s['tensors'].shape[0]
        idx = np.sort(rng.choice(n, min(verify_n, n), replace=False))
        for i in idx:
            if not np.array_equal(s['tensors'][i], d['tensors'][i]):
                raise AssertionError(f"tensor {i} differs")
    os.replace(tmp, path)
    print(f"  [ok] verified ({len(idx)} tensors + labels) and replaced -> {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--level', type=int, default=1)
    ap.add_argument('--verify-n', type=int, default=200)
    a = ap.parse_args()
    for p in a.paths:
        if not os.path.exists(p):
            print(f"  [miss] {p}", file=sys.stderr)
            continue
        rechunk(p, a.level, a.verify_n)


if __name__ == '__main__':
    main()
