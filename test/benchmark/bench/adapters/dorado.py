"""Dorado: MM/ML tags of $BENCH_WORK/<sample>/<subtool>/calls.bam (step tool).

prob at a read position = sum of ML probabilities over the subtool's codes (codes '*' = every code
the model emits, e.g. 5mC + 5hmC for 5mC_5hmC). A base with no MM entry counts as unmodified.
pysam's modified_bases gives positions in BAM SEQ orientation, so they map onto the reference
through get_aligned_pairs on either strand.
"""
import numpy as np
import pandas as pd
import pysam

from .. import common as C

CHUNK = 2000


def iter_calls(sample, subtool):
    bam = C.WORK / sample['sample'] / subtool['subtool'] / 'calls.bam'
    codes = subtool['codes']
    buf = []
    with pysam.AlignmentFile(str(bam), 'rb') as f:
        for i, r in enumerate(f.fetch(until_eof=True)):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            mb = r.modified_bases
            if not mb:
                continue
            p = {}
            for (_base, _strand, code), lst in mb.items():
                if codes is not None and str(code) not in codes:
                    continue
                for q, ml in lst:
                    p[q] = p.get(q, 0.0) + (ml + 0.5) / 256
            if not p:
                continue
            pairs = np.array(r.get_aligned_pairs(matches_only=True), dtype=np.int64).reshape(-1, 2)
            q2r = dict(zip(pairs[:, 0].tolist(), pairs[:, 1].tolist()))
            qs = [q for q in p if q in q2r]
            if not qs:
                continue
            buf.append(pd.DataFrame({'read_id': r.query_name, 'contig': r.reference_name,
                                     'pos': [q2r[q] for q in qs],
                                     'prob': np.minimum([p[q] for q in qs], 1.0)}))
            if len(buf) >= CHUNK:
                yield pd.concat(buf, ignore_index=True)
                buf = []
    if buf:
        yield pd.concat(buf, ignore_index=True)
