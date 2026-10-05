"""RockFish: per-read predictions.tsv of step tool (rockfish inference, rf_5kHz.ckpt): read_id, pos, prob.

pos indexes the read as sequenced (verified on the E. coli M.SssI library, where every CpG is methylated, and by the
coordinate check in bench.aggregate), so a reverse read's call at read position p sits at BAM SEQ index L-1-p; it is
projected onto the reference through that read's primary alignment in the sample's basecall (sub_bam). Probabilities outside [0, 1] are logits -> sigmoid.
"""
import numpy as np
import pandas as pd
import pysam

from .. import common as C


def iter_calls(sample, subtool):
    aln = {}
    with pysam.AlignmentFile(sample['sub_bam'], 'rb') as b:
        for r in b.fetch(until_eof=True):
            if r.is_unmapped or r.is_secondary or r.is_supplementary:
                continue
            m = np.full(r.query_length, -1, dtype=np.int64)
            pr = np.array(r.get_aligned_pairs(matches_only=True), dtype=np.int64).reshape(-1, 2)
            m[pr[:, 0]] = pr[:, 1]
            aln[r.query_name] = (r.reference_name, r.is_reverse, m)
    src = C.WORK / sample['sample'] / subtool['subtool'] / C.INPUTS / 'predictions.tsv'
    for df in pd.read_csv(src, sep='\t', dtype={'read_id': str}, chunksize=5_000_000):
        out = []
        for rid, g in df.groupby('read_id', sort=False):
            a = aln.get(rid)
            if a is None:
                continue
            ctg, rev, m = a
            q = g['pos'].to_numpy(np.int64)
            if rev:
                q = len(m) - 1 - q
            ok = (q >= 0) & (q < len(m))
            ref = np.full(len(q), -1, np.int64)
            ref[ok] = m[q[ok]]
            keep = ref >= 0
            p = g['prob'].to_numpy(float)[keep]
            p = np.where((p < 0) | (p > 1), 1 / (1 + np.exp(-p)), p)
            out.append(pd.DataFrame({'read_id': rid, 'contig': ctg, 'pos': ref[keep], 'prob': p}))
        if out:
            yield pd.concat(out, ignore_index=True)
