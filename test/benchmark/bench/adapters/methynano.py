"""MethyNano: predict_aln.py per-read CSV (step tool; patches/methynano-4f66a60-predict_aln.patch): row_idx, read_id, chrom, strand, pos0,
k_mer, prob_pos, label_pred. pos0 is the 0-based reference position; prob = prob_pos."""
import pandas as pd

from .. import common as C


def iter_calls(sample, subtool):
    src = C.WORK / sample['sample'] / subtool['subtool'] / 'mrb50' / 'calls.csv'
    for df in pd.read_csv(src, usecols=['read_id', 'chrom', 'pos0', 'prob_pos'],
                          dtype={'read_id': str, 'chrom': str}, chunksize=5_000_000):
        yield pd.DataFrame({'read_id': df['read_id'], 'contig': df['chrom'],
                            'pos': df['pos0'].astype('int64'), 'prob': df['prob_pos']})
