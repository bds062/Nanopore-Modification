"""UniMeth: per-read TSV of step tool (released unimeth infer --output_format tsv).

Columns: chrom, pos, strand, pos_in_strand, read_id, read_pos, type, prob_0, prob_1, label, '.'.
pos is the 0-based reference position; pos = -1 marks calls on read bases with no aligned reference
base (insertions, soft clips) and is dropped. prob = prob_1.
"""
import pandas as pd

from .. import common as C

COLS = ['contig', 'pos', 'strand', 'pos_in_strand', 'read_id', 'read_pos', 'type', 'p0', 'prob', 'label', 'extra']


def iter_calls(sample, subtool):
    src = C.WORK / sample['sample'] / subtool['subtool'] / C.INPUTS / 'calls.tsv'
    for df in pd.read_csv(src, sep='\t', header=None, names=COLS, usecols=['contig', 'pos', 'read_id', 'prob'],
                          dtype={'contig': str, 'pos': 'int64', 'read_id': str, 'prob': 'float64'},
                          chunksize=5_000_000, on_bad_lines='skip'):
        yield df[df['pos'] >= 0]
