"""DeepMod2: output.per_read of step tool (deepmod2 detect, model bilstm_r10.4.1_5khz_v5.0).

Columns by name: read_name, chromosome, ref_position_before, ref_position, read_position, strand,
methylation_score, ... ref_position_before is the 0-based reference position (NA when the read base
is not aligned, dropped); prob = methylation_score.
"""
import pandas as pd

from .. import common as C


def iter_calls(sample, subtool):
    src = C.WORK / sample['sample'] / subtool['subtool'] / C.INPUTS / 'output.per_read'
    for df in pd.read_csv(src, sep='\t', usecols=['read_name', 'chromosome', 'ref_position_before', 'methylation_score'],
                          dtype={'read_name': str, 'chromosome': str}, na_values=['NA'], chunksize=5_000_000):
        df = df.dropna(subset=['ref_position_before'])
        yield pd.DataFrame({'read_id': df['read_name'], 'contig': df['chromosome'],
                            'pos': df['ref_position_before'].astype('int64'), 'prob': df['methylation_score']})
