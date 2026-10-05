"""RawMod: per-site scores from step tool (bench.steps.rawmod) -- one image per (site, strand) built from exactly the
benchmark's selected reads (featurization.py --read-select), scored by the final checkpoint.

RawMod is not per-read, so this adapter yields sites directly: contig, pos, strand, score, n_reads.
bench.aggregate writes them in the common site format with sum_p = score * K (mean_P = score) and
n_mod = K if score >= 0.5 else 0 (call_freq >= 0.5 <=> score >= 0.5).
"""
import pandas as pd

from .. import common as C

PER_SITE = True


def iter_sites(sample, subtool):
    d = C.WORK / sample['sample'] / subtool['subtool'] / C.TAG
    for name, st in (('plus', '+'), ('minus', '-')):
        p = d / f'scores_{name}.tsv'
        df = pd.read_csv(p, sep='\t', dtype={'contig': str})
        yield pd.DataFrame({'contig': df['contig'], 'pos': df['pos'], 'strand': st,
                            'score': df['score'], 'n_reads': df['n_reads']})
