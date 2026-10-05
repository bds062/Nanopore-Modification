"""One module per tool family. Each exposes

    iter_calls(sample: dict, subtool: dict) -> Iterator[pandas.DataFrame]

yielding per-read calls with columns  read_id (str), contig (str), pos (int, 0-based reference),
prob (float, P(modified) for this subtool's codes). Nothing else -- the 10-read selection,
strand assignment and aggregation are done once, in bench.aggregate, for every tool.

Families whose output is already per-site and built from the selected reads (RawMod) expose
iter_sites() instead, yielding contig, pos, strand, score.
"""
import importlib


def get(family: str):
    return importlib.import_module(f'{__name__}.{family}')
