"""RawMod models as benchmark columns.

config/rawmod_models.tsv names every RawMod checkpoint the benchmark knows. Each entry becomes a subtool of family
'rawmod' -- 'rawmod' for the entry named rawmod (the paper's model), 'rawmod_<name>' for any other -- so it is
featurized, scored, aggregated and tabulated exactly like the paper's RawMod column:

  name   checkpoint (absolute, or relative to the repository)   mask_bases (0|1)   legacy_ch9 (0|1)   marks

mask_bases = 1 for checkpoints trained with base identity blanked (PILEUP_MASK_BASES); legacy_ch9 = 1 for models
trained before the ch9 fix (score_genome.py --legacy-ch9); marks = the modifications the model is trained on (bold
cells / "own rows"; default all five). Image height (reads per image) is read from the checkpoint.

`rawmod_bench.py rawmod --checkpoint PATH --name NAME` adds an entry and runs it; `register()` is that step.
"""
from __future__ import annotations

from pathlib import Path

from .settings import HERE, REPO

REGISTRY = HERE / 'config' / 'rawmod_models.tsv'
ALL_MARKS = '5mC,5hmC,6mA,4mC,5hmU'
HEADER = '#name\tcheckpoint\tmask_bases\tlegacy_ch9\tmarks\n'


def subtool_name(name: str) -> str:
    return 'rawmod' if name == 'rawmod' else f'rawmod_{name}'


def _resolve(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else REPO / q


def load() -> dict:
    """subtool -> {'name', 'subtool', 'checkpoint' (Path), 'mask_bases', 'legacy_ch9', 'marks'}"""
    out = {}
    if not REGISTRY.exists():
        return out
    for line in open(REGISTRY):
        if not line.strip() or line.startswith('#'):
            continue
        c = line.rstrip('\n').split('\t') + [''] * 5
        name = c[0]
        out[subtool_name(name)] = {'name': name, 'subtool': subtool_name(name), 'checkpoint': _resolve(c[1]),
                                   'mask_bases': c[2] == '1', 'legacy_ch9': c[3] == '1', 'marks': c[4] or ALL_MARKS}
    return out


def register(name: str, checkpoint: str, mask_bases=False, legacy_ch9=False, marks=ALL_MARKS):
    """Add (or replace) a registry entry; returns its subtool name."""
    if not name.replace('_', '').isalnum():
        raise SystemExit(f'model name {name!r}: letters, digits and _ only')
    ck = Path(checkpoint).expanduser().resolve()
    if not ck.is_file():
        raise SystemExit(f'checkpoint not found: {ck}')
    lines = [l for l in (open(REGISTRY).readlines() if REGISTRY.exists() else [HEADER])
             if l.startswith('#') or not l.strip() or l.split('\t', 1)[0] != name]
    shown = ck.relative_to(REPO) if ck.is_relative_to(REPO) else ck      # repo checkpoints stay portable
    lines.append(f'{name}\t{shown}\t{int(mask_bases)}\t{int(legacy_ch9)}\t{marks}\n')
    REGISTRY.write_text(''.join(lines))
    return subtool_name(name)


def reads_per_image(checkpoint: Path) -> int:
    """Reads per pileup image the checkpoint was trained with: its positional table has one row per read plus one
    (results81 mixed.pt: pos (1, 16, 96) <-> --max-reads 15)."""
    import torch
    raw = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    sd = raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw else raw
    return int(sd['pos'].shape[1]) - 1 if 'pos' in sd else 15
