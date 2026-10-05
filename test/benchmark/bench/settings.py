"""Where the benchmark's external tools and data live on this machine.

Every key is read, in order, from: the environment, then the RawMod site file ($RAWMOD_PATHS_FILE, default
~/.config/rawmod/paths.env, see paths.env.example at the repository root), then $BENCH_SITE_FILE (default
test/benchmark/config/site.env). config/site.example.env lists every key with a description.

A key that is not set resolves to None; the step that needs it stops with the key's name (settings.need), so a
missing tool is reported where it matters instead of at start-up.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent            # test/benchmark
TEST = HERE.parent                                       # test
REPO = TEST.parent                                       # repository root
DATA_DIR = TEST / 'data'                                 # fixed benchmark definition (read sets, rows, held-out positions)
DEFAULTS = {'BENCH_WORK': str(TEST / 'work'), 'BENCH_DATA': str(DATA_DIR), 'SAMTOOLS': 'samtools', 'POD5': 'pod5'}

KEYS = {
    # data and outputs
    'BENCH_WORK': 'output root: one directory per sample, tables and figures (default test/work)',
    'BENCH_DATA': 'where the downloaded datasets live (default test/data, where test/data/download_*.sh put them)',
    'BENCH_SAMPLES': 'optional: an existing sample sheet (sample, bam, pod5, ref) to use instead of building one',
    # basecalling / alignment
    'DORADO': 'dorado 1.4.0 binary (basecalls with move tables, Dorado modification models)',
    'DORADO_MODELS': 'directory holding dna_r10.4.1_e8.2_400bps_sup@v5.0.0 and its modification models',
    'DORADO_092_DIR': 'dorado 0.9.2 bin/ directory with dna_r10.4.1_e8.2_400bps_hac@v4.2.0 inside (MethyNano only)',
    'SAMTOOLS': 'samtools >= 1.16 (default: samtools on PATH)',
    'POD5': 'pod5 command-line tool, for read subsets (default: pod5 on PATH)',
    # RawMod
    'RAWMOD_PYTHON': 'python of the environment RawMod is installed in (pip install -e .)',
    'RAWHASH2_DIR': 'RawHash2 checkout: test/scripts/refine_moves_remora.py and the R10.4.1 k-mer level table',
    'REFINE_PYTHON': 'python of an environment with ont-remora >= 3.3, pod5, pysam (the refinement step)',
    # baselines
    'UNIMETH': 'unimeth command (UniMeth v0.3.3)',
    'UNIMETH_CHECKPOINTS': 'directory with the released unimeth_r10.4.1_5kHz_{5mC,6mA}.pt',
    'UNIMETH_FT_PYTHONPATH': 'optional: UniMeth source with patches/unimeth-11215d4-finetune.patch applied (UniMeth-FT columns)',
    'UNIMETH_FT_CHECKPOINTS': 'optional: directory with the UniMeth-FT checkpoints {5hmC,4mC,5hmU}/final.pt (evaluation/train_unimeth_ft.sh)',
    'DEEPMOD2': 'DeepMod2 entry script (deepmod2), run with RAWMOD_PYTHON unless DEEPMOD2_PYTHON is set',
    'DEEPMOD2_PYTHON': 'optional: python for DeepMod2',
    'ROCKFISH': 'rockfish command (r10.4.1 branch; needs FlashAttention, Ampere or newer GPU)',
    'ROCKFISH_MODEL': 'rf_5kHz.ckpt',
    'METHYNANO_DIR': 'MethyNano checkout with patches/methynano-4f66a60-predict_aln.patch applied',
    'METHYNANO_PYTHON': 'optional: python for MethyNano',
    # figures
    'BENCH_FONTS': 'optional: folder of .ttf files for the figures (the paper used Times New Roman)',
    'BENCH_PAPER_FIGDIR': 'optional: a second folder the figures are also written to',
    # cluster
    'BENCH_SLURM_CPU': 'sbatch arguments for CPU steps, e.g. "--partition=cpu --mem=64G --time=06:00:00"',
    'BENCH_SLURM_GPU': 'sbatch arguments for GPU steps, e.g. "--partition=gpu --gres=gpu:1 --mem=64G --time=08:00:00"',
}


def _read_env_file(path):
    vals = {}
    p = Path(path).expanduser()
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                vals[k.strip().removeprefix('export ').strip()] = os.path.expandvars(v.strip().strip('"\''))
    return vals


def _load():
    site = _read_env_file(os.environ.get('BENCH_SITE_FILE', HERE / 'config' / 'site.env'))
    paths = _read_env_file(os.environ.get('RAWMOD_PATHS_FILE', Path.home() / '.config' / 'rawmod' / 'paths.env'))
    return {k: os.environ.get(k) or paths.get(k) or site.get(k) or DEFAULTS.get(k) for k in KEYS}


S = _load()


def get(key, default=None):
    return S.get(key) or default


def need(key, why=''):
    v = S.get(key)
    if not v:
        raise SystemExit(f'{key} is not set ({KEYS.get(key, "")}){": needed for " + why if why else ""}. '
                         f'Set it in the environment or in test/benchmark/config/site.env '
                         f'(template: config/site.example.env).')
    return v


def expand(text):
    """Replace ${KEY} by the value of setting KEY (config files refer to installed models this way)."""
    import re
    def sub(m):
        return need(m.group(1), f'expanding {text}') if m.group(1) in KEYS else os.environ.get(m.group(1), m.group(0))
    return re.sub(r'\$\{(\w+)\}', sub, text)


def level_table():
    return f"{need('RAWHASH2_DIR', 'RawMod refinement')}/extern/local_kmer_models/uncalled_r1041_model_only_means.txt"


def report():
    """(key, value or None, description) for every key: `rawmod_bench.py config`."""
    return [(k, S.get(k), d) for k, d in KEYS.items()]
