"""Site-specific locations used by the training, featurization and analysis scripts.

No machine-specific path is stored in this repository. Each location is read from
an environment variable of the same name or, failing that, from a KEY=VALUE file
at ``$RAWMOD_PATHS_FILE`` (default ``~/.config/rawmod/paths.env``); see
``paths.env.example`` at the repository root for the list. A location that is not
set resolves to the literal string ``$NAME``, so a command built from it fails with
the variable's name in the error message (or is expanded by the shell, if the
variable is exported there).
"""
import os
import sys
from pathlib import Path

REPO_DIR = str(Path(__file__).resolve().parents[1])
PYTHON = sys.executable

_NAMES = (
    'RAWMOD_RESULTS',          # training runs, benchmark outputs, evaluation results
    'RAWMOD_DATA',             # pod5, references and ground-truth BEDs
    'RAWMOD_STORE',            # bulk storage (feature trees, large pod5 sets)
    'RAWMOD_SHARED',           # shared lab data (e.g. SPO1/UMCES runs, basecallers)
    'RAWMOD_SCRATCH',          # scratch root (logs, helper scripts, dorado models)
    'RAWMOD_LEGACY_RESULTS',   # results of the earlier ONT-construct pipeline
    'RAWMOD_LEGACY_DATA',      # ONT open-data subsets and references
    'RAWMOD_TOOLS',            # standalone tool installs (e.g. dorado)
    'RAWMOD_ENV',              # conda environment with rawmod's dependencies
    'RAWMOD_ENVS',             # directory holding the conda environments
    'CONDA_BASE',              # conda installation
    'CONDA_SH',                # $CONDA_BASE/etc/profile.d/conda.sh
    'RAWHASH2_DIR',            # RawHash2 checkout (refinement script, k-mer level tables)
)


def _load_file():
    path = Path(os.environ.get('RAWMOD_PATHS_FILE',
                               Path.home() / '.config' / 'rawmod' / 'paths.env'))
    vals = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                vals[k.strip().removeprefix('export ').strip()] = os.path.expandvars(v.strip().strip('"\''))
    return vals


_file = _load_file()
for _n in _NAMES:
    globals()[_n] = os.environ.get(_n) or _file.get(_n) or f'${_n}'

LEVEL_TABLE = os.environ.get('RAWMOD_LEVEL_TABLE') or _file.get('RAWMOD_LEVEL_TABLE') or \
    f'{RAWHASH2_DIR}/extern/local_kmer_models/uncalled_r1041_model_only_means.txt'  # noqa: F821
