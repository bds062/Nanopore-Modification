#!/bin/bash
# Supplementary leave-out table: every row scored by a RawMod model trained without the row's organism or library
# (checkpoints/rawmod_leaveout/), next to the released model, at the same held-out positions.
# Needs 1_run_benchmark.sh (basecalls, read selection, rows, refinement) and 2_score.sh.
#   bash test/evaluation/3_leaveout.sh [--slurm]
set -euo pipefail
B=$(cd "$(dirname "$0")/../benchmark" && pwd)
T=$(python - <<PY
import sys; sys.path.insert(0, '$B')
from bench import models
print(','.join(s for s in models.load() if s.startswith('rawmod_r90_logo_') and not s.endswith('j99tden')))
PY
)
python "$B/rawmod_bench.py" run --tools "$T" --steps tool,aggregate "$@"
if [[ " $* " == *" --slurm "* ]]; then
  echo "When the jobs have finished: python $B/rawmod_bench.py leaveout --holdout r81"
else
  python "$B/rawmod_bench.py" leaveout --holdout r81
fi
