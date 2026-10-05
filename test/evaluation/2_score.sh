#!/bin/bash
# Tables and figures of the benchmark, at the positions the released RawMod model never trained on (--holdout r81) and,
# for comparison, at all sites. Output: $BENCH_WORK/results_<tag>/rawmod/ (see test/README.md, "Outputs").
#   bash test/evaluation/2_score.sh
set -euo pipefail
B=$(cd "$(dirname "$0")/../benchmark" && pwd)
python "$B/rawmod_bench.py" score --holdout r81 "$@"
python "$B/rawmod_bench.py" score "$@"
