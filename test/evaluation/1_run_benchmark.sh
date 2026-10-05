#!/bin/bash
# Run every tool of the benchmark on every sample: read subsets, the shared Dorado basecall, read selection, ground-truth
# rows, RawMod refinement, each tool, and per-site aggregation. Steps whose output exists are skipped, so the script can
# be rerun after a failure. Tools whose settings are missing are skipped with a message (see rawmod_bench.py config).
#
#   bash test/evaluation/1_run_benchmark.sh                 # locally, in order
#   bash test/evaluation/1_run_benchmark.sh --slurm         # one SLURM job per task (BENCH_SLURM_CPU / BENCH_SLURM_GPU)
#   bash test/evaluation/1_run_benchmark.sh --tools all     # also the UniMeth-FT columns (train_unimeth_ft.sh first)
# Further options: --samples S ..., --steps subset,basecall,..., --dry-run (rawmod_bench.py run --help).
set -euo pipefail
B=$(cd "$(dirname "$0")/../benchmark" && pwd)
python "$B/rawmod_bench.py" run --steps subset,basecall,readsel,rows,refine,tool,aggregate "$@"
