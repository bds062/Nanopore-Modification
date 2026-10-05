# Evaluation

We assume the datasets are downloaded ([`../data`](../data/README.md)) and the tools are configured
([`../README.md`](../README.md#configuration)). Run the scripts from the repository root, in this order:

| Script | What it does | Output |
|---|---|---|
| `train_unimeth_ft.sh` (optional) | fine-tunes UniMeth on the released RawMod model's training positions (5hmC, 4mC, 5hmU) | `$UNIMETH_FT_CHECKPOINTS/<mod>/final.pt` |
| `1_run_benchmark.sh [--slurm] [--tools all]` | read subsets, basecalls, read selection, rows, refinement, every tool, per-site aggregation | `$BENCH_WORK/<sample>/` |
| `2_score.sh` | nulls, tables and figures at held-out positions and at all sites | `$BENCH_WORK/results_<tag>/rawmod/` |
| `3_leaveout.sh [--slurm]` | the leave-out checkpoints on every sample, and the supplementary leave-out table | `tables/benchmark_leaveout.tex` in the held-out results folder |

`--tools all` adds the UniMeth-FT columns to the default tools (RawMod, Dorado, UniMeth, DeepMod2, Rockfish,
MethyNano). Every step skips work that is already done, so a script can be rerun after a failure. With `--slurm`, each
task is one SLURM job, chained by dependencies; run the next script when the jobs have finished.
