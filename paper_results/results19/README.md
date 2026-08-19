# results19 — final, paper-reported models

Recipe: `RAWMOD_DATA_GEN=strand15 EXTRA_ORGANISMS=1 INCLUDE_HUMAN=1 SUPCON_DIM=128
SUPCON_WEIGHT=1.0 SUPCON_TEMP=0.20 CURRICULUM=1 CURRICULUM_EPOCHS=15 SAD_DIM=32
SAD_WEIGHT=1.0 SAD_ETA=1.0 BCE_WEIGHT=1.0` — see the repo README's "Train + test"
section for the full launch command.

9 folds: `mixed`, `loco_{5hmU,4mC,6mA,5mC,5hmC}`, `logo_{bacteria,plant,mammal}`.

## Status

- `mixed`, `loco_5hmC`, `loco_5hmU`, `logo_bacteria`, `logo_plant`, `logo_mammal`:
  copied unchanged from `results17_temp0.20` — these 6 folds are unaffected by the
  BENCH:: chemistry-leak fix (5hmC/5hmU are never present in the curriculum
  organisms; the LOGO folds hold out organisms, not chemistries, and already
  excluded held-out groups from `extra_idx`).
- `loco_6mA`, `loco_5mC`, `loco_4mC`: **rerun with the leak fix** (see
  `run_matched_loco.py`'s `BENCH_ORG_CHEMS`/`clean_extra` — organisms
  biologically carrying the held-out chemistry are now excluded from stage-2
  training, not just absent from the core matched pool under that label).
  Launched 2026-08-19, SLURM jobs 7275700/7275701/7275702 — metrics added here
  once complete.

Model checkpoints (`models/<fold>/best_model.pt`) are NOT checked into git
(`.gitignore` excludes `*.pt`) — they live on scratch at
`/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results19/`.
