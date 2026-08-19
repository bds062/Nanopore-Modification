#!/bin/bash
# Chemistry-diversity sweep (ORCA-style): for each held-out chemistry, how does
# zero-shot AUROC scale with the number of OTHER chemistries seen in training
# (2, 3, or 4 -- 4 is exactly the existing loco_<CHEM> LOCO experiment)?
#
# A trained model on chemistry-subset S is zero-shot-scored against EVERY
# chemistry not in S, so the full 2-4 sweep only needs C(5,2)+C(5,3)+C(5,4)=25
# unique trainings, not one retrain per (target, subset) pair. The 5 size-4
# subsets are exactly results17_temp0.20's loco_5mC/5hmC/6mA/4mC/5hmU
# checkpoints/metrics (same recipe) and are reused as-is (see
# make_diversity_sweep_figures.py) -- this script only launches the 20 new
# size-2/size-3 subset_ folds.
#
# Recipe = results17_temp0.20 exactly (confirmed via wandb config + SLURM
# logs): SUPCON_TEMP=0.20 SUPCON_WEIGHT=1.0 SUPCON_DIM=128 CURRICULUM=1
# CURRICULUM_EPOCHS=15 SAD_DIM=32 SAD_WEIGHT=1.0 SAD_ETA=1.0 BCE_WEIGHT=1.0
# (default) EXTRA_ORGANISMS=1 INCLUDE_HUMAN=1 RAWMOD_DATA_GEN=strand15 --
# so the size-4 anchor point is directly comparable to the new size-2/3 points.
#
# Usage (login node): bash run_chem_diversity_sweep.sh [--dry-run]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${HERE}/../../scripts/train" && pwd)"

CHEMS=(5hmU 4mC 6mA 5mC 5hmC)
FOLDS=""
# size-2 and size-3 combinations of CHEMS, sorted within each combo for a
# canonical fold name (subset_<chem>+<chem>[+...])
FOLDS=$(python3 - <<'EOF'
import itertools
CHEMS = ('5hmU', '4mC', '6mA', '5mC', '5hmC')
combos = list(itertools.combinations(CHEMS, 2)) + list(itertools.combinations(CHEMS, 3))
print(' '.join('subset_' + '+'.join(sorted(c)) for c in combos))
EOF
)
echo "20 new subset folds:"
echo "  ${FOLDS}" | tr ' ' '\n' | sed 's/^/    /'

FOLDS="${FOLDS}" \
OUTDIR=/fs/cbcb-scratch/bds062/results/rawmod_matched_loco/results18_chem_diversity \
RAWMOD_DATA_GEN=strand15 \
EXTRA_ORGANISMS=1 \
INCLUDE_HUMAN=1 \
SUPCON_DIM=128 \
SUPCON_WEIGHT=1.0 \
SUPCON_TEMP=0.20 \
CURRICULUM=1 \
CURRICULUM_EPOCHS=15 \
SAD_DIM=32 \
SAD_WEIGHT=1.0 \
SAD_ETA=1.0 \
BCE_WEIGHT=1.0 \
TIME_LIMIT=12:00:00 \
bash "${TRAIN_DIR}/run_matched_loco.sh" "$@"
