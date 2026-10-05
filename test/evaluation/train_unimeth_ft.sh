#!/bin/bash
# UniMeth-FT: UniMeth fine-tuned on exactly the training positions of the released RawMod model, so that both models are
# scored only at positions neither has trained on. One model per modification RawMod covers and UniMeth does not:
#   5hmC  ONT oligonucleotides, syn_5hmC vs. syn_control     base: released 5 kHz 5mC model
#   4mC   H. pylori 26695, native vs. WGA                    base: released 5 kHz 5mC model
#   5hmU  SPO1, native vs. PCR                               base: released 5 kHz 6mA model
# Labels (bench.unimeth_ft_labels) are written into copies of the benchmark basecalls: label 255 / 0 only at RawMod
# training positions of the organism; every other base is masked. Fine-tuning: --frequency 4khz, 3,000 steps, batch 32,
# 800 validation reads per library.
#
# Requires: the benchmark basecalls of the six samples (rawmod_bench.py run --steps subset,basecall), UNIMETH,
# UNIMETH_CHECKPOINTS and UNIMETH_FT_PYTHONPATH (UniMeth 11215d4 + patches/unimeth-11215d4-finetune.patch), one GPU.
#   bash test/evaluation/train_unimeth_ft.sh [5hmC|4mC|5hmU ...]        (default: all three)
# Output: $UNIMETH_FT_CHECKPOINTS/<mod>/final.pt (default $BENCH_WORK/unimeth_ft).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../benchmark" && pwd)
cd "$HERE"
PY=$(python3 -c "from bench import settings as S; print(S.get('RAWMOD_PYTHON') or 'python')")
setting() { $PY -c "from bench import settings as S; import sys; print(S.need('$1', 'UniMeth-FT training'))"; }
sample() { $PY -c "from bench import common as C; print(C.samples()['$1']['$2'])"; }
UNIMETH=$(setting UNIMETH); CK=$(setting UNIMETH_CHECKPOINTS); FT=$(setting UNIMETH_FT_PYTHONPATH)
UPY=$(dirname "$UNIMETH")/python
OUT=$($PY -c "from bench import settings as S; import os; print(S.get('UNIMETH_FT_CHECKPOINTS') or os.path.join(S.get('BENCH_WORK'), 'unimeth_ft'))")
mkdir -p "$OUT/bam" "$OUT/pod5"
GT4=$HERE/../data/unimeth_ft/hp26695_4mC_labels.bed.gz

MODS=("$@"); [ ${#MODS[@]} -gt 0 ] || MODS=(5hmC 4mC 5hmU)
for MOD in "${MODS[@]}"; do
  case $MOD in
    5hmC) POS=syn_5hmC;  NEG=syn_control; BASE=C; CODE=h;     ORG='^ONT::';  BASEMODEL=$CK/unimeth_r10.4.1_5kHz_5mC.pt
          POSARGS=(--pos-files '^ONT::5hmC');                FLAGS="--cpg 1 --chg 1 --chh 1 --hmC 1" ;;
    4mC)  POS=hp26695;   NEG=hp26695_wga; BASE=C; CODE=21839; ORG='^HP::';   BASEMODEL=$CK/unimeth_r10.4.1_5kHz_5mC.pt
          POSARGS=(--pos-files '^HP::WT' --gt "$GT4");       FLAGS="--cpg 1 --chg 1 --chh 1 --m4C 1" ;;
    5hmU) POS=spo1_bc07; NEG=spo1_bc01;   BASE=T; CODE=g;     ORG='^SPO1::'; BASEMODEL=$CK/unimeth_r10.4.1_5kHz_6mA.pt
          POSARGS=(--all-modified);                          FLAGS="--hmU 1" ;;
    *) echo "unknown modification $MOD" >&2; exit 1 ;;
  esac
  # 1. labels at RawMod training positions (bench.holdout source table)
  [ -s "$OUT/bam/${MOD}_pos.bam" ] || BENCH_HOLDOUT=r81 $PY -m bench.unimeth_ft_labels --sample $POS --state modified \
      --base $BASE --code $CODE --org-files "$ORG" "${POSARGS[@]}" --out "$OUT/bam/${MOD}_pos.bam"
  [ -s "$OUT/bam/${MOD}_neg.bam" ] || BENCH_HOLDOUT=r81 $PY -m bench.unimeth_ft_labels --sample $NEG --state unmodified \
      --base $BASE --code $CODE --org-files "$ORG" --out "$OUT/bam/${MOD}_neg.bam"
  # 2. validation reads: the first 800 labelled reads of each library
  for S in pos neg; do
    smp=$([ $S = pos ] && echo $POS || echo $NEG)
    if [ ! -s "$OUT/pod5/${MOD}_${S}_val.pod5" ]; then
      $UPY -c "import pysam; f=pysam.AlignmentFile('$OUT/bam/${MOD}_${S}.bam'); ids=[]
for r in f:
    if r.has_tag('MM') and r.query_name not in ids: ids.append(r.query_name)
    if len(ids) >= 800: break
open('$OUT/pod5/${MOD}_${S}_val_ids.txt','w').write('\n'.join(ids)+'\n')"
      $(dirname "$UNIMETH")/pod5 filter --ids "$OUT/pod5/${MOD}_${S}_val_ids.txt" --missing-ok \
        --output "$OUT/pod5/${MOD}_${S}_val.pod5" "$(sample $smp sub_pod5)"
    fi
  done
  # 3. fine-tuning
  RUN=$OUT/$MOD; mkdir -p "$RUN"
  if [ -s "$RUN/final.pt" ]; then echo "$MOD: $RUN/final.pt exists"; continue; fi
  for S in pos neg; do     # UniMeth builds its read index before the data-loader workers start
    PYTHONNOUSERSITE=1 PYTHONPATH=$FT $UPY -c "from unimeth.ioutils.reader import BamReader; BamReader('$OUT/bam/${MOD}_${S}.bam')" >/dev/null
  done
  (cd "$RUN" && PYTHONNOUSERSITE=1 PYTHONPATH=$FT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True UNIMETH_OUT_DIR=$RUN \
     UNIMETH_RESUME=0 UNIMETH_LOG_STEPS=50 UNIMETH_EVAL_STEPS=500 UNIMETH_SAVE_STEPS=500 UNIMETH_DL_WORKERS=6 \
     $UPY -m unimeth.training --mode finetune --bam_dir "$OUT/bam/${MOD}_pos.bam,$OUT/bam/${MOD}_neg.bam" \
       --train_pod5_dir "$(sample $POS sub_pod5),$(sample $NEG sub_pod5)" \
       --val_pod5_dir "$OUT/pod5/${MOD}_pos_val.pod5,$OUT/pod5/${MOD}_neg_val.pod5" \
       --model_dir "$BASEMODEL" --pore_type R10.4.1 --frequency 4khz $FLAGS --dorado_version 1.4 --max_steps 3000 \
       --batch_size 32 --run_name ${MOD}_ft > "$RUN/train.log" 2>&1)
  echo "$MOD: $RUN/final.pt"
done
