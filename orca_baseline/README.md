# ORCA baseline (DNA modifications)

A faithful reimplementation of the ORCA model (bioinfo-biols/ORCA) trained on our
DNA modification data, as a baseline for the DeepMod paper. ORCA ships no training
code, so this reproduces the architecture and training from the paper.

## Files
- `model.py` — the domain-adversarial model, structurally identical to ORCA's
  released `Models_20.py` (Extractor LSTMs → presence + stoichiometry heads →
  gradient-reversal domain head).
- `train.py` — the DANN training loop + evaluation (incl. leave-one-modification-out
  zero-shot, matching the paper's protocol).

## Matched to the paper/code
- AdamW, lr 5e-4
- Losses: NLL (presence) + NLL (domain) + MSE (stoichiometry)
- Positive label = position with >10% modified reads
- 4:1 train/test split; leave-one-mod-out zero-shot evaluation
- Architecture identical to the released ORCA code

## Set to standard DANN defaults (NOT in the paper — flagged `[ASSUMED]` in code)
- GRL lambda schedule `2/(1+exp(-10p))-1`
- Loss weighting between the three heads (equal, 1:1:1)
- Batch size, epochs

These are the honest limits of "matching ORCA exactly": the paper omits them, so
they're set to conventional DANN values and can be tuned.

## The one thing to wire: `load_features()` in `train.py`
It must return, for Bhargav's featurized DNA data:
- `X`        float32 (N, window, channels) — per-position features
- `y_mod`    int64   (N,) — 0 unmodified / 1 modified (>10% reads)
- `y_stoich` float32 (N,) — modification rate in [0,1]
- `y_domain` int64   (N,) — modification-type index (control, 5mC, 5hmC, 6mA)

Once Bhargav confirms the path + format, wiring this is the only remaining step
before training.

## Run
```bash
python train.py --features /path/to/bhargav/dna/features --out-dir orca_dna_out
# zero-shot (hold out one modification):
python train.py --features ... --leave-out 6mA
```
