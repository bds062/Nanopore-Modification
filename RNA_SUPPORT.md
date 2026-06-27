# Running DeepMod on RNA (RNA004)

Notes on the RNA-specific changes on this branch. The DNA path is unchanged:
every RNA behavior is either gated behind `--rna` or auto-detected in a way that
is a no-op for DNA inputs.

## What changed in `deepmod/featurization.py`

1. **U alphabet.** `BASE_ONEHOT` maps `U` to the `is_T` channel, and the
   complement table includes `U`. RNA reads basecalled as U and DNA-style T
   references both land in the same channel.

2. **Signal orientation auto-correction** (`normalize_orientation`). Direct RNA
   is sequenced 3'->5', so reference-anchored segmentation (our Remora-refined
   peaks) comes back with descending sample indices relative to the 5'->3'
   reference. When peaks are descending, they are reversed together with the
   reference pairing so each segment lines up with the base it covers. This
   triggers only on descending peaks, so ascending DNA / move-table peaks are
   untouched.

3. **`--rna` flag.** Folds any `U`-keyed entries in the k-mer level table onto
   `T` (reference k-mers are built from the FASTA, which uses T) and records
   `chemistry=RNA` in the HDF5 metadata.

## What did NOT need changing

- **`model.py` reverse-complement augmentation** is already gated behind
  `--rc-augment` (default off), and its docstring says to leave it off for
  strand-specific pore models. **Do not pass `--rc-augment` when training on
  RNA** — direct RNA is single-stranded, so complementary-strand augmentation
  would fabricate non-existent training examples.
- `visualization.py` / `lodo.py` use base labels only cosmetically and derive
  dataset names from filenames, so they work on RNA as-is.

## How to validate orientation is correct

`featurization.py` auto-detects the k-mer center by maximizing the Pearson
correlation between expected k-mer level and observed signal. After a run it
prints:

```
center_idx=N  (mean r=0.XX on M reads)
```

A high `r` (~0.7+) means the signal, segmentation, and reference are in register
and the RNA handling is correct. A low/negative `r` means orientation or the
level table is still off. Treat this number as the RNA sanity check before
trusting any features.

## Run order for RNA

1. `scripts/slurm/refine_rna_remora.sbatch` — Remora refinement -> refined peaks
2. `python -m deepmod.featurization --rna --level-table rna004_9mer_levels.tsv ...`
   (check the printed Pearson r)
3. `scripts/slurm/train_deepmod.sbatch` — train (no `--rc-augment`)
