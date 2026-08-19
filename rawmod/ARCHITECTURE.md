# deepmod — Model Architecture Specification

Nanopore base-modification detection via a DeepVariant-style pileup image fed to
a scaled InceptionV3 CNN, trained with BCE **+ Supervised Contrastive (SupCon)**
loss and a cross-dataset balanced sampler.

This document describes the full pipeline front to back: input representation,
data loading, model, losses, training loop, and evaluation. Line references
point at [`model.py`](model.py), [`lodo.py`](lodo.py), and
[`featurization.py`](featurization.py).

---

## 0. Problem framing

Each **candidate reference position** is classified as **modified (1)** or
**unmodified (0)**. Evidence for a position is the set of nanopore reads that
cover it. We render that evidence as a 2D multi-channel *pileup image* (reads ×
signal, like DeepVariant renders variant calls) and classify the image with a
CNN. Multiple images per position (when coverage is high) are averaged back into
one probability per position at scoring time.

The central difficulty is **genomic-context leakage**: with training negatives
drawn from different genomic regions than the positives, a CNN can score
"modified" by memorizing *which region a read came from* instead of learning the
*signal deviation* that a modification actually produces. The SupCon loss and
the cross-dataset sampler (Sections 4 & 6.3) exist specifically to defeat this.

---

## 1. Input representation — the pileup image

Produced by [`featurization.py`](featurization.py) and stored in HDF5.

**Tensor per image on disk:** `(H, W, C)` = `(max_reads+1, window_positions·L, n_channels)`
- `H = max_reads + 1` — **row 0 is the reference track**, rows `1..max_reads` are individual reads (zero-padded if coverage < max_reads).
- `W = window_positions · L` where `window_positions = 2·half_window + 1` and `L` = signal samples per base. With the current run: `half_window=10 → 21 positions`, `L=10` → **W = 210**.
- `C = 9` on disk; **2 delta channels are appended at load time → 11** (Section 3).
- Current shape: **(31, 210, 9) on disk → (11, 31, 210) fed to the model** after transpose + delta channels.

### 1.1 The 9 on-disk channels

| Ch | Name | Read rows (1..H) | Reference row (0) |
|----|------|------------------|-------------------|
| 0 | `raw_signal` | resampled to L samples, z-scored/MAD per read | **expected k-mer level** from pore model, broadcast flat across L |
| 1 | `dwell_log1p` | `log1p(dwell)`, broadcast across L | 0 |
| 2 | `is_A` | one-hot base identity | ref base one-hot |
| 3 | `is_C` | one-hot base identity | ref base one-hot |
| 4 | `is_G` | one-hot base identity | ref base one-hot |
| 5 | `is_T` | one-hot base identity | ref base one-hot |
| 6 | `strand` | +1 fwd / −1 rev, broadcast | 0 |
| 7 | `mapq_norm` | MAPQ/60 clipped to [0,1], broadcast | 0 |
| 8 | `matches_ref` | 1 if read base == ref base else 0 | 0 |

The reference row (row 0, channel 0) carries the **expected signal level** for
the k-mer at each window position, drawn from the pore model level table. This
is the anchor against which read deviation is measured — the basis of the delta
channels.

### 1.2 HDF5 datasets

```
/tensors    float32  (N, max_reads+1, W·L, C)   the images
/labels     int8     (N,)                         1=modified, 0=unmodified
/ref_names  bytes    (N,)                         reference contig
/ref_pos    int64    (N,)                         reference position
/n_reads    int16    (N,)                         reads in this image
/image_idx  int16    (N,)                         image index within its position
```
Key attrs: `n_channels`, `W`, `L`, `center_idx` (which window position is the
candidate base; defaults to `W//2`).

`N ≥ N_positions` — high-coverage positions yield multiple non-overlapping images.

---

## 2. Dataset & leakage-safe splitting

`PileupDataset` ([model.py:246](model.py)) wraps one or more HDF5 files, presenting
them as one concatenated index space. Per-process HDF5 handle caching
(`_get_h5`, `_worker_init_fn`) keeps `num_workers>0` safe.

### 2.1 Position-grouped split

`split_position_groups` ([model.py](model.py)) assigns every `(ref_name, ref_pos)`
group entirely to **one** of train/val/test (default 70/15/15), stratified by
label. All images of a position — including the same coordinate appearing in
multiple files — stay together, preventing site-level leakage across splits.
A stricter `split_contig_groups` mode splits by whole contig.

### 2.2 Scoring keys vs split keys

- **Split** uses bare `(ref_name, ref_pos)` so a base is never shared across splits.
- **Scoring** uses `(file_idx, ref_name, ref_pos)` so the same coordinate can be
  scored separately as an unmodified control and a modified treatment
  (`source_position_keys`).

---

## 3. Delta channels (computed at load time)

Appended in `PileupDataset.__getitem__` ([model.py:269](model.py)) **after**
augmentation, so they stay consistent with the (possibly noised) signal the
model actually sees. No re-featurization required.

Let `ci = center_idx`, `L = samples_per_base`, center column slice `cs:ce =
[ci·L, ci·L+L)`.

- **Ch 9 — `read_supports_modification`** (the deepmod analogue of DeepVariant's
  `read_supports_variant`): per-read mean signal at the center base minus the
  expected reference level, **broadcast across the entire row**:
  ```
  ref_ctr  = x[0, 0, cs:ce].mean()            # expected level at center (scalar)
  read_ctr = x[0, 1:, cs:ce].mean(axis=1)     # each read's center signal (H-1,)
  ch9[0, 1:, :] = (read_ctr - ref_ctr)[:, None]
  ```
  The CNN reads the resulting **column pattern** across reads: at a true
  modification most rows agree on a large deviation; noise fires in only one or
  two rows.

- **Ch 10 — `window_delta`**: full-window per-sample deviation of each read from
  the reference track:
  ```
  ch10[0, 1:] = x[0, 1:] - x[0, 0]
  ```

Row 0 (reference) is left zero in both delta channels. Final tensor: **(11, H, W)**.

Ablate with `--no-delta-channels` (drops back to 9 input channels).

### 3.1 Training augmentation (train split only)

In `__getitem__`, before delta channels:
- **Read dropout:** zero out up to 30%·rand() of read rows (row 0 never masked).
- **Signal noise:** additive Gaussian σ=`signal_noise_std` (default 0.05) on channel 0 only.
- **Reverse-complement** (`--rc-augment`, off by default): reverses width, swaps A↔T/C↔G one-hot, flips strand sign.
- **Mixup** (`mixup_batch`, batch-level): convex-combines pairs; **disabled for SupCon runs** (soft labels break pair mining). Incompatible with focal loss.

---

## 4. `DatasetBalancedSampler` — cross-dataset batches

[model.py:734](model.py). **Required for SupCon**: if a batch were all one
dataset, the contrastive loss would only see within-dataset pairs and couldn't
enforce cross-dataset invariance.

- Buckets training samples by `(file_idx, label)`. **Stores positional indices
  into the dataset's `indices` array** (not global H5 indices — the DataLoader
  passes these straight to `__getitem__`, which indexes `self.indices[item]`).
- Each epoch cycles files **round-robin**, drawing one unmod then one mod index
  per file → an interleaved sequence guaranteeing cross-file pairs in every
  batch regardless of batch size.
- Missing `(file, label)` buckets (e.g. all-unmod barcodes) are skipped without
  error. Buckets are reshuffled each epoch via `set_epoch(epoch)` (seed+epoch).

Auto-enabled whenever `--supcon-weight > 0`; supersedes `--balanced-sampler`.
Without SupCon, training uses either a `WeightedRandomSampler`
(`--balanced-sampler`) or plain shuffling.

---

## 5. Model — `PileupInceptionV3`

[model.py:551](model.py). Scaled-down InceptionV3 (~25% channel width),
**~1,584,697 parameters**. Backbone topology preserved so it still captures
multi-scale spatial structure in the pileup.

**`ConvBnRelu`** ([model.py](model.py)) is the primitive: Conv2d(bias=False) →
BatchNorm2d(eps=1e-3, **momentum=0.01**) → ReLU. *(momentum=0.01 not the torch
default 0.1, and specifically not the old 0.001 that caused val-loss explosions
at batch 512 — running stats must converge fast enough to match the batch stats
training uses.)*

### 5.1 End-to-end tensor flow

Input **(B, 11, 31, 210)**:

| Stage | Module | Output shape |
|-------|--------|--------------|
| Stem | 3×ConvBnRelu + 2×MaxPool (two stride-2 convs + two stride-2 pools) | (B, 64, 4, 27) |
| InceptionA1 | 4-branch, pool_proj=8 → 16+16+24+8 | (B, 64, 4, 27) |
| InceptionA2 | pool_proj=16 → 16+16+24+16 | (B, 72, 4, 27) |
| InceptionA3 | pool_proj=16 | (B, 72, 4, 27) |
| *(CrossReadAttention)* | optional, at 72ch | (B, 72, 4, 27) |
| InceptionB | grid reduction, stride 2 → 96+24+72 | (B, 192, 2, 14) |
| InceptionC1–C4 | factorized 7×7, 4× → 48·4 | (B, 192, 2, 14) |
| InceptionD | grid reduction, stride 2 → 80+48+192 | (B, 320, 1, 7) |
| InceptionE1 | expanded branches → 80+192+192+48 | (B, 512, 1, 7) |
| InceptionE2 | expanded branches | (B, 512, 1, 7) |
| **Head** | AdaptiveAvgPool2d(1) → Flatten → Dropout(0.4) → Linear(512,1) | (B, 1) logit |

Weight init: Kaiming-normal (conv), ones/zeros (BN), truncated-normal std 0.01 (linear).

### 5.2 Block designs

- **InceptionA** — 4 branches: 1×1; 1×1→5×5; 1×1→3×3→3×3; avgpool→1×1(pool_proj). Concatenated.
- **InceptionB / InceptionD** — grid reductions (stride-2 conv branch + stride-2 factorized branch + maxpool), roughly halving H,W while growing channels.
- **InceptionC** — factorized 7×7 as (1×7)+(7×1) stacks; 4 branches → 192.
- **InceptionE** — expanded parallel (1×3)/(3×1) branches → 512.

### 5.3 CrossReadAttention (optional, `--cross-read-attention`)

[model.py:567](model.py), inserted between InceptionA3 and InceptionB (~26K params).
Lets reads attend to each other before the first grid reduction:
```
tokens = x.mean(dim=3).permute(0,2,1)   # (B, H, C): pool each read row over W
tokens = LayerNorm(tokens)
attn   = MultiheadAttention(4 heads)(tokens, tokens, tokens)
x = x + Linear(attn).permute(0,2,1).unsqueeze(3)   # residual, broadcast over W
```
Off in the current SupCon runs (tested separately).

### 5.4 ProjectionHead (SupCon only, `--supcon-weight > 0`)

[model.py:677](model.py). A throwaway MLP used **only during training** to map
the 512-dim backbone embedding to the sphere where SupCon is computed:
```
Linear(512, 256) → ReLU → Linear(256, 128) → L2-normalize
```
`supcon_proj_dim=0` ⇒ `self.proj_head = None` and the model behaves exactly as
before (inference and all existing checkpoints unaffected).

### 5.5 Dual-output forward

[model.py:622](model.py):
```python
def forward(self, x, return_embedding=False):
    ... backbone ...
    if return_embedding and self.proj_head is not None:
        embed = self.head[2](self.head[1](self.head[0](x)))  # (B,512) post-dropout
        logit = self.head[3](embed)                           # (B,1)
        return logit, self.proj_head(embed)                   # (B,1), (B,128)
    return self.head(x)                                       # (B,1)
```
The embedding is tapped at `head[2]` (after AdaptiveAvgPool→Flatten→Dropout,
before the classifier `head[3]`). Because `self.head` keys are unchanged, every
prior checkpoint still loads cleanly. Inference calls the default single-output
path — the projection branch never runs at eval time.

---

## 6. Losses

### 6.1 Classification loss

- **Default:** `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)` — pos_weight rebalances the minority modified class.
- **Focal** (`--focal`): `FocalBCELoss` ([model.py:129](model.py)) for very small positive counts (<1K); normalizes α so loss scale is stable. Not combined with mixup.

### 6.2 SupConLoss

[model.py:167](model.py). Supervised Contrastive Loss (Khosla et al. 2020) on the
L2-normalized 128-dim projections. **Positive pairs = same binary label,
regardless of source file.** For a batch of embeddings `z` (N×128) and labels:

```
sim      = (z @ zᵀ) / temperature                 # (N,N) cosine sims, τ=0.07
pos_mask = (label == labelᵀ), diagonal removed    # same-label pairs
sim      = sim − rowmax(sim).detach()             # numerical stability
log_prob = sim − log( Σ_{k≠i} exp(sim_ik) )       # log-softmax over non-self
loss_i   = − mean_{p∈pos(i)} log_prob_ip
loss     = mean over anchors that have ≥1 positive
```
Returns 0 when a batch has no valid positive pair (all one label). This is what
forces a *modified UMCES* site and a *modified ONT* site to share embedding
space — the encoder cannot satisfy it by memorizing genomic context.

### 6.3 Combined objective

```
total_loss = BCE(logit, y) + λ · SupCon(z, y_int)     # λ = --supcon-weight (0.1)
```
`y_int = y.long()` is captured **before** mixup so SupCon always mines pairs
from the true integer labels. λ=0 removes the projection head and the sampler
override entirely, recovering the plain BCE model.

---

## 7. Training loop

[model.py](model.py) `main()` and mirrored in [lodo.py](lodo.py) `run_lodo()`.

1. **Optimizer:** AdamW, `lr=1.2e-3`, `weight_decay=1e-3`.
2. **LR warmup** (`--lr-warmup-steps 500`): `LinearLR` ramps ~0→lr over 500 optimizer steps. `ReduceLROnPlateau(mode='max', factor=0.5, patience=7, min_lr=1e-6)` on **val AUPRC** takes over only after warmup completes.
3. **Per epoch:** `sampler.set_epoch(epoch)` reshuffles balanced buckets. Per batch:
   ```
   y_int = y.long()                        # pre-mixup labels for SupCon
   (mixup if enabled & not focal & not supcon)
   if supcon:  logit, z = model(x, return_embedding=True)
               loss = BCE(logit, y) + λ·SupCon(z, y_int)
   else:       loss = criterion(model(x), y)
   loss.backward()
   clip_grad_norm_(params, --grad-clip=1.0)   # if > 0
   optimizer.step();  warmup_sched.step() while global_step ≤ warmup steps
   ```
4. **Validation:** image probabilities aggregated to positions (`aggregate_by_position`), val AUPRC drives checkpointing and the scheduler. When val has no positives, a negative-class proxy `mean(1−p)` is used.
5. **Checkpointing:** best-AUPRC model saved to `best_model.pt` with `in_channels`, `cross_read_attention`, and `supcon_proj_dim` so eval can reconstruct the exact architecture. Full `training_state.pt` (optimizer/scheduler/warmup/global_step) enables crash-resume — re-running the same command skips finished stages.
6. **Early stopping:** patience 15 epochs on val AUPRC.

---

## 8. Evaluation

### 8.1 Internal test split
After training, the held-out 15% test split is scored: images → positions via
`aggregate_by_position`, F1-optimal threshold from the PR curve, then confusion
matrix, PR curve, training curves, and `test_predictions.npz`. A **ZeroR**
majority-class baseline is always reported.

### 8.2 External eval — `eval_umces5.py`
[eval_umces5.py](/fs/cbcb-scratch/bds062/results/deepmod_umces/eval_umces5.py)
scores fully held-out datasets (bc01 PCR control, bc06/07 WGS test files, ONT
6mA/5mC) at a fixed threshold=0.5, reading `supcon_proj_dim` from the checkpoint
to rebuild the model. This is the true generalization test.

### 8.3 LODO (Leave-One-Dataset-Out)
[lodo.py](lodo.py) `run_lodo`: retrain on all files but one, test on the held-out
file's every position. Runs as parallel SLURM folds; `collect_lodo.py` gathers
them into a summary. All Section 5–7 machinery (delta channels, warmup, grad
clip, SupCon, balanced sampler) is threaded through identically.

### 8.4 Channel importance
Permutation importance (`permutation_channel_importance`): shuffle each channel
across the batch, measure AUPRC drop. Confirms whether ch9/ch10 (delta) carry
the signal. Skipped in the current runs (`--skip-channel-importance`).

---

## 9. Current run (`train5.sh`) configuration

```
--epochs 50 --batch 512 --lr 1.2e-3 --lr-warmup-steps 500 --grad-clip 1.0
--supcon-weight 0.1 --supcon-temp 0.07 --supcon-proj-dim 128
--mixup-alpha 0            # disabled: soft labels break SupCon pair mining
--num-workers 4 --seed 42
```
- **deepmod_umces** (`results5/`): trains on bc02–07, delta channels on, ~1.58M params, BCE pos_weight≈2.13, SupCon λ=0.1. `barcode_unmod`/PCR barcodes provide the unmodified cross-context anchors.
- **deepmod_ont+umces** (`results5/`): 7 files (bc_unmod, bc06, bc07, 5mC, 5hmC, 6mA, control) — the richest cross-dataset setup, exactly what SupCon needs to prove context-invariant learning.

Input tensor per sample: **(11, 31, 210)** → logit `(1,)` + (training only) SupCon embedding `(128,)`.

---

## 10. Design rationale summary

| Choice | Why |
|--------|-----|
| Delta channels 9/10 | Give the CNN explicit per-read signal deviation, the DeepVariant `read_supports_variant` analogue — the actual physical basis of a modification call. |
| SupCon + projection head | Force modified-vs-unmodified separation in embedding space **across datasets**, defeating genomic-context leakage that BCE alone permits. |
| DatasetBalancedSampler | Guarantee cross-dataset positive/negative pairs exist in every batch, without which SupCon is inert. |
| BN momentum 0.01 | Running stats must track batch stats at batch 512; the old 0.001 caused eval-mode val-loss divergence. |
| LR warmup + grad clip | Stabilize the high LR (1.2e-3, linear-scaled for batch 512) during the first ~2 epochs. |
| Position-grouped split | Prevent site-level train/test leakage independent of the context problem. |
| ~1.4M params, unchanged | Small training set (~70–140K images) would overfit a larger backbone; revisit only with multi-genome data. |
```
