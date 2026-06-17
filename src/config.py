"""
config.py
=========
All hard-coded paths, feature definitions, and hyperparameters for the
Transformer modification classifier.
"""

import torch

# ── Input paths ───────────────────────────────────────────────────────────────
DATA_DIR  = "data"                              # override with --data-dir
TSV_UNMOD = f"{DATA_DIR}/control.tsv"
TSV_5MC   = f"{DATA_DIR}/5mC.tsv"
TSV_5HMC  = f"{DATA_DIR}/5hmC.tsv"
TSV_6MA   = f"{DATA_DIR}/6mA.tsv"

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR               = "."
MODEL_OUT             = "transformer_mod_classifier.pt"
PRED_OUT              = "transformer_predictions.tsv"
PR_FIG_OUT            = "transformer_prc.png"
TRAIN_FIG_OUT         = "transformer_training_curves.png"
CONFUSION_DEFAULT_OUT = "transformer_confusion_matrix_default.png"
CONFUSION_OPTIMAL_OUT = "transformer_confusion_matrix_optimal.png"
FEAT_IMP_FIG_OUT      = "transformer_feature_importance.png"
LOO_FIG_OUT           = "transformer_loo_results.png"
LOO_METRICS_OUT       = "transformer_loo_metrics.tsv"
LOO_TRAIN_FIG_PREFIX  = "transformer_loo_training_curves"   # + _{name}.png

# ── Feature configuration ─────────────────────────────────────────────────────
SCALAR_FEATURES = ["mean_dev", "std_dev", "diff1", "t_stat", "mean_dwell", "dwell_var"]
KMER_COL        = "kmer"          # set to "" to disable k-mer features

# ── Training hyperparameters ──────────────────────────────────────────────────
WINDOW_SIZE   = 512      # positions per tile
WINDOW_STRIDE = 256      # stride for overlapping training tiles; None = non-overlapping
BATCH_SIZE    = 32
NUM_EPOCHS    = 300
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 30
TEST_SIZE     = 0.2      # fraction of contigs held out for testing
THRESHOLD     = 0.5
SEED          = 42

# ── Transformer architecture hyperparameters ──────────────────────────────────
D_MODEL         = 128    # embedding / attention dimension (must be divisible by NHEAD)
NHEAD           = 8      # number of attention heads
NUM_LAYERS      = 6      # number of TransformerEncoderLayer stacks
DIM_FEEDFORWARD = 512    # inner dimension of the position-wise FFN
DROPOUT         = 0.1
MAX_SEQ_LEN     = 4096   # maximum window length for positional encoding table

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")