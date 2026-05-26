"""
model.py
========
Transformer encoder for per-genomic-position modification detection.

  SinusoidalPositionalEncoding — fixed sinusoidal PE from "Attention is All You Need"
  ModTransformer               — full encoder: projection → PE → N×encoder → head
  count_parameters             — convenience helper
"""

import math
import torch
import torch.nn as nn

from config import D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT, MAX_SEQ_LEN


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic fixed sinusoidal encoding from "Attention is All You Need".

    A lookup table of shape (MAX_SEQ_LEN, D_MODEL) is pre-computed and
    registered as a non-trainable buffer.  At forward time the first L rows
    are added to the projected input embeddings.

    Why sinusoidal over learned:
      Generalises to sequence lengths unseen during training (important when
      the same model is run on contigs of varying lengths at inference).
      Also slightly fewer parameters, which matters less here but costs nothing.
    """

    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10_000.0) / d_model)
        )                                                              # (D/2,)

        pe[:, 0::2] = torch.sin(pos * div)    # even dims
        pe[:, 1::2] = torch.cos(pos * div)    # odd dims
        pe = pe.unsqueeze(0)                   # (1, max_len, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, D_MODEL)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ModTransformer(nn.Module):
    """
    Transformer encoder for per-genomic-position modification detection.

    Pipeline
    --------
    1. Input projection   (C → D_MODEL)          — lifts raw features into the
                                                    attention-compatible space
    2. Positional encoding                        — injects sequence order
    3. N × TransformerEncoderLayer               — multi-head self-attention +
                                                    position-wise FFN + LayerNorm
    4. Output projection  (D_MODEL → 1)          — scalar logit per position

    Padding masking
    ---------------
    `nn.TransformerEncoder` accepts a `src_key_padding_mask` of shape (B, L)
    where True marks positions to IGNORE.  We invert the real-position mask
    (real=True) so padding positions are excluded from attention and never
    contribute to the loss.

    Returns raw logits of shape (B, 1, L) — same layout as a U-Net so the
    training loop and evaluation code are interchangeable.

    Parameters
    ----------
    in_channels     : number of input feature channels (C)
    d_model         : internal embedding dimension
    nhead           : number of attention heads (d_model must be divisible by nhead)
    num_layers      : depth of the encoder stack
    dim_feedforward : width of the position-wise FFN inside each layer
    dropout         : applied after positional encoding and inside each layer
    """

    def __init__(
        self,
        in_channels:     int = 42,
        d_model:         int = D_MODEL,
        nhead:           int = NHEAD,
        num_layers:      int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout:         float = DROPOUT,
    ):
        super().__init__()

        # ── 1. Input projection ───────────────────────────────────────────────
        # A small MLP rather than a bare linear so the model can learn
        # non-linear feature interactions before applying attention.
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── 2. Positional encoding ────────────────────────────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # ── 3. Transformer encoder stack ──────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",   # GELU outperforms ReLU on sequence tasks
            batch_first     = True,     # (B, L, D) convention throughout
            norm_first      = True,     # Pre-LN: more stable gradients in deep stacks
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers           = num_layers,
            enable_nested_tensor = False,   # keep off for masking correctness
        )

        # ── 4. Classification head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x:    torch.Tensor,                    # (B, C, L)
        mask: torch.Tensor | None = None,      # (B, L)  True = real position
    ) -> torch.Tensor:
        """Returns raw logits of shape (B, 1, L)."""
        B, C, L = x.shape
        x = x.permute(0, 2, 1)                # (B, L, C)

        x = self.input_proj(x)                # (B, L, D)
        x = self.pos_enc(x)                   # (B, L, D)

        # PyTorch's key_padding_mask convention: True = IGNORE (invert our mask)
        key_padding_mask = (~mask) if mask is not None else None

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)   # (B, L, D)

        logits = self.head(x)                 # (B, L, 1)
        return logits.permute(0, 2, 1)        # (B, 1, L) — matches U-Net layout


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)