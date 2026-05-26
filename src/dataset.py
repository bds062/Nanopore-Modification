"""
dataset.py
==========
PyTorch Dataset and collate function for the Transformer modification classifier.

  ContigTileDataset — splits contigs into fixed-length overlapping/non-overlapping
                      windows, zero-padding the final tile if needed
  collate_fn        — stacks tiles into batches, flattening per-tile metadata
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from config import WINDOW_SIZE, BATCH_SIZE


class ContigTileDataset(Dataset):
    """
    Splits each contig into fixed-length windows.

    Training  — overlapping tiles (stride < window) for more gradient updates.
    Test      — non-overlapping tiles (stride = window) so each position is
                evaluated exactly once (unbiased metrics).

    Each item:
        x    : FloatTensor  (C, WINDOW_SIZE)   — features × positions
        y    : FloatTensor  (1, WINDOW_SIZE)   — binary labels (pad → 0)
        mask : BoolTensor   (WINDOW_SIZE,)     — True for real positions
        meta : list[(ref_name, ref_pos)]       — for TSV reconstruction
    """

    def __init__(
        self,
        X:        np.ndarray,
        y:        np.ndarray,
        groups:   np.ndarray,
        ref_pos:  np.ndarray,
        ref_name: np.ndarray,
        window:   int = WINDOW_SIZE,
        stride:   int | None = None,
    ):
        self.window = window
        stride      = stride if stride is not None else window
        self.tiles  = []

        for contig in np.unique(groups):
            idx   = np.where(groups == contig)[0]
            order = np.argsort(ref_pos[idx])
            idx   = idx[order]

            feat  = X[idx]
            labs  = y[idx]
            pos   = ref_pos[idx]
            rname = ref_name[idx]
            L     = len(feat)

            starts = list(range(0, max(L - window, 0) + 1, stride))
            if not starts or starts[-1] + window < L:
                starts.append(max(L - window, 0))

            for s in starts:
                e       = s + window
                pad_len = max(e - L, 0)
                real    = window - pad_len

                f_tile = np.zeros((window, feat.shape[1]), dtype=np.float32)
                l_tile = np.zeros(window,                  dtype=np.float32)
                m_tile = np.zeros(window,                  dtype=bool)
                p_tile = np.full(window, -1,               dtype=pos.dtype)
                r_tile = np.full(window, contig,           dtype=object)

                f_tile[:real] = feat[s:s + real]
                l_tile[:real] = labs[s:s + real]
                m_tile[:real] = True
                p_tile[:real] = pos[s:s + real]
                r_tile[:real] = rname[s:s + real]

                self.tiles.append((
                    f_tile.T,                            # (C, W)
                    l_tile,                              # (W,)
                    m_tile,                              # (W,)
                    list(zip(r_tile, p_tile)),
                ))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        feat, lab, mask, meta = self.tiles[idx]
        return (
            torch.from_numpy(feat),                # (C, W)
            torch.from_numpy(lab).unsqueeze(0),    # (1, W)
            torch.from_numpy(mask),                # (W,)
            meta,
        )


def collate_fn(batch):
    x_list, y_list, mask_list, meta_list = zip(*batch)
    meta_flat = [item for tile in meta_list for item in tile]
    return (
        torch.stack(x_list),
        torch.stack(y_list),
        torch.stack(mask_list),
        meta_flat,
    )