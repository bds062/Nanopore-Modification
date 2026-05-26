"""
features.py
===========
Featurisation helpers for the Transformer modification classifier.

  encode_kmer          — one-hot encode a Series of k-mer strings
  load_and_label       — load a TSV, assign binary labels, tag with dataset name
  build_feature_matrix — concatenate scalar features and one-hot k-mer encodings
"""

import pandas as pd

from config import SCALAR_FEATURES, KMER_COL

BASES = ["A", "C", "G", "T"]


def encode_kmer(kmer_series: pd.Series) -> pd.DataFrame:
    """
    One-hot encode a Series of k-mer strings.

    Each position in the k-mer produces 4 binary columns (A, C, G, T).
    Unknown bases (N) map to all zeros.

      "ACN" → [1,0,0,0,  0,1,0,0,  0,0,0,0]
               A C G T   A C G T   A C G T

    Returns a DataFrame with columns kmer_pos_0_A … kmer_pos_{k-1}_T.
    Total columns = k-mer length × 4  (9-mer → 36 columns).
    """
    kmer_len  = len(kmer_series.iloc[0])
    col_names = [f"kmer_pos_{i}_{b}" for i in range(kmer_len) for b in BASES]

    def _one_hot(kmer: str) -> list[int]:
        row = []
        for base in kmer:
            for b in BASES:
                row.append(1 if base == b else 0)
        return row

    encoded = kmer_series.apply(_one_hot).tolist()
    return pd.DataFrame(encoded, columns=col_names, index=kmer_series.index)


def load_and_label(path: str, is_modified_dataset: bool, dataset_name: str) -> pd.DataFrame:
    """
    Load a TSV, assign binary labels, and tag rows with dataset name.

    Label rules
    -----------
      unmodified dataset, gt=False → label 0  (unmodified)
      unmodified dataset, gt=True  → dropped  (label error)
      modified    dataset, gt=False → label 0  (unmodified reference site)
      modified    dataset, gt=True  → label 1  (confirmed modification)
    """
    print(f"  Loading {dataset_name} from {path} …")
    df = pd.read_csv(path, sep="\t")
    df["dataset"] = dataset_name

    if not is_modified_dataset:
        n_spurious = int(df["gt"].sum())
        if n_spurious > 0:
            print(f"  ⚠  Dropping {n_spurious} gt=True rows from unmodified dataset.")
            df = df[~df["gt"]].copy()
        df["label"] = 0
    else:
        df["label"] = df["gt"].astype(int)

    n_mod   = int(df["label"].sum())
    n_unmod = int((df["label"] == 0).sum())
    print(f"     {dataset_name}: {n_unmod} unmodified, {n_mod} modified positions")
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate scalar features and one-hot k-mer encodings."""
    scalar = df[SCALAR_FEATURES].reset_index(drop=True)
    if not KMER_COL:
        print("  i  KMER_COL empty — scalar features only.")
        return scalar
    kmer_df = encode_kmer(df[KMER_COL].reset_index(drop=True))
    return pd.concat([scalar, kmer_df], axis=1)