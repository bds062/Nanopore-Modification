#!/usr/bin/env python3
"""
Detect statistically outlier ref_pos positions from a TSV file
using selected features and a one-sample t-test approach.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from statsmodels.stats.multitest import multipletests
import argparse

# ── Configuration ────────────────────────────────────────────────────────────
FEATURES = ["mean_dev", "std_dev", "diff1", "t_stat", "mean_dwell", "dwell_var"]   # ← change features here
ALPHA    = 0.05                          # significance threshold
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_gt(path: str) -> set[tuple[str, int]]:
    """Load BED file and return a set of (ref_name, ref_pos) tuples."""
    bed = pd.read_csv(path, sep="\t", header=None)
    return set(zip(bed.iloc[:, 0], bed.iloc[:, 1].astype(int)))


def mahalanobis_outliers(df: pd.DataFrame, features: list[str], alpha: float) -> pd.DataFrame:
    X = df[features].values.astype(float)

    mean_vec = X.mean(axis=0)
    cov      = np.cov(X, rowvar=False)
    cov     += np.eye(cov.shape[0]) * 1e-6
    inv_cov  = np.linalg.inv(cov)

    distances = np.array([mahalanobis(row, mean_vec, inv_cov) for row in X])

    k        = len(features)
    d_sq     = distances ** 2
    p_values = 1 - stats.chi2.cdf(d_sq, df=k)

    _, p_adj, _, _ = multipletests(p_values, method="fdr_bh")

    result = df.copy()                   # ← preserves ALL original columns
    result["mahal_dist"] = distances
    result["p_value"]    = p_values
    result["p_adj"]      = p_adj
    result["outlier"]    = p_adj < alpha

    return result


def compute_metrics(outlier: pd.Series, gt: pd.Series) -> dict:
    tp = ((outlier == True)  & (gt == True)).sum()
    fp = ((outlier == True)  & (gt == False)).sum()
    fn = ((outlier == False) & (gt == True)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--features", nargs="+", default=FEATURES)
    parser.add_argument("--alpha",    type=float, default=ALPHA)
    parser.add_argument(
        "--gt",
        nargs="?",
        const="__EMPTY__",
        default=None,
        help="Path to ground truth BED file (or use --gt alone for all zeros)"
    )
    parser.add_argument("--out",      default=None, help="Path for output TSV")
    args = parser.parse_args()

    df = load_data(args.tsv)
    print(f"\nFeatures : {args.features}")
    print(f"Alpha    : {args.alpha} (BH-corrected)")

    results = mahalanobis_outliers(df, args.features, args.alpha)

    # ── Ground truth ─────────────────────────────────────────────────────────
    if args.gt is None:
        # --gt not provided at all
        results["gt"] = False

    elif args.gt == "__EMPTY__":
        # --gt provided but no file → fill with 0s
        results["gt"] = False

    else:
        # --gt with file path
        gt_set = load_gt(args.gt)
        results["gt"] = list(zip(results["ref_name"], results["ref_pos"].astype(int)))
        results["gt"] = results["gt"].apply(lambda x: x in gt_set)

    # ── Column order: all original cols | stats | outlier | gt ───────────────
    # Non-numeric columns are preserved from the original dataframe but were
    # never passed to the statistical analysis.
    stat_cols  = ["mahal_dist", "p_value", "p_adj"]
    added_cols = stat_cols + ["outlier", "gt"]
    orig_cols  = [c for c in results.columns if c not in added_cols]
    out_cols   = orig_cols + added_cols
    results    = results[out_cols]

    # ── stdout summary ────────────────────────────────────────────────────────
    outliers = results[results["outlier"]]
    print(f"\n{'─'*50}")
    print(f"Total positions : {len(results)}")
    print(f"Outliers flagged: {len(outliers)}")

    if args.gt:
        n_gt = results["gt"].sum()
        print(f"GT positives    : {n_gt}")
        m = compute_metrics(results["outlier"], results["gt"])
        print(f"\n  TP        : {m['TP']}")
        print(f"  FP        : {m['FP']}")
        print(f"  FN        : {m['FN']}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1        : {m['f1']:.4f}")
    print(f"{'─'*50}\n")

    if args.out:
        results.to_csv(args.out, sep="\t", index=False)
        print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()