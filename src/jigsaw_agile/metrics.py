from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


def column_averaged_auc(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray | None = None,
) -> float:
    """Compute competition-style mean AUC.

    If groups is provided, AUC is computed per group and averaged. The Kaggle
    competition describes this as column-averaged AUC; for local experiments the
    rule or subreddit column is usually used as the grouping axis.
    """
    if groups is None:
        return float(roc_auc_score(y_true, y_score))

    scores: list[float] = []
    df = pd.DataFrame({"y": y_true, "p": y_score, "g": groups})
    for _, part in df.groupby("g"):
        if part["y"].nunique() < 2:
            continue
        scores.append(float(roc_auc_score(part["y"], part["p"])))
    if not scores:
        raise ValueError("No group has both positive and negative labels.")
    return float(np.mean(scores))


def make_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "rule",
    label_col: str = "rule_violation",
) -> list[tuple[np.ndarray, np.ndarray]]:
    if group_col in df and df[group_col].nunique() >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, df[label_col], groups=df[group_col]))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(df, df[label_col]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--label-col", default="rule_violation")
    parser.add_argument("--pred-col", default="rule_violation")
    parser.add_argument("--group-col", default=None)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)
    pred = pd.read_csv(args.pred)
    df = truth.merge(pred, on="row_id", suffixes=("_true", "_pred"))
    label_col = args.label_col + "_true" if args.label_col in pred.columns else args.label_col
    pred_col = args.pred_col + "_pred" if args.pred_col in truth.columns else args.pred_col
    groups = df[args.group_col] if args.group_col else None
    score = column_averaged_auc(df[label_col], df[pred_col], groups=groups)
    print(f"AUC: {score:.6f}")
