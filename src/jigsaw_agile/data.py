from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE = "Yes"
NEGATIVE = "No"
JUDGE_WORDS = "Violation:"
SYSTEM_PROMPT = """You are given a comment from reddit and a rule.
Your task is to classify whether the comment violates the rule.
Only respond Yes/No."""

EXAMPLE_COLUMNS = [
    "positive_example_1",
    "positive_example_2",
    "negative_example_1",
    "negative_example_2",
]


def read_train(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "train.csv")


def read_test(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "test.csv")


def attach_random_examples(df: pd.DataFrame, seed: int = 22) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.copy()
    out["positive_example"] = out.apply(
        lambda row: rng.choice([row["positive_example_1"], row["positive_example_2"]]),
        axis=1,
    )
    out["negative_example"] = out.apply(
        lambda row: rng.choice([row["negative_example_1"], row["negative_example_2"]]),
        axis=1,
    )
    return out.drop(columns=EXAMPLE_COLUMNS, errors="ignore")


def build_prompt(row: pd.Series) -> str:
    return f"""{SYSTEM_PROMPT}
Subreddit: r/{row["subreddit"]}
Rule: {row["rule"]}
Examples:
1) {row["positive_example"]}
{JUDGE_WORDS} Yes
2) {row["negative_example"]}
{JUDGE_WORDS} No
Comment: {row["body"]}
{JUDGE_WORDS}"""


def build_qwen_training_frame(
    data_dir: str | Path,
    use_train: bool = True,
    test_rule_frac: float = 0.05,
    seed: int = 22,
) -> pd.DataFrame:
    """Build SFT rows from train.csv plus labeled examples in test.csv.

    The competition test rows contain per-rule positive and negative examples.
    We transform these examples into extra supervised rows, then prompt the LLM
    with one positive and one negative demonstration.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    if use_train:
        train_df = read_train(data_dir)[
            ["body", "rule", "subreddit", "rule_violation", *EXAMPLE_COLUMNS]
        ].copy()
        train_df["positive_example"] = np.where(
            rng.random(len(train_df)) < 0.5,
            train_df["positive_example_1"],
            train_df["positive_example_2"],
        )
        train_df["negative_example"] = np.where(
            rng.random(len(train_df)) < 0.5,
            train_df["negative_example_1"],
            train_df["negative_example_2"],
        )
        frames.append(train_df.drop(columns=EXAMPLE_COLUMNS))

    test_df = read_test(data_dir)
    if test_rule_frac < 1.0:
        test_df = (
            test_df.groupby("rule", group_keys=False)
            .apply(lambda x: x.sample(frac=test_rule_frac, random_state=seed))
            .reset_index(drop=True)
        )

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_df = test_df[["rule", "subreddit", *EXAMPLE_COLUMNS]].copy()
            body_col = f"{violation_type}_example_{i}"
            other_same_col = f"{violation_type}_example_{3 - i}"
            anti_type = "negative" if violation_type == "positive" else "positive"
            sub_df["body"] = sub_df[body_col]
            sub_df[f"{violation_type}_example"] = sub_df[other_same_col]
            sub_df[f"{anti_type}_example"] = np.where(
                rng.random(len(sub_df)) < 0.5,
                sub_df[f"{anti_type}_example_1"],
                sub_df[f"{anti_type}_example_2"],
            )
            sub_df["rule_violation"] = 1 if violation_type == "positive" else 0
            frames.append(sub_df.drop(columns=EXAMPLE_COLUMNS))

    return pd.concat(frames, axis=0).drop_duplicates(ignore_index=True)


def flatten_examples_for_classifier(data_dir: str | Path) -> pd.DataFrame:
    """Create labeled rows from train labels and all positive/negative examples."""
    train_df = read_train(data_dir)
    test_df = read_test(data_dir)
    frames = [train_df[["body", "rule", "subreddit", "rule_violation"]].copy()]

    for source_df in [train_df, test_df]:
        for violation_type in ["positive", "negative"]:
            label = 1 if violation_type == "positive" else 0
            for i in range(1, 3):
                col = f"{violation_type}_example_{i}"
                if col not in source_df:
                    continue
                sub_df = source_df[[col, "rule", "subreddit"]].copy()
                sub_df = sub_df.rename(columns={col: "body"})
                sub_df["rule_violation"] = label
                sub_df = sub_df.dropna(subset=["body"])
                sub_df = sub_df[sub_df["body"].str.strip().str.len() > 0]
                if not sub_df.empty:
                    frames.append(sub_df)

    out = pd.concat(frames, axis=0)
    out = out.drop_duplicates(subset=["body", "rule", "subreddit"], ignore_index=True)
    out = out.drop_duplicates(subset=["body", "rule"], keep="first")
    return out.sample(frac=1, random_state=42).reset_index(drop=True)
