from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jigsaw_agile.config import DEFAULT_PATHS, BlendConfig


def rank_normalize(values: pd.Series) -> pd.Series:
    return values.rank(method="average") / (len(values) + 1)


def blend_submissions(
    deberta_path: str | Path,
    gte_path: str | Path,
    llm_path: str | Path,
    output_path: str | Path,
    config: BlendConfig = BlendConfig(),
) -> pd.DataFrame:
    deberta = pd.read_csv(deberta_path).sort_values("row_id").reset_index(drop=True)
    gte = pd.read_csv(gte_path).sort_values("row_id").reset_index(drop=True)
    llm = pd.read_csv(llm_path).sort_values("row_id").reset_index(drop=True)

    if not deberta["row_id"].equals(gte["row_id"]) or not deberta["row_id"].equals(llm["row_id"]):
        raise ValueError("Submission row_id columns must match before blending.")

    blend = (
        config.deberta_weight * rank_normalize(deberta["rule_violation"])
        + config.gte_weight * rank_normalize(gte["rule_violation"])
        + config.llm_weight * rank_normalize(llm["rule_violation"])
    )
    out = deberta[["row_id"]].copy()
    out["rule_violation"] = blend
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    default_config = BlendConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--deberta", default=str(DEFAULT_PATHS.output_dir / "sub_deberta_v3.csv"))
    parser.add_argument("--gte", default=str(DEFAULT_PATHS.output_dir / "sub_gte.csv"))
    parser.add_argument("--llm", default=str(DEFAULT_PATHS.output_dir / "sub_llm.csv"))
    parser.add_argument("--output", default=str(DEFAULT_PATHS.output_dir / "submission.csv"))
    parser.add_argument("--w-deberta", type=float, default=default_config.deberta_weight)
    parser.add_argument("--w-gte", type=float, default=default_config.gte_weight)
    parser.add_argument("--w-llm", type=float, default=default_config.llm_weight)
    args = parser.parse_args()
    blend_submissions(
        args.deberta,
        args.gte,
        args.llm,
        args.output,
        BlendConfig(args.w_deberta, args.w_gte, args.w_llm),
    )
