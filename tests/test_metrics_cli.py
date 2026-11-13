import sys

import pandas as pd
import pytest

from jigsaw_agile.metrics import main


def run_score(tmp_path, monkeypatch, truth, prediction, extra=()):
    truth_path, pred_path = tmp_path / "truth.csv", tmp_path / "prediction.csv"
    truth.to_csv(truth_path, index=False)
    prediction.to_csv(pred_path, index=False)
    monkeypatch.setattr(sys, "argv", ["jigsaw-score", "--truth", str(truth_path), "--pred", str(pred_path), *extra])
    main()


@pytest.mark.parametrize(
    "truth_ids,pred_ids",
    [([1, 2, 3], [1, 2]), ([1, 2, 3], [1, 2, 3, 4]),
     ([1, 1, 2, 3], [1, 2, 3]), ([1, 2, 3], [1, 1, 2, 3]),
     ([1, 2, None], [1, 2, 3]), ([1, 2, 3], [1, 2, None])],
)
def test_rejects_incomplete_or_ambiguous_alignment(tmp_path, monkeypatch, capsys, truth_ids, pred_ids):
    truth = pd.DataFrame({"row_id": truth_ids, "rule_violation": [int(i == 2) for i in truth_ids]})
    prediction = pd.DataFrame({"row_id": pred_ids, "rule_violation": [0.9 if i == 2 else 0.1 for i in pred_ids]})
    with pytest.raises(ValueError, match="row_id"):
        run_score(tmp_path, monkeypatch, truth, prediction)
    assert "AUC:" not in capsys.readouterr().out


def test_shuffled_complete_predictions_are_aligned_by_id(tmp_path, monkeypatch, capsys):
    truth = pd.DataFrame({"row_id": [1, 2, 3], "rule_violation": [0, 1, 0]})
    prediction = pd.DataFrame({"row_id": [3, 1, 2], "rule_violation": [0.1, 0.2, 0.9]})
    run_score(tmp_path, monkeypatch, truth, prediction)
    assert capsys.readouterr().out.strip() == "AUC: 1.000000"


def test_custom_columns_and_grouped_scores_remain_supported(tmp_path, monkeypatch, capsys):
    truth = pd.DataFrame({"row_id": [1, 2, 3, 4], "label": [0, 1, 0, 1], "rule": ["a", "a", "b", "b"]})
    prediction = pd.DataFrame({"row_id": [4, 3, 2, 1], "score": [0.9, 0.1, 0.8, 0.2]})
    run_score(tmp_path, monkeypatch, truth, prediction, ("--label-col", "label", "--pred-col", "score", "--group-col", "rule"))
    assert capsys.readouterr().out.strip() == "AUC: 1.000000"
