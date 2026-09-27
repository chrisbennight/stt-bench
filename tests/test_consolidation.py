"""Ranking and common-reference aggregation contracts for the published table."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "consolidation", Path(__file__).resolve().parents[1] / "scripts/consolidate_results.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_medals_preserve_ties_and_metric_direction():
    assert module.medal(2, [1, 2, 2, 4]) == "🥈 "
    assert module.medal(4, [1, 2, 2, 4]) == ""
    assert module.medal(4, [1, 2, 4], higher=True) == "🥇 "
    assert module.medal(None, [1, 2]) == ""
    assert module.medal(1, [1]) == ""


def test_speaker_metrics_share_clip_ids_and_weight_by_reference_size():
    def row(identifier, errors, length):
        score = {"errors": errors, "length": length}
        return {"id": identifier, "scores": {
            "cpwer": score, "tcpwer": score,
            "der": {"total": length, "false alarm": errors,
                    "missed detection": 0, "confusion": 0},
        }}

    groups = {
        "a": [row("small", 1, 1), row("large", 0, 100), row("exclusive", 50, 50)],
        "b": [row("small", 0, 1), row("large", 20, 100)],
    }
    common, scores = module.shared_speaker_scores(groups)
    assert common == ["large", "small"]
    for metric in ("cpwer", "tcpwer", "der"):
        assert scores["a"][metric] == pytest.approx(1 / 101)
        assert scores["b"][metric] == pytest.approx(20 / 101)
