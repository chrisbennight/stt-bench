"""Ranking and common-reference aggregation contracts for the published table."""

import importlib.util
from copy import deepcopy
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


def test_untimed_speaker_output_keeps_cpwer_on_the_same_selected_clips():
    rows = [{"id": "a", "scores": {
        "cpwer": {"errors": 2, "length": 10}, "tcpwer": None, "der": None,
    }}]
    ids, scores = module.shared_speaker_scores({"streaming": rows}, ["a"])
    assert ids == ["a"]
    assert scores["streaming"] == {"cpwer": 0.2, "tcpwer": None, "der": None}


@pytest.mark.parametrize("field", ["id", "duration", "audio_sha256", "reference",
                                   "reference_activity"])
def test_consolidation_rejects_different_audio_or_reference_windows(field):
    original = [{"id": "a", "duration": 60, "audio_sha256": "audio-hash",
                 "reference": [{"text": "hello"}], "reference_activity": []}]
    changed = deepcopy(original)
    changed[0][field] = "different"
    with pytest.raises(ValueError, match="must match exactly"):
        module.validate_same_records(original, changed)
    module.validate_same_records(original, deepcopy(original))


def test_consolidation_rejects_a_different_clip_count():
    with pytest.raises(ValueError, match="must match exactly"):
        module.validate_same_records([{"id": "a"}], [])


def test_missing_cells_distinguish_capability_and_exclusion():
    assert module.missing_cell({"cpwer": 0.2}, "der") == "No speaker times"
    assert module.missing_cell({"cpwer": None}, "der") == "No labels"
    assert module.missing_cell({"excluded": True}, "wer") == "Excluded"


def test_costs_include_rejected_transcripts_but_never_invent_failed_request_costs():
    assert module.reported_cost({"status": "invalid_output", "metadata": {
        "response_usage": {"cost": 0.01}}}) == 0.01
    assert module.reported_cost({"status": "inference_failed"}) is None
    assert module.reported_cost({"status": "ok", "prediction": {
        "metadata": {"usage": {"cost": 0}}}}) == 0
