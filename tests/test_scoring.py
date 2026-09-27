import pytest

from speaker_benchmark.schema import Prediction, Segment
from speaker_benchmark.scoring import normalize, score_record


def record():
    reference = [Segment("a", "hello there", 0, 1), Segment("b", "good morning", 2, 3)]
    return {"reference": reference, "reference_activity": reference, "duration": 10}


def test_speaker_names_are_permutation_invariant():
    hyp = Prediction([Segment("x", "hello there", 0, 1), Segment("y", "good morning", 2, 3)])
    scores = score_record(record(), hyp)
    assert scores["cpwer"]["errors"] == 0
    assert scores["tcpwer"]["errors"] == 0
    assert scores["der"]["diarization error rate"] == 0


def test_collapsing_speakers_is_penalized_despite_correct_text():
    hyp = Prediction([Segment("x", "hello there", 0, 1), Segment("x", "good morning", 2, 3)])
    scores = score_record(record(), hyp)
    assert scores["wer"]["errors"] == 0
    assert scores["cpwer"]["errors"] > 0
    assert scores["der"]["confusion"] == pytest.approx(1)


def test_empty_hypothesis_is_complete_deletion_not_dropped():
    scores = score_record(record(), Prediction([]))
    assert scores["cpwer"]["deletions"] == 4
    assert scores["coverage"]["ratio"] == 0
    assert scores["der"]["missed detection"] == 2


def test_streaming_without_timestamps_does_not_get_fake_timing_scores():
    scores = score_record(record(), Prediction([Segment("x", "hello there")], timing="unavailable"))
    assert scores["cpwer"]["deletions"] == 2
    assert scores["tcpwer"] is None
    assert scores["der"] is None
    assert scores["coverage"] is None


def test_delayed_transcript_passes_cp_but_fails_tcp():
    hyp = Prediction([Segment("a", "hello there", 7, 8), Segment("b", "good morning", 8, 9)])
    scores = score_record(record(), hyp, tcp_collar=0.5)
    assert scores["cpwer"]["errors"] == 0
    assert scores["tcpwer"]["errors"] > 0


def test_diarization_activity_retains_overlapping_speakers():
    ref = [Segment("a", "one", 0, 2), Segment("b", "two", 1, 3)]
    r = {"reference": ref, "reference_activity": ref, "duration": 3}
    scores = score_record(r, Prediction(ref, activity=[Segment("x", "", 0, 3)]))
    assert scores["der"]["missed detection"] == pytest.approx(1)
    assert scores["coverage"]["ratio"] == 1  # Coverage alone cannot certify correct attribution.


def test_silence_false_alarm_is_retained():
    scores = score_record(
        {"reference": [], "reference_activity": [], "duration": 3},
        Prediction([Segment("x", "hallucination", 1, 2)]),
    )
    assert scores["cpwer"]["insertions"] == 1
    assert scores["der"]["false alarm"] == 1


def test_normalization_preserves_fillers_and_negation():
    assert normalize("Um, I DON’T want twenty-one.") == "um i don't want twenty one"
