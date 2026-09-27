import io
import json

import numpy as np
import pytest
import soundfile as sf

from speaker_benchmark.adapters.openrouter import (
    PROVIDER_OPTIONS,
    OpenRouter,
    error_diagnostic,
    parse_response,
    validate_options,
)
from speaker_benchmark.schema import Segment
from speaker_benchmark.scoring import score_record


def test_fish_inline_labels_restore_cpwer_without_inventing_times():
    raw = "<|speaker:0|>[quiet] hello <|speaker:1|>there [laughter] <|speaker:0|>again"
    p = parse_response({"text": raw}, 3, model="fish-audio/transcribe-1-pro")
    assert p.segments == [Segment("0", "hello"), Segment("1", "there"), Segment("0", "again")]
    assert p.metadata["raw_transcript"]["text"] == raw
    ref = [Segment("a", "hello", 0, 1), Segment("b", "there", 1, 2),
           Segment("a", "again", 2, 3)]
    result = score_record({"reference": ref, "reference_activity": ref, "duration": 3}, p)
    assert result["wer"]["errors"] == result["cpwer"]["errors"] == 0
    assert result["tcpwer"] is None and result["der"] is None and result["coverage"] is None
    ordinary = parse_response({"text": raw}, 3, model="other/model")
    assert ordinary.segments[0].text == raw


def test_fish_alignment_gives_coverage_but_not_guessed_speaker_times():
    p = parse_response({"text": "<|speaker:0|>hello <|speaker:1|>there", "segments": [
        {"text": "hello there", "start": 0, "end": 2},
    ]}, 3, model="fish-audio/transcribe-1-pro")
    ref = [Segment("a", "hello", 0, 1), Segment("b", "there", 1, 2)]
    result = score_record({"reference": ref, "reference_activity": ref, "duration": 3}, p)
    assert result["cpwer"]["errors"] == 0
    assert result["coverage"]["ratio"] == 1
    assert result["tcpwer"] is None and result["der"] is None


def test_fish_unlabelled_prefix_is_preserved():
    p = parse_response({"text": "hello <|speaker:2|>there"}, 2,
                       model="fish-audio/transcribe-1-pro")
    assert p.segments == [Segment("unassigned", "hello"), Segment("2", "there")]


@pytest.mark.parametrize("text", ["<|speaker:0|>hello ] broken", "<|speaker:0|>hi <|speaker:x|>x"])
def test_malformed_fish_annotations_are_not_silently_scored(text):
    with pytest.raises(ValueError, match="malformed_fish_annotation"):
        parse_response({"text": text}, 2, model="fish-audio/transcribe-1-pro")


def test_fish_incomplete_final_annotation_is_recorded_and_excluded():
    raw = "<|speaker:0|>hello [laught"
    p = parse_response({"text": raw}, 2, model="fish-audio/transcribe-1-pro")
    assert p.segments == [Segment("0", "hello")]
    assert p.metadata["annotation_tail_incomplete"]
    assert p.metadata["raw_transcript"]["text"] == raw


def test_fish_annotation_only_response_counts_as_deleted_speech():
    p = parse_response({"text": "<|speaker:0|>[noise]"}, 2,
                       model="fish-audio/transcribe-1-pro")
    ref = [Segment("a", "hello", 0, 1)]
    result = score_record({"reference": ref, "reference_activity": ref, "duration": 2}, p)
    assert result["cpwer"]["deletions"] == 1


def test_point_aligned_word_is_scored_without_inventing_speech_duration():
    p = parse_response({"text": "a word", "words": [
        {"word": "a", "start": 0.5, "end": 0.5, "speaker": 0},
        {"word": "word", "start": 0.5, "end": 1, "speaker": 0},
    ]}, 2)
    ref = [Segment("a", "a word", 0, 1)]
    result = score_record({"reference": ref, "reference_activity": ref, "duration": 2}, p)
    for key in ("wer", "cpwer", "tcpwer"):
        assert result[key]["errors"] == 0
    assert result["coverage"]["ratio"] == 0.5
    assert result["der"]["missed detection"] == 0.5


def test_http_error_diagnostics_do_not_retain_arbitrary_message_text():
    body = json.dumps({"error": {"message": "private-value: verbose_json not supported"}})
    result = error_diagnostic(body)
    assert result == {"mentions": ["verbose_json", "not supported"]}
    assert "private-value" not in json.dumps(result)


def test_provider_options_cannot_forward_arbitrary_configuration():
    base = {"model": "x-ai/grok-stt-1.0", "max_requests": 1, "max_audio_seconds_total": 60,
            "provider_options": PROVIDER_OPTIONS["x-ai/grok-stt-1.0"]}
    validate_options(base)
    for bad in ({"xai": {"api_key": "not-permitted"}}, {"xai": {"diarize": 1}}):
        with pytest.raises(ValueError, match="reviewed"):
            validate_options({**base, "provider_options": bad})


@pytest.mark.parametrize("model", list(PROVIDER_OPTIONS))
def test_candidate_provider_options_reach_the_request(model, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-only")
    monkeypatch.setenv("OPENROUTER_ALLOW_PAID_REQUESTS", "1")
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(16000), 16000)
    requests = []

    class Response(io.BytesIO):
        headers = {}

    class Transport:
        def open(self, request, timeout):
            requests.append(json.loads(request.data))
            return Response(b'{"text":"hello"}')

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Transport())
    adapter = OpenRouter({"model": model, "max_requests": 1, "max_audio_seconds_total": 1,
                          "provider_options": PROVIDER_OPTIONS[model]})
    adapter.load()
    adapter.transcribe(path, 1, "en")
    assert len(requests) == 1
    assert requests[0]["provider"] == {"options": PROVIDER_OPTIONS[model]}
    assert "fixture-only" not in json.dumps(requests)
