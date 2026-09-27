import io
import json
import urllib.error
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from speaker_benchmark.adapters.common import InvalidModelOutput
from speaker_benchmark.adapters.openrouter import (
    ENDPOINT,
    NoRedirect,
    OpenRouter,
    RemoteRequestError,
    parse_response,
    validate_options,
)
from speaker_benchmark.openrouter_catalog import estimate, make_config
from speaker_benchmark.runner import plan, report
from speaker_benchmark.schema import Prediction, Segment, write_json
from speaker_benchmark.scoring import score_record


def options(**kwargs):
    return {"model": "openai/whisper-1", "max_requests": 1, "max_audio_seconds_total": 2, **kwargs}


def test_transcription_without_speakers_is_not_scored_as_diarization():
    prediction = parse_response(
        {
            "text": "hello there",
            "segments": [
                {"start": 0, "end": 1, "text": "hello there"},
            ],
        },
        2,
    )
    refs = [Segment("a", "hello", 0, 0.5), Segment("b", "there", 0.5, 1)]
    result = score_record(
        {"reference": refs, "reference_activity": refs, "duration": 2}, prediction
    )
    assert result["wer"]["errors"] == 0
    assert result["cpwer"] is None and result["der"] is None and result["tcpwer"] is None
    assert result["coverage"]["ratio"] == 1


def test_word_timestamps_do_not_discard_phrase_speakers():
    prediction = parse_response(
        {
            "text": "hello",
            "segments": [
                {"start": 0, "end": 1, "text": "hello", "speaker": 0},
            ],
            "words": [{"start": 0.1, "end": 0.9, "word": "hello"}],
        },
        2,
    )
    assert prediction.segments[0].speaker == "0"
    assert prediction.segments[0].start == 0
    assert prediction.metadata["speaker_labels_available"]


def test_word_speakers_and_plain_text():
    prediction = parse_response(
        {
            "text": "hello",
            "words": [
                {"start": 0.1, "end": 0.9, "word": "hello", "speaker": 2},
            ],
            "usage": {"cost": 0.01, "unknown": "not retained"},
        },
        2,
    )
    assert prediction.segments[0].speaker == "2"
    assert prediction.metadata["usage"] == {"cost": 0.01}
    plain = parse_response({"text": "hello"}, 2)
    assert plain.timing == "unavailable"
    assert not plain.metadata["speaker_labels_available"]


@pytest.mark.parametrize(
    "response",
    [
        {"error": "bad request"},
        {"text": "x", "usage": {"cost": float("nan")}},
        {"text": "x", "segments": [{"text": "x", "start": 0, "end": 4}]},
        {"text": "x", "words": ["not a word object"]},
    ],
)
def test_invalid_api_output_is_rejected(response):
    with pytest.raises(ValueError):
        parse_response(response, 2)


def test_deepgram_boundary_word_is_clipped_without_losing_text_or_raw_timing():
    data = {
        "text": "test",
        "words": [
            {"word": "test", "speaker": 0, "start": 59.794937, "end": 61.314938},
        ],
        "usage": {"cost": 0.0043},
    }
    prediction = parse_response(data, 60, clip_timestamps=True)
    assert prediction.segments == [Segment("0", "test", 59.794937, 60)]
    assert prediction.timing == "native_clipped"
    assert prediction.metadata["raw_transcript"]["words"][0]["end"] == 61.314938
    assert prediction.metadata["timestamp_adjustments"][0]["original_end"] == 61.314938
    assert data["words"][0]["end"] == 61.314938
    with pytest.raises(ValueError, match="transcript_extends_past_audio"):
        parse_response(data, 60)


@pytest.mark.parametrize(
    "start,end", [(60, 61), (-2, -1), (1, 1), (2, 1), (0, float("nan")), (None, 1)]
)
def test_clipping_does_not_hide_invalid_or_wholly_outside_intervals(start, end):
    with pytest.raises(ValueError):
        parse_response(
            {
                "text": "test",
                "segments": [
                    {"text": "test", "speaker": 0, "start": start, "end": end},
                ],
            },
            60,
            clip_timestamps=True,
        )


def test_phrase_boundary_clipping_and_unchanged_valid_timestamps():
    data = {
        "text": "hello there",
        "segments": [
            {"text": "hello", "speaker": 0, "start": -0.1, "end": 1},
            {"text": "there", "speaker": 1, "start": 1, "end": 2},
        ],
    }
    prediction = parse_response(data, 2, clip_timestamps=True)
    assert prediction.segments == [Segment("0", "hello", 0, 1), Segment("1", "there", 1, 2)]
    assert len(prediction.metadata["timestamp_adjustments"]) == 1
    assert prediction.metadata["timestamp_adjustments"][0]["source"] == "segments"


def test_rejected_response_keeps_safe_diagnostics_and_redacts_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-credential-never-persist")
    monkeypatch.setenv("OPENROUTER_ALLOW_PAID_REQUESTS", "1")
    sf.write(tmp_path / "clip.wav", np.zeros(16000), 16000)

    class Transport:
        def open(self, request, timeout):
            return Response(
                json.dumps(
                    {
                        "text": "test-credential-never-persist",
                        "words": [
                            {
                                "word": "test",
                                "start": 0,
                                "end": float("nan"),
                                "request_headers": {"Authorization": "must-not-save"},
                            },
                        ],
                        "usage": {"cost": 0.0043},
                        "error": "must-not-save",
                        "headers": {"Authorization": "must-not-save"},
                    }
                ).encode()
            )

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Transport())
    adapter = OpenRouter(options(model="deepgram/nova-3"))
    adapter.load()
    with pytest.raises(InvalidModelOutput) as raised:
        adapter.transcribe(tmp_path / "clip.wav", 1, "en")
    exc = raised.value
    saved = json.dumps({"raw_output": exc.raw_output, "metadata": exc.metadata}, allow_nan=False)
    assert "must-not-save" not in saved and "test-credential-never-persist" not in saved
    assert exc.metadata["reason"] == "invalid_transcript_timestamps"
    assert exc.metadata["response_usage"] == {"cost": 0.0043}
    assert exc.raw_output["words"][0]["end"] == "[non-finite number]"


@pytest.mark.parametrize(
    "changes",
    [
        {"api_key": "must-not-be-in-config"},
        {"max_requests": True},
        {"request_timeout_seconds": float("inf")},
        {"model": "../../outside"},
        {"timestamp_granularities": ["word"]},
        {"diarization": True},
    ],
)
def test_options_reject_secrets_and_invalid_limits(changes):
    with pytest.raises(ValueError):
        validate_options(options(**changes))


def test_explicit_spending_opt_in(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ALLOW_PAID_REQUESTS", raising=False)
    with pytest.raises(ValueError, match="reviewing the estimate"):
        OpenRouter(options()).load()


class Response(io.BytesIO):
    headers = {"X-Generation-Id": "gen-fixture"}


def test_one_request_no_secret_in_artifact_and_audio_allowance(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-credential-never-persist")
    monkeypatch.setenv("OPENROUTER_ALLOW_PAID_REQUESTS", "1")
    sf.write(tmp_path / "clip.wav", np.zeros(32000), 16000)
    calls = []

    class Transport:
        def open(self, request, timeout):
            calls.append(request)
            assert request.full_url == ENDPOINT
            body = json.loads(request.data)
            assert body["language"] == "en"
            assert "reference" not in body
            assert request.headers["Authorization"] == "Bearer test-credential-never-persist"
            assert timeout == 90
            return Response(b'{"text":"hello","usage":{"cost":0.01}}')

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Transport())
    adapter = OpenRouter(options())
    adapter.load()
    prediction = adapter.transcribe(tmp_path / "clip.wav", 2, "English")
    assert "test-credential-never-persist" not in json.dumps(prediction.to_dict())
    assert prediction.metadata["generation_id"] == "gen-fixture"
    with pytest.raises(ValueError, match="allowance"):
        adapter.transcribe(tmp_path / "clip.wav", 2, "English")
    assert len(calls) == 1


def test_http_failure_is_not_retried_or_logged_verbatim(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-credential-never-persist")
    monkeypatch.setenv("OPENROUTER_ALLOW_PAID_REQUESTS", "1")
    sf.write(tmp_path / "clip.wav", np.zeros(16000), 16000)
    calls = []

    class Transport:
        def open(self, request, timeout):
            calls.append(request)
            raise urllib.error.HTTPError(ENDPOINT, 429, "sensitive upstream message", {}, None)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Transport())
    adapter = OpenRouter(options())
    adapter.load()
    with pytest.raises(RemoteRequestError) as error:
        adapter.transcribe(tmp_path / "clip.wav", 1, "en")
    assert error.value.http_status == 429
    assert "sensitive" not in str(error.value)
    assert len(calls) == 1
    assert NoRedirect().redirect_request(None, None, 302, None, None, "https://example.com") is None


def test_remote_model_id_is_not_resolved_as_a_path_and_limits_are_preflighted(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(16000), 16000)
    row = {"id": "a", "audio": "a.wav", "reference": [asdict(Segment("a", "hi", 0, 1))]}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    config = tmp_path / "config.json"
    write_json(config, {"models": [{"id": "api", "adapter": "openrouter", "options": options()}]})
    blueprint, _ = plan(config, manifest)
    assert blueprint["models"][0]["options"]["model"] == "openai/whisper-1"
    assert blueprint["models"][0]["paths_exist"] == {}
    manifest.write_text(json.dumps(row) + "\n" + json.dumps({**row, "id": "b"}) + "\n")
    with pytest.raises(ValueError, match="allowance"):
        plan(config, manifest)


def test_catalog_units_and_unknown_models_are_not_guessed():
    catalog = {
        "models": [
            {
                "id": "microsoft/mai-transcribe-2",
                "billing_unit": "hour",
                "pricing": {"prompt": ".1", "completion": "0"},
            },
            {
                "id": "google/chirp-3",
                "billing_unit": "second",
                "pricing": {"prompt": ".000266666666667", "completion": "0"},
            },
            {
                "id": "new/model",
                "billing_unit": "unreviewed",
                "pricing": {"prompt": "1", "completion": "0"},
            },
        ]
    }
    result = estimate(catalog, 3600)
    assert result["models"][0]["estimated_usd"] == 0.1
    assert result["models"][1]["estimated_usd"] == pytest.approx(0.96)
    assert result["models"][2]["estimated_usd"] is None
    assert not result["complete"]
    assert result["estimated_total_usd"] == pytest.approx(1.06)
    for model in make_config(catalog)["models"]:
        validate_options(model["options"])


def test_token_estimates_use_model_specific_rates():
    catalog = {
        "models": [
            {
                "id": "google/gemini-3.5-transcribe",
                "billing_unit": "token",
                "pricing": {"prompt": ".000002", "completion": ".000012"},
            },
            {
                "id": "openai/gpt-4o-transcribe",
                "billing_unit": "token",
                "pricing": {"prompt": ".0000025", "completion": ".00001"},
            },
        ]
    }
    result = estimate(catalog, 60)
    assert result["estimated_total_usd"] == pytest.approx(0.0111)
    catalog["models"][1]["pricing"]["prompt"] = ".1"
    assert not estimate(catalog, 60)["complete"]


def test_report_mixes_local_remote_and_failed_models_without_inventing_speaker_scores(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(16000), 16000)
    reference = [Segment("a", "hello", 0, 1)]
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "id": "a",
                "audio": "a.wav",
                "reference": [asdict(s) for s in reference],
            }
        )
        + "\n"
    )
    write_json(
        tmp_path / "plan.json",
        {
            "models": [
                {"id": "local"},
                {"id": "remote", "adapter": "openrouter"},
                {"id": "failed", "adapter": "openrouter"},
            ],
            "protocol": {"tcp_collar": 5, "der_collar": 0},
        },
    )
    for name, prediction in [
        ("local", Prediction(reference)),
        ("remote", parse_response({"text": "hello", "usage": {"cost": 0.01}}, 1)),
        ("failed", None),
    ]:
        directory = tmp_path / name / "predictions"
        directory.mkdir(parents=True)
        if prediction:
            write_json(
                directory / "a.json",
                {
                    "id": "a",
                    "status": "ok",
                    "wall_seconds": 1,
                    "prediction": prediction.to_dict(),
                },
            )
    rows = report(tmp_path)
    assert rows[0]["cpwer"] == 0
    assert rows[1]["cpwer"] is None
    assert rows[1]["api_reported_cost_usd"] == 0.01 and rows[1]["api_cost_complete"]
    assert rows[2]["cpwer"] is None and rows[2]["wer"] == 1
    assert (tmp_path / "summary.csv").is_file()
