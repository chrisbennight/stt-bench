import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from speaker_benchmark.adapters.openrouter import parse_response

spec = importlib.util.spec_from_file_location(
    "probe_validator", Path(__file__).resolve().parents[1] / "scripts/validate_openrouter_probe.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    record = {"id": "clip", "duration": 2, "audio_sha256": "hash", "reference": [
        {"speaker": "a", "text": "hello", "start": 0, "end": 1},
    ], "reference_activity": [{"speaker": "a", "text": "", "start": 0, "end": 1}]}
    config = {"models": [{"id": "model", "options": {"response_format": "verbose_json"}}]}
    audit = {"probe_record": deepcopy(record), "models": [
        {"id": "model", "required_probe_metrics": ["wer", "cpwer", "tcpwer", "der", "coverage"]},
    ]}
    return config, audit, deepcopy(config), [record]


def test_http_success_without_requested_speakers_cannot_pass_probe():
    args = fixture()
    predictions = {"model": [{"id": "clip", "status": "ok",
                              "prediction": parse_response({"text": "hello"}, 2).to_dict()}]}
    findings = module.validate_probe(*args, predictions)
    assert findings[0]["metrics"] == ["cpwer", "tcpwer", "der", "coverage"]


def test_probe_accepts_actual_structured_output():
    args = fixture()
    p = parse_response({"text": "hello", "segments": [
        {"speaker": 0, "text": "hello", "start": 0, "end": 1},
    ]}, 2)
    assert module.validate_probe(*args, {"model": [
        {"id": "clip", "status": "ok", "prediction": p.to_dict()},
    ]}) == []


def test_probe_rejects_changed_options_or_audio():
    args = fixture()
    args[2]["models"][0]["options"]["response_format"] = "json"
    with pytest.raises(ValueError, match="options differ"):
        module.validate_probe(*args, {"model": []})
    args = fixture()
    args[3][0]["audio_sha256"] = "different"
    with pytest.raises(ValueError, match="audio and references differ"):
        module.validate_probe(*args, {"model": []})


def test_probe_rejects_failure_and_duplicate_trials():
    args = fixture()
    row = {"id": "clip", "status": "inference_failed"}
    for rows in ([row], [row, row], []):
        assert module.validate_probe(*args, {"model": rows})[0]["reason"] == "probe_not_successful"
