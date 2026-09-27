import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speaker_benchmark.schema import Prediction, Segment, digest, read_json, write_json

spec = importlib.util.spec_from_file_location(
    "local_preflight", Path(__file__).resolve().parents[1] / "scripts/run_local_preflight.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("first_ok", [True, False])
def test_preflight_retains_first_clip_and_gates_remaining_inference(
    tmp_path, monkeypatch, first_ok,
):
    sf.write(tmp_path / "audio.wav", np.zeros(32000), 16000)
    reference = [{"speaker": "a", "text": "hello", "start": 0.1, "end": 0.5}]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps({
        "id": name, "audio": "audio.wav", "reference": reference,
        "audio_sha256": digest(tmp_path / "audio.wav"),
    }) + "\n" for name in ["first", "second"]))
    config = tmp_path / "config.json"
    write_json(config, {"models": [
        {"id": "fixture", "adapter": "fixture", "python": sys.executable},
    ]})
    seen = []

    def workers(jobs, parallel, timeout):
        assert parallel == 1
        for directory, argv in jobs:
            job = read_json(argv[-1])
            (directory / "predictions").mkdir(exist_ok=True)
            for record in job["records"]:
                seen.append(record["id"])
                assert "reference" not in record
                valid = first_ok or record["id"] != "first"
                prediction = Prediction([Segment("a", "hello", 0.1, 0.5)])
                write_json(directory / "predictions" / f"{record['id']}.json", {
                    "id": record["id"], "status": "ok" if valid else "invalid_output",
                    "prediction": prediction.to_dict(), "wall_seconds": 1,
                })
            write_json(directory / "worker.json", {
                "exit_code": 0, "timed_out": False, "sampled_process_tree_peak_rss_bytes": 100,
            })
            write_json(directory / "environment.json", {})

    monkeypatch.setattr(module, "run_workers", workers)
    output = tmp_path / "output"
    result = module.run(config, manifest, manifest, output)
    assert seen == (["first", "second"] if first_ok else ["first"])
    assert result[0]["completed"] == (2 if first_ok else 0)
    assert read_json(output / "completion.json")["all_complete"] is first_ok


def test_mismatched_baseline_prevents_inference(tmp_path, monkeypatch):
    sf.write(tmp_path / "audio.wav", np.zeros(16000), 16000)
    manifest, baseline = tmp_path / "manifest.jsonl", tmp_path / "baseline.jsonl"
    row = {"id": "clip", "audio": "audio.wav", "reference": []}
    manifest.write_text(json.dumps(row) + "\n")
    baseline.write_text(json.dumps({**row, "id": "different"}) + "\n")
    config = tmp_path / "config.json"
    write_json(config, {"models": [{"id": "fixture", "adapter": "fixture"}]})
    with pytest.raises(ValueError, match="differ"):
        module.run(config, manifest, baseline, tmp_path / "output")
    assert not (tmp_path / "output").exists()
