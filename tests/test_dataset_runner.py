import json
import sys
import zipfile
from dataclasses import asdict

import numpy as np
import pytest
import soundfile as sf

from speaker_benchmark.datasets import ami_words, write_windows
from speaker_benchmark.runner import aggregate, plan, report, run
from speaker_benchmark.schema import Prediction, Segment, load_manifest, write_json
from speaker_benchmark.scoring import score_record
from speaker_benchmark.worker import run_job


def test_windows_do_not_duplicate_boundary_words(tmp_path):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(4000), 1000)
    ref = [Segment("a", "first", 0.5, 1), Segment("b", "boundary", 1.9, 2.1)]
    rows = write_windows(audio, ref, ref, tmp_path, "meeting", 2, {})
    assert [s["text"] for r in rows for s in r["reference"]] == ["first", "boundary"]
    assert sum(len(r["reference_activity"]) for r in rows) == 3


def test_ami_zip_reads_only_word_tags_and_rejects_entities(tmp_path):
    path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "words/ES2004a.A.words.xml",
            '<root><w starttime="0" endtime="1">hi</w>'
            '<w punc="true">.</w><noise starttime="1" endtime="2"/></root>',
        )
    assert ami_words(path, "ES2004a")[0].text == "hi"


def test_worker_never_receives_reference_and_saves_prediction(tmp_path, monkeypatch):
    class FixtureAdapter:
        max_duration = 8

        def load(self):
            pass

        def transcribe(self, audio, duration, language):
            assert duration == 4
            return Prediction([Segment("s", "hello", 0, 1)])

    monkeypatch.setattr("speaker_benchmark.worker.create_adapter", lambda *args: FixtureAdapter())
    path = tmp_path / "job.json"
    write_json(
        path,
        {
            "adapter": "fixture",
            "options": {},
            "output": str(tmp_path),
            "records": [{"id": "one", "audio": "unused.wav", "duration": 4}],
        },
    )
    run_job(path)
    output = json.loads((tmp_path / "predictions" / "one.json").read_text())
    assert output["status"] == "ok"
    assert output["wall_seconds"] >= 0
    assert output["prediction"]["segments"][0]["text"] == "hello"


def test_worker_retains_invalid_output_and_continues_with_stable_seeds(tmp_path, monkeypatch):
    from speaker_benchmark.adapters.common import InvalidModelOutput
    from speaker_benchmark.worker import seed_record

    class FixtureAdapter:
        max_duration = None

        def load(self):
            pass

        def transcribe(self, audio, duration, language):
            if audio == "bad.wav":
                raise InvalidModelOutput('[{"incomplete":', {"generation_limit_hit": True})
            return Prediction([Segment("s", "hello", 0, 1)])

    monkeypatch.setattr("speaker_benchmark.worker.create_adapter", lambda *args: FixtureAdapter())
    job = tmp_path / "job.json"
    write_json(
        job,
        {
            "adapter": "fixture",
            "options": {},
            "output": str(tmp_path),
            "records": [
                {"id": "bad", "audio": "bad.wav", "duration": 2},
                {"id": "good", "audio": "good.wav", "duration": 2},
            ],
        },
    )
    run_job(job)
    bad = json.loads((tmp_path / "predictions/bad.json").read_text())
    good = json.loads((tmp_path / "predictions/good.json").read_text())
    assert bad["status"] == "invalid_output" and good["status"] == "ok"
    assert bad["raw_output"] == '[{"incomplete":'
    assert bad["metadata"]["generation_limit_hit"]
    assert bad["wall_seconds"] >= 0
    seed_record("unrelated")
    assert seed_record("good") == good["seed"]
    first = np.random.random()
    seed_record("good")
    assert np.random.random() == first


def test_invalid_output_counts_as_failure_and_retains_runtime():
    ref = [Segment("a", "hello", 0, 1)]
    record = {"id": "one", "duration": 2, "reference": ref, "reference_activity": ref}
    row = {
        "id": "one",
        "status": "invalid_output",
        "wall_seconds": 10,
        "metadata": {"generation_limit_hit": True},
        "scores": score_record(record, Prediction([], timing="failed")),
    }
    result = aggregate("model", [row], [record])
    assert result["completed"] == 0 and result["invalid_output"] == 1
    assert result["cpwer"] == 1 and result["generation_limit_hits"] == 1
    assert result["rtf_attempted"] == 5 and result["rtf_completed"] is None


def test_manifest_rejects_duplicate_ids_and_multichannel(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(400), 1000)
    manifest = tmp_path / "manifest.jsonl"
    row = {"id": "a", "audio": "a.wav", "reference": []}
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(manifest)
    sf.write(tmp_path / "a.wav", np.zeros((400, 2)), 1000)
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="single audio channel"):
        load_manifest(manifest)


def test_report_keeps_missing_recordings_and_weights_by_words(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(4000), 1000)
    ref = [Segment("a", "one two", 0, 2)]
    records = [
        {"id": name, "audio": "a.wav", "reference": [asdict(s) for s in ref]}
        for name in ["one", "two"]
    ]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    write_json(
        tmp_path / "plan.json",
        {"models": [{"id": "fixture"}], "protocol": {"tcp_collar": 5, "der_collar": 0}},
    )
    folder = tmp_path / "fixture"
    folder.mkdir()
    (folder / "predictions").mkdir()
    write_json(
        folder / "predictions" / "one.json",
        {"id": "one", "status": "ok", "wall_seconds": 1, "prediction": Prediction(ref).to_dict()},
    )
    result = report(tmp_path)[0]
    assert result["completed"] == 1 and result["failed_or_unsupported"] == 1
    assert result["cpwer"] == 0.5
    assert result["speech_time_coverage"] == 0.5


def test_aggregate_cannot_publish_partial_timing_as_full_score():
    ref = [Segment("a", "hello", 0, 1)]
    record = {"id": "one", "duration": 2, "reference": ref, "reference_activity": ref}
    prediction = Prediction([Segment("s", "hello")], timing="unavailable")
    row = {
        "id": "one",
        "status": "ok",
        "wall_seconds": 1,
        "prediction": prediction.to_dict(),
        "scores": score_record(record, prediction),
    }
    result = aggregate("streaming", [row], [record])
    assert result["cpwer"] == 0
    assert result["tcpwer"] is None and result["der"] is None


def test_config_rejects_credentials(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros(400), 1000)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"id": "a", "audio": "a.wav", "reference": []}) + "\n"
    )
    config = {
        "models": [
            {
                "id": "a",
                "adapter": "moss",
                "python": sys.executable,
                "options": {"token": "not-a-real-credential"},
            }
        ]
    }
    write_json(tmp_path / "config.json", config)
    with pytest.raises(ValueError, match="credentials"):
        plan(tmp_path / "config.json", tmp_path / "manifest.jsonl")


def test_real_subprocess_plugin_integration(tmp_path, monkeypatch):
    # Install only entry-point metadata on a temporary import path, without pip or network access.
    plugin = tmp_path / "fixture_plugin.py"
    plugin.write_text("""from speaker_benchmark.adapters.common import Adapter
from speaker_benchmark.schema import Prediction, Segment
class Fixture(Adapter):
    def load(self):
        pass
    def transcribe(self, audio_path, duration, language):
        return Prediction([Segment("speaker", "hello", 0, 1)])
""")
    metadata = tmp_path / "fixture_plugin-0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: fixture-plugin\nVersion: 0.1\n")
    (metadata / "entry_points.txt").write_text(
        "[speaker_benchmark.adapters]\nfixture = fixture_plugin:Fixture\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    sf.write(tmp_path / "a.wav", np.zeros(2000), 1000)
    reference = [asdict(Segment("a", "hello", 0, 1))]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "job", "audio": "a.wav", "reference": reference}) + "\n")
    # A symlink must remain a symlink in argv; resolving venv Python bypasses its environment.
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "models": [
                {"id": "fixture", "adapter": "fixture", "python": sys.executable, "options": {}}
            ]
        },
    )
    result = run(config, manifest, tmp_path / "run")
    assert result[0]["cpwer"] == 0
    assert result[0]["completed"] == 1
    job = json.loads((tmp_path / "run" / "fixture" / "job.json").read_text())
    assert "reference" not in job["records"][0]
    write_json(
        config,
        {
            "models": [
                {"id": "fixture", "adapter": "fixture", "python": str(executable), "options": {}}
            ]
        },
    )
    blueprint, _ = plan(config, manifest)
    assert blueprint["models"][0]["python"] == str(executable)
