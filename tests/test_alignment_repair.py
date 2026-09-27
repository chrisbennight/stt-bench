import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from speaker_benchmark.schema import Prediction, Segment, read_json, write_json

spec = importlib.util.spec_from_file_location(
    "alignment_repair", Path(__file__).resolve().parents[1] / "scripts/repair_local_alignment.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_repair_reuses_transcript_preserves_original_and_counts_all_work(tmp_path, monkeypatch):
    directory = tmp_path / "model"
    (directory / "predictions").mkdir(parents=True)
    write_json(tmp_path / "completion.json", {"all_complete": False})
    write_json(tmp_path / "plan.json", {"models": [{"id": "model", "options": {}}]})
    write_json(directory / "job.json", {"records": [
        {"id": "clip", "audio": "audio.wav", "duration": 1, "language": "English"},
    ]})
    original = {
        "id": "clip", "duration": 1, "seed": 3, "status": "invalid_output",
        "raw_output": "£17", "wall_seconds": 2, "cuda_peak": None,
        "metadata": {"reason": "alignment_changed_transcript"},
    }
    path = directory / "predictions/clip.json"
    write_json(path, original)

    class Saved:
        def __init__(self, options):
            pass

        def load(self):
            pass

        def transcribe(self, audio, duration, language):
            assert self.text == "£17"
            return Prediction([Segment("a", self.text, 0, 1)])

    monkeypatch.setattr(module, "SavedTranscript", Saved)
    monkeypatch.setattr(module, "cuda_stats", lambda **kwargs: None)
    monkeypatch.setattr(module, "seed_record", lambda identifier: None)
    times = iter([10, 13])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(module, "report", lambda path: [{"completed": 1, "recordings": 1}])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: None),
    ))
    assert module.repair(tmp_path) == [{"model": "model", "clip": "clip"}]
    replacement = read_json(path)
    assert replacement["wall_seconds"] == 5
    assert replacement["prediction"]["segments"][0]["text"] == "£17"
    assert read_json(tmp_path / "reprocessing/model/clip.json") == original
    with pytest.raises(FileExistsError):
        module.repair(tmp_path)
