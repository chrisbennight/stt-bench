import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from speaker_benchmark.adapters.common import InvalidModelOutput
from speaker_benchmark.adapters.local_asr import AlignedPyannote, NemoPyannote, aligned_words
from speaker_benchmark.schema import Segment


def item(text, start, end):
    return SimpleNamespace(text=text, start_time=start, end_time=end)


def test_alignment_preserves_words_and_point_timestamps_at_boundary():
    words, clipped = aligned_words(
        "Hello, there!", [item("hello", 0, 1), item("there", 2, 2.1)], 2,
    )
    assert [w.text for w in words] == ["hello", "there"]
    assert words[1].start == words[1].end == 2
    assert clipped == 1


def test_alignment_restores_currency_symbols_without_changing_words_or_times():
    words, _ = aligned_words("cost £17 or €25", [
        item("cost", 0, 0.2), item("17", 0.2, 0.4),
        item("or", 0.4, 0.6), item("25", 0.6, 0.8),
    ], 1)
    assert [w.text for w in words] == ["cost", "£17", "or", "€25"]
    assert words[1].start == 0.2 and words[1].end == 0.4


def test_nemotron_manifest_forces_language_and_inference_prompt_mode(monkeypatch, tmp_path):
    manifests = []

    class ASR:
        def transcribe(self, inputs, *, batch_size, return_hypotheses, target_lang):
            assert batch_size == 1 and return_hypotheses and target_lang == "en-US"
            path = Path(inputs[0])
            manifests.append(path)
            record = json.loads(path.read_text())
            assert record["lang"] == record["target_lang"] == "en-US"
            assert record["prompt_mode"] == "langID"
            assert record["text"] == "" and record["duration"] == 1
            return [SimpleNamespace(text="hello<en-US>")]

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    adapter = NemoPyannote({"target_lang": "en-US"})
    adapter.asr = ASR()
    text, metadata = adapter.recognize(tmp_path / "audio.wav", np.zeros(16000), "English")
    assert text == "hello" and metadata["language_tag_removed"]
    assert metadata["raw_asr_text"] == "hello<en-US>"
    assert not manifests[0].exists()


@pytest.mark.parametrize("items", [
    [item("hello", 0, 1)],
    [item("hello", 0, 1), item("there", 2, 1)],
    [item("hello", 0, 1), item("there", 1, float("inf"))],
])
def test_alignment_never_silently_loses_words_or_accepts_invalid_times(items):
    with pytest.raises(InvalidModelOutput):
        aligned_words("hello there", items, 2)


def test_pipeline_uses_full_clip_and_preserves_overlapping_diarization(tmp_path):
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(32000), 16000)

    class Aligner:
        def align(self, *, audio, text, language):
            assert len(audio[0]) == 32000
            assert text == "hello there" and language == "English"
            return [[item("hello", 0.1, 0.4), item("there", 1.2, 1.5)]]

    class Pipeline(AlignedPyannote):
        def recognize(self, audio_path, audio, language):
            return "Hello, there!", {"asr_mode": "test"}

        def diarize(self, audio_path, audio):
            return (
                [Segment("a", "", 0, 1.5), Segment("b", "", 1, 2)],
                [Segment("a", "", 0, 1), Segment("b", "", 1, 2)],
            )

    adapter = Pipeline({})
    adapter.aligner = Aligner()
    prediction = adapter.transcribe(audio_path, 2, "English")
    assert [s.speaker for s in prediction.segments] == ["a", "b"]
    assert prediction.activity[0].end > prediction.activity[1].start
    assert prediction.metadata["raw_text"] == "Hello, there!"
    assert prediction.timing == "forced_alignment"
