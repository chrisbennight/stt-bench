import pytest

from speaker_benchmark.adapters.common import assign_words, parse_moss, parse_vibe
from speaker_benchmark.adapters.joint import parse_stream_chunk, parse_stream_events
from speaker_benchmark.schema import Segment


def test_moss_parser_preserves_overlap():
    rows = parse_moss("[0.1][S01]hello[2.0][1.0][S02]yes[1.5]")
    assert len(rows) == 2
    assert rows[1].start < rows[0].end


@pytest.mark.parametrize("text", ["garbage", "[0][S01]unfinished", "[0][S01]ok[1] trailing"])
def test_moss_parser_rejects_partial_output(text):
    with pytest.raises(ValueError):
        parse_moss(text)


def test_vibe_json_parser_does_not_evaluate_code():
    with pytest.raises(ValueError):
        parse_vibe("__import__('os').system('false')")
    row = parse_vibe('[{"Start time":0,"End time":1,"Speaker ID":2,"Content":"hi"}]')[0]
    assert row.speaker == "2"


def test_streaming_speaker_carries_across_chunks_and_silence():
    a, speaker = parse_stream_chunk("\n Speaker 0:hello ", None)
    b, speaker = parse_stream_chunk("there.\n Speaker 1:yes", speaker)
    silence, speaker = parse_stream_chunk("", speaker)
    assert [s.speaker for s in a + b] == ["0", "0", "1"]
    assert silence == [] and speaker == "1"
    unattributed, speaker = parse_stream_chunk("unattributed words", None)
    assert unattributed == [Segment("unassigned", "unattributed words")]
    assert speaker == "unassigned"


def test_vibe_assistant_prefix_and_explicit_sound_events():
    text = (
        'assistant\n[{"Start":0,"End":1,"Content":"[Environmental Sounds]"},'
        '{"Start":1,"End":2,"Speaker":0,"Content":"hello"}]'
    )
    assert parse_vibe(text) == [Segment("0", "hello", 1, 2)]
    assert parse_vibe('[{"Start":0,"End":1,"Content":"unattributed speech"}]') == [
        Segment("unassigned", "unattributed speech", 0, 1)
    ]


def test_vibe_noise_and_streaming_silence_are_annotations_not_words():
    assert parse_vibe('[{"Start":0,"End":1,"Content":"[Noise]"}]') == []
    assert parse_stream_chunk("[Silence]", None) == ([], None)
    segments, speaker = parse_stream_chunk("[Silence]\n Speaker 2:Hello [Noise] there", None)
    assert speaker == "2"
    assert segments[0].text.split() == ["Hello", "there"]


def test_unmatched_words_are_not_deleted():
    words = [Segment("?", "one", 0, 1), Segment("?", "two", 2, 3)]
    turns = [Segment("a", "", 0, 0.75), Segment("b", "", 0.75, 1)]
    assigned = assign_words(words, turns)
    assert [s.speaker for s in assigned] == ["a", "unassigned"]
    assert [s.text for s in assigned] == ["one", "two"]


def test_streaming_annotations_words_and_labels_may_span_chunks():
    chunks = ["[Environmental ", "", "Sounds] \n Speak", "er 1:hel", "lo there."]
    events = [
        {"text": text, "emitted_seconds": float(i + 1), "has_text": True}
        for i, text in enumerate(chunks)
    ]
    segments, corrected = parse_stream_events(events)
    assert segments == [Segment("1", "hello there.")]
    assert [e["has_text"] for e in corrected] == [False, False, False, True, True]
    assert [e["emitted_seconds"] for e in corrected] == [1, 2, 3, 4, 5]
    assert "".join(e["text"] for e in corrected) == "".join(chunks)
    assert all(e["has_text"] for e in events)


def test_streaming_transcript_is_independent_of_chunk_boundaries():
    text = "[Silence]\n Speaker 2:Hello there.\n Speaker 1:Yes [Noise] please."
    expected, _ = parse_stream_chunk(text, None)
    for boundary in range(len(text) + 1):
        segments, _ = parse_stream_events([{"text": text[:boundary]}, {"text": text[boundary:]}])
        assert segments == expected


@pytest.mark.parametrize("start,end", [(float("nan"), 2), (0, float("inf")), (2, 1), (-1, 2)])
def test_invalid_timestamps_rejected(start, end):
    with pytest.raises(ValueError):
        Segment("a", "word", start, end)


def test_qwen_pipeline_offsets_alignment_and_keeps_global_speakers(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import soundfile as sf

    from speaker_benchmark.adapters.pipeline import QwenPipeline

    class FakeASR:
        calls = 0

        def transcribe(self, *, audio, language, return_time_stamps):
            assert language == "English" and return_time_stamps
            self.calls += 1
            return [
                SimpleNamespace(
                    text="hello",
                    time_stamps=[SimpleNamespace(text="hello", start_time=0.1, end_time=0.2)],
                )
            ]

    class Pipeline(QwenPipeline):
        def diarize(self, path, audio):
            turns = [Segment("a", "", 0, 1), Segment("b", "", 1, 2)]
            return turns, turns

    audio = tmp_path / "test.wav"
    sf.write(audio, np.zeros(32000), 16000)
    pipeline = Pipeline({"asr_chunk_seconds": 1})
    pipeline.asr = FakeASR()
    result = pipeline.transcribe(str(audio), 2, "English")
    assert pipeline.asr.calls == 2
    assert [s.speaker for s in result.segments] == ["a", "b"]
    assert result.segments[1].start == pytest.approx(1.1)
    assert result.activity[1].start == 1
