"""Shared adapter contracts and strict output parsing."""

import json
import re
from pathlib import Path

from speaker_benchmark.schema import Segment

SOUND_ANNOTATION_PATTERN = re.compile(r"\[[^\[\]\r\n]+\]")


class InvalidModelOutput(ValueError):
    """Generation finished, but its output cannot satisfy the prediction schema."""

    def __init__(self, raw_output, metadata):
        super().__init__("Model output could not be parsed")
        self.raw_output = raw_output
        self.metadata = metadata


class Adapter:
    max_duration = None

    def __init__(self, options):
        self.options = options
        self.device = options.get("device", "cuda:0")

    def load(self):
        raise NotImplementedError

    def transcribe(self, audio_path, duration, language):
        raise NotImplementedError

    def local_model(self, key="model"):
        path = Path(self.options[key]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configure an existing local {key} snapshot")
        return str(path)

    def dtype(self):
        import torch

        return torch.bfloat16 if self.device.startswith("cuda") else torch.float32


def parse_moss(text):
    pattern = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.S)
    segments, cursor = [], 0
    for match in pattern.finditer(text):
        if text[cursor : match.start()].strip():
            raise ValueError("Unparsed text in MOSS output")
        start, speaker, words, end = match.groups()
        segments.append(Segment(speaker, words, float(start), float(end)))
        cursor = match.end()
    if text[cursor:].strip():
        raise ValueError("Incomplete or invalid MOSS output")
    return segments


def parse_vibe(text):
    text = text.strip()
    if text.startswith("assistant\n"):
        text = text[len("assistant\n") :].strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("VibeVoice must return JSON segments")
    rows = []
    for item in data:

        def get(*keys):
            for key in keys:
                if key in item:
                    return item[key]
            raise ValueError("Missing required VibeVoice field")

        content = remove_sound_annotations(get("Content", "text"))
        # Explicit sound-event records remain in raw_output but are not spoken words.
        if not content.strip():
            continue
        rows.append(
            Segment(
                str(
                    next(
                        (item[k] for k in ("Speaker ID", "Speaker", "speaker_id") if k in item),
                        "unassigned",
                    )
                ),
                content,
                float(get("Start time", "Start", "start_time")),
                float(get("End time", "End", "end_time")),
            )
        )
    return rows


def remove_sound_annotations(text):
    """VibeVoice square-bracket annotations describe non-speech events, including silence."""
    return SOUND_ANNOTATION_PATTERN.sub(" ", text)


def assign_words(words, activity):
    """Assign by maximum temporal overlap; preserve unmatched words explicitly."""
    output = []
    for word in words:
        scores = {}
        for turn in activity:
            overlap = max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
            scores[turn.speaker] = scores.get(turn.speaker, 0) + overlap
        best = sorted(scores, key=lambda spk: (-scores[spk], spk))
        speaker = best[0] if best and scores[best[0]] > 0 else "unassigned"
        output.append(Segment(speaker, word.text, word.start, word.end))
    return output
