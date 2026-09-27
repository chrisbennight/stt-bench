"""Inspect saved hosted outputs and diagnostic responses without making API requests."""

import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf

from speaker_benchmark.adapters.openrouter import OpenRouter, parse_response

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/openrouter-validated-2026-09-27"
REVIEW = ROOT / "results/openrouter-diarization-review-2026-09-27"


def speaker_paths(value, path=""):
    """Inspect every nested JSON field, including fields outside the benchmark schema."""
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            location = f"{path}/{key}"
            if any(term in key.casefold()
                   for term in ("speaker", "diari", "turn", "utterance", "channel")):
                found.add(location)
            found.update(speaker_paths(child, location))
    elif isinstance(value, list):
        for child in value:
            found.update(speaker_paths(child, path + "[]"))
    return sorted(found)


def adapter_payload(options):
    """Capture actual adapter serialization with synthetic audio and a fake transport."""
    captured = []

    class Response(io.BytesIO):
        headers = {}

    class Capture:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return Response(b'{"text":"fixture"}')

    with TemporaryDirectory() as temporary, patch.dict(
        os.environ, {"OPENROUTER_API_KEY": "audit-fixture"}
    ):
        audio = Path(temporary) / "audio.wav"
        sf.write(audio, np.zeros(16000, dtype=np.int16), 16000, subtype="PCM_16")
        adapter = OpenRouter(options)
        adapter.opener = Capture()
        adapter.transcribe(audio, 1, "en")
    assert len(captured) == 1
    return {k: v for k, v in captured[0].items() if k != "input_audio"}


def audit():
    sources = json.loads((BUNDLE / "sources.json").read_text())["models"]
    models = []
    for source in sources:
        if source.get("excluded"):
            continue
        rows = json.loads((BUNDLE / "scores" / f"{source['model']}.json").read_text())
        assert len(rows) == 94
        counts, top, words, segments = Counter(), Counter(), Counter(), Counter()
        for row in rows:
            counts[row["status"]] += 1
            metadata = row.get("prediction", {}).get("metadata", {})
            raw = metadata.get("raw_transcript", {})
            counts["parsed_labelled"] += bool(metadata.get("speaker_labels_available"))
            counts["raw_structured_labelled"] += any(
                s.get("speaker") is not None
                for name in ("words", "segments") for s in raw.get(name, [])
            )
            counts["inline_fish_labels"] += "<|speaker:" in raw.get("text", "")
            top.update(metadata.get("response_top_level_fields", []))
            words.update(metadata.get("response_field_names", {}).get("words", []))
            segments.update(metadata.get("response_field_names", {}).get("segments", []))
        assert counts["parsed_labelled"] == (
            counts["inline_fish_labels"] if source["api_model"] == "fish-audio/transcribe-1-pro"
            else counts["raw_structured_labelled"]
        )
        models.append({"model": source["api_model"], "counts": dict(counts),
                       "top_fields": dict(top), "word_fields": dict(words),
                       "segment_fields": dict(segments)})

    options = {m["api_model"]: m["options"] for m in sources if not m.get("excluded")}
    diagnostics = []
    for path in sorted((REVIEW / "responses").glob("*.json")):
        result = json.loads(path.read_text())
        request = result["request"]
        raw = result.get("raw_response")
        item = {"name": path.stem, "model": request["model"],
                "http_status": result["http_status"], "reported_cost_usd": None}
        if path.stem.endswith("-enabled"):
            expected = {k: v for k, v in request.items() if k != "input_audio"}
            assert adapter_payload(options[request["model"]]) == expected
            item["matches_benchmark_request_options"] = True
        if raw is not None:
            canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
            item.update(response_json_sha256=hashlib.sha256(canonical).hexdigest(),
                        top_fields=sorted(raw), speaker_paths=speaker_paths(raw),
                        reported_cost_usd=raw.get("usage", {}).get("cost"))
            prediction = parse_response(raw, request["input_audio"]["duration"],
                                        clip_timestamps=True, model=request["model"])
            item["parser_labels_available"] = prediction.metadata["speaker_labels_available"]
            item["raw_label_count"] = len({str(s["speaker"])
                                           for name in ("words", "segments")
                                           for s in raw.get(name, [])
                                           if s.get("speaker") is not None})
        diagnostics.append(item)
    assert len(models) == 20 and len(diagnostics) == 10
    return {"models": models, "diagnostics": diagnostics,
            "reported_diagnostic_cost_usd": sum(r["reported_cost_usd"] or 0 for r in diagnostics),
            "unknown_billing_requests": sum(r["reported_cost_usd"] is None for r in diagnostics)}


if __name__ == "__main__":
    result = audit()
    (REVIEW / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Audited all 20 model outputs and 10 wire diagnostics; request serialization matches.")
