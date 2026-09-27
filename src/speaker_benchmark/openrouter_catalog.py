"""Explicit billing units for the reviewed STT catalog; unknown models stay unpriced."""

import json
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal

CATALOG_URL = "https://openrouter.ai/api/v1/models?output_modalities=transcription"
COLLECTION_URL = "https://openrouter.ai/collections/speech-to-text-models"
SECOND_PRICED = {
    "fish-audio/transcribe-1-pro",
    "assemblyai/universal-3-5-pro",
    "meta/muse-voice-transcribe-1.0",
    "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
    "mistralai/voxtral-small-24b-2507-stt",
    "mistralai/voxtral-mini-3b-2507",
    "qwen/qwen3-asr-1.7b",
    "qwen/qwen3-asr-0.6b",
    "openai/gpt-transcribe",
    "fish-audio/transcribe-1",
    "x-ai/grok-stt-1.0",
    "deepgram/nova-3",
    "nvidia/parakeet-tdt-0.6b-v3",
    "mistralai/voxtral-mini-transcribe",
    "qwen/qwen3-asr-flash-2026-02-10",
    "google/chirp-3",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
    "openai/whisper-1",
}
HOUR_PRICED = {"microsoft/mai-transcribe-2", "microsoft/mai-transcribe-1.5"}
TOKEN_PRICED = {
    "openai/gpt-4o-transcribe",
    "openai/gpt-4o-mini-transcribe",
    "google/gemini-3.5-transcribe",
}


def fetch_catalog():
    with urllib.request.urlopen(CATALOG_URL, timeout=30) as response:
        data = json.load(response)
    return {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": CATALOG_URL,
        "billing_units_source": COLLECTION_URL,
        "models": [
            {
                "id": m["id"],
                "pricing": m["pricing"],
                "billing_unit": (
                    "second"
                    if m["id"] in SECOND_PRICED
                    else "hour"
                    if m["id"] in HOUR_PRICED
                    else "token"
                    if m["id"] in TOKEN_PRICED
                    else "unreviewed"
                ),
            }
            for m in data["data"]
            if "transcription" in m["architecture"]["output_modalities"]
        ],
    }


def estimate(catalog, seconds):
    seconds = Decimal(str(seconds))
    if not seconds.is_finite() or seconds <= 0:
        raise ValueError("Audio duration must be positive and finite")
    rows = []
    for model in catalog["models"]:
        prompt = Decimal(model["pricing"]["prompt"])
        completion = Decimal(model["pricing"]["completion"])
        if any(not p.is_finite() or p < 0 for p in (prompt, completion)):
            raise ValueError("Invalid catalog price")
        unit = model["billing_unit"]
        cost, basis = None, "Unreviewed billing unit: excluded from total"
        if unit in {"second", "hour"} and completion == 0:
            cost = seconds * prompt / (3600 if unit == "hour" else 1)
            basis = f"Catalog price per {unit}; excludes rounding and optional feature charges"
        elif model["id"] == "google/gemini-3.5-transcribe":
            cost = seconds * 25 * prompt + seconds / 60 * 175 * completion
            basis = "Google estimate: 25 audio tokens/second and 175 output tokens/minute"
        elif model["id"] in {"openai/gpt-4o-transcribe", "openai/gpt-4o-mini-transcribe"}:
            # OpenAI publishes approximate blended rates, not guaranteed duration tariffs.
            expected_prompt = Decimal("0.0000025")
            expected_completion = Decimal("0.00001")
            minute = Decimal("0.006")
            if "mini" in model["id"]:
                expected_prompt /= 2
                expected_completion /= 2
                minute /= 2
            if prompt == expected_prompt and completion == expected_completion:
                cost = seconds / 60 * minute
                basis = "OpenAI published approximate blended cost/minute; actual tokens vary"
            else:
                basis = "Token prices changed; refresh the blended rate before spending"
        rows.append(
            {
                "model": model["id"],
                "billing_unit": unit,
                "basis": basis,
                "estimated_usd": float(cost) if cost is not None else None,
            }
        )
    known = [r for r in rows if r["estimated_usd"] is not None]
    return {
        "audio_seconds_per_model": float(seconds),
        "trials": 1,
        "models": rows,
        "estimated_total_usd": sum(r["estimated_usd"] for r in known),
        "complete": len(known) == len(rows),
        "most_expensive_estimate": max(known, key=lambda r: r["estimated_usd"]) if known else None,
        "not_a_spending_cap": True,
    }


def make_config(catalog, requests=4, seconds=240):
    from speaker_benchmark.adapters.openrouter import DIARIZATION

    return {
        "models": [
            {
                "id": m["id"].replace("/", "--"),
                "adapter": "openrouter",
                "options": {
                    "model": m["id"],
                    "max_requests": requests,
                    "max_audio_seconds_total": seconds,
                    "max_audio_seconds": 60,
                    "request_timeout_seconds": 90,
                    **(
                        {
                            "response_format": "verbose_json",
                            "diarization": True,
                            "timestamp_granularities": ["segment", "word"],
                        }
                        if m["id"] in DIARIZATION
                        else {"response_format": "json"}
                    ),
                },
            }
            for m in catalog["models"]
        ]
    }
