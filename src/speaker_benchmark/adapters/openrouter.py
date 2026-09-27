"""OpenRouter's dedicated transcription API; one request per recording, no retries."""

import base64
import io
import json
import math
import os
import re
import urllib.error
import urllib.request

from speaker_benchmark.adapters.common import Adapter, InvalidModelOutput
from speaker_benchmark.schema import Prediction, Segment

ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
OPTIONS = {
    "model",
    "response_format",
    "timestamp_granularities",
    "diarization",
    "request_timeout_seconds",
    "max_audio_seconds",
    "max_requests",
    "max_audio_seconds_total",
}
# These option shapes are documented by OpenRouter. Other models may support
# diarization upstream without exposing a verified option through this API.
DIARIZATION = {
    "microsoft/mai-transcribe-2": {"azure": {"diarization": {"enabled": True}}},
    "deepgram/nova-3": {"deepgram": {"diarize": True}},
}


def validate_options(options):
    if set(options) - OPTIONS:
        raise ValueError("Unknown OpenRouter option; credentials belong in the environment")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.:-]+", options.get("model", "")):
        raise ValueError("OpenRouter model must be an organization/model ID")
    for key in ("max_requests", "max_audio_seconds_total"):
        if key not in options:
            raise ValueError(f"OpenRouter requires an explicit {key} limit")
    for key in (
        "max_requests",
        "max_audio_seconds_total",
        "max_audio_seconds",
        "request_timeout_seconds",
    ):
        if key in options:
            value = options[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("OpenRouter limits must be positive finite numbers")
    if type(options["max_requests"]) is not int:
        raise ValueError("max_requests must be an integer")
    if options.get("response_format", "json") not in {"json", "verbose_json"}:
        raise ValueError("Unsupported response format")
    granularities = options.get("timestamp_granularities", [])
    if not isinstance(granularities, list) or any(
        g not in ("word", "segment") for g in granularities
    ):
        raise ValueError("Invalid timestamp granularities")
    if granularities and options.get("response_format") != "verbose_json":
        raise ValueError("Timestamps require verbose_json")
    if type(options.get("diarization", False)) is not bool:
        raise ValueError("diarization must be a boolean")
    if options.get("diarization") and (
        options["model"] not in DIARIZATION or options.get("response_format") != "verbose_json"
    ):
        raise ValueError("Diarization requires a documented model option and verbose_json")


class RemoteRequestError(RuntimeError):
    """A request failed; do not expose response bodies or retry a possibly billed request."""

    def __init__(self, http_status=None):
        self.http_status = http_status
        super().__init__("OpenRouter request failed; check the provider request history")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # An API redirect must never forward the Authorization header to another origin.
        return None


def parse_response(data, duration):
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("Missing transcription text")
    words, phrases = data.get("words") or [], data.get("segments") or []
    if not isinstance(words, list) or not isinstance(phrases, list):
        raise ValueError("Invalid structured transcript")
    if any(not isinstance(s, dict) for s in words + phrases):
        raise ValueError("Invalid transcript segment")
    # Prefer words only when they retain the speaker information available on phrases.
    rows, text_key = phrases, "text"
    if words and (
        not any(s.get("speaker") is not None for s in phrases)
        or all(s.get("speaker") is not None for s in words)
    ):
        rows, text_key = words, "word"
    segments = []
    speakers_available = bool(rows) and all(s.get("speaker") is not None for s in rows)
    for row in rows:
        text = row.get(text_key)
        if not isinstance(text, str):
            raise ValueError("Invalid transcript text")
        start, end = row.get("start"), row.get("end")
        if start is not None and (type(start) not in (int, float) or type(end) not in (int, float)):
            raise ValueError("Invalid transcript timestamps")
        segment = Segment(str(row.get("speaker", "unassigned")), text, start, end)
        if segment.end is not None and segment.end > duration + 0.05:
            raise ValueError("Transcript extends past the audio")
        segments.append(segment)
    if not segments and data["text"].strip():
        segments = [Segment("unassigned", data["text"])]
    timed = bool(segments) and all(s.start is not None for s in segments)
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("Invalid usage object")
    safe_usage = {}
    for key in ("seconds", "total_tokens", "input_tokens", "output_tokens", "cost"):
        if key in usage:
            value = usage[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid usage number")
            safe_usage[key] = value
    return Prediction(
        segments,
        timing="native" if timed else "unavailable",
        metadata={
            "speaker_labels_available": speakers_available,
            "usage": safe_usage,
            "raw_transcript": {k: data[k] for k in ("text", "segments", "words") if k in data},
            "execution": "remote_api",
        },
    )


class OpenRouter(Adapter):
    def __init__(self, options):
        super().__init__(options)
        validate_options(options)
        self.max_duration = options.get("max_audio_seconds", 60)
        self.requests = 0
        self.audio_seconds = 0.0

    def load(self):
        if os.environ.get("OPENROUTER_ALLOW_PAID_REQUESTS") != "1":
            raise ValueError("Set OPENROUTER_ALLOW_PAID_REQUESTS=1 after reviewing the estimate")
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is required")
        self.opener = urllib.request.build_opener(NoRedirect())

    def transcribe(self, audio_path, duration, language):
        import soundfile as sf

        if not math.isfinite(duration) or duration <= 0 or duration > self.max_duration:
            raise ValueError("Audio exceeds the configured per-request limit")
        if (
            self.requests >= self.options["max_requests"]
            or self.audio_seconds + duration > self.options["max_audio_seconds_total"] + 1e-6
        ):
            raise ValueError("OpenRouter request or audio allowance exhausted")
        # Normalize the transport format, not the signal: no denoising or resampling.
        info = sf.info(audio_path)
        if info.channels != 1 or info.samplerate != 16000:
            raise ValueError("Prepare mono 16 kHz audio before remote inference")
        if abs(info.duration - duration) > 1 / info.samplerate:
            raise ValueError("Audio duration differs from the approved recording")
        audio, rate = sf.read(audio_path, dtype="int16")
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV", subtype="PCM_16")
        payload = {
            "model": self.options["model"],
            "input_audio": {"data": base64.b64encode(buffer.getvalue()).decode(), "format": "wav"},
            "response_format": self.options.get("response_format", "json"),
        }
        if language:
            if language.lower() == "english":
                language = "en"
            if not re.fullmatch(r"[a-z]{2}", language):
                raise ValueError("OpenRouter requires an ISO-639-1 language code")
            payload["language"] = language
        if self.options.get("timestamp_granularities"):
            payload["timestamp_granularities"] = self.options["timestamp_granularities"]
        if self.options.get("diarization"):
            payload["provider"] = {"options": DIARIZATION[self.options["model"]]}
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
            },
        )
        # Reserve before sending. A timeout can still be billed; never repeat it automatically.
        self.requests += 1
        self.audio_seconds += duration
        try:
            with self.opener.open(
                request, timeout=self.options.get("request_timeout_seconds", 90)
            ) as response:
                body = response.read(8 * 1024 * 1024 + 1)
                generation_id = response.headers.get("X-Generation-Id")
        except urllib.error.HTTPError as exc:
            raise RemoteRequestError(exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RemoteRequestError() from None
        metadata = {"execution": "remote_api", "model": self.options["model"]}
        if generation_id and re.fullmatch(r"[A-Za-z0-9_-]{1,200}", generation_id):
            metadata["generation_id"] = generation_id
        if len(body) > 8 * 1024 * 1024:
            raise InvalidModelOutput("", {**metadata, "reason": "response_too_large"})
        try:
            data = json.loads(body)
            prediction = parse_response(data, duration)
        except (ValueError, TypeError, KeyError):
            # Do not persist arbitrary server error bodies that might echo request headers.
            raise InvalidModelOutput("", {**metadata, "reason": "invalid_response"}) from None
        prediction.metadata.update(metadata)
        return prediction
