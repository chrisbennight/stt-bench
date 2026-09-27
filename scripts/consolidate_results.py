"""Build one comparison table from published scores, without new model requests."""

import argparse
import csv
import json
import math
from pathlib import Path

from speaker_benchmark.runner import aggregate
from speaker_benchmark.schema import Prediction, Segment, read_json, write_json
from speaker_benchmark.scoring import score_record

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/openrouter-validated-2026-09-27"
LOCAL_BUNDLE = ROOT / "results/ami-4090-94clips-2026-09-27"
OPENWEIGHTS_BUNDLE = ROOT / "results/local-openweights-2026-09-27"
INTRO = """# Consolidated benchmark results

**One comparison, the same audio for every system.**

- **Benchmark data — AMI Meeting Corpus:** [Paper](https://doi.org/10.1007/11677482_3) ·
  [Dataset](https://groups.inf.ed.ac.uk/ami/corpus/) ·
  [Diarization references on GitHub](https://github.com/pyannote/AMI-diarization-setup).
- **Transcription scoring — MeetEval:** [Paper](https://arxiv.org/abs/2307.11394) ·
  [GitHub](https://github.com/fgnt/meeteval). DER uses
  [pyannote.metrics](https://github.com/pyannote/pyannote-metrics).

- **Test audio:** 94 clips, 92.28 minutes total, up to 60 seconds per clip.
- **Speaker comparison:** the same {speaker_clips} labelled clips for every scored system.
- **Sort order:** WER, lowest first, following the general transcription ranking used by Open ASR.
- **Medals:** 🥇🥈🥉 mark the three lowest measured values per column.
- **Missing results:** cells identify missing labels or timing in the tested output.

"""
METRICS = """**What the metrics mean**

- **WER — word error rate ↓:** substituted, deleted, and inserted words divided by
  reference words, ignoring speaker identity. Lower WER generally means fewer words
  you need to correct before using the transcript.
- **cpWER — concatenated minimum-permutation WER ↓:** measures transcription and speaker
  attribution errors, allowing anonymous speaker labels to be renamed for the best match.
  It helps assess whether a meeting transcript accurately records who said what.
- **tcpWER — time-constrained cpWER ↓:** adds timing constraints to speaker-attributed
  word matching, using a five-second collar here. It helps assess whether the right
  person's words appear at the right point in the recording for playback and review.
- **DER — diarization error rate ↓:** missed speech, false alarms, and speaker confusion
  divided by reference speaker time, with zero collar and overlapping speech included.
  It measures how reliably the system marks when each person is speaking, regardless
  of whether it transcribes their words correctly.
- **RTF — real-time factor ↓:** processing seconds divided by audio seconds; 0.1 means
  processing took one tenth of the audio duration. It helps estimate how long you will
  wait for a recording to finish processing, rather than the delay of live captions.
- **API $ ↓:** reported response charges in US dollars, including parser-rejected responses;
  failed requests have unknown billing unless reconciled in the cost ledger.
  It helps estimate the service bill for processing a similar amount of audio.
- **VRAM GiB ↓:** peak allocated GPU memory during local inference, not total device memory.
  It helps assess whether a local pipeline will fit on your GPU, allowing extra room
  for other allocations and applications.
- **Clips:** successful outputs / total clips; missing or invalid transcripts count as
  empty hypotheses in dataset error rates. It shows how much of the workload returned
  usable output and how many clips still need attention, rather than whether the
  returned words were accurate.

"""
NOTES = """

**Reading the results fairly**

- **Timing:** local inference uses one RTX 4090 without artificial real-time waits.
  API timing includes network/provider time and covers completed responses;
  local timing includes attempted inference.
- **Aggregation:** WER sums word-error counts before division; DER sums error durations.
  Overlapping reference words are ordered chronologically for WER, which makes it
  sensitive to word ordering during overlap.
- **Output limitations:** no external diarizer or invented timestamps are added to hosted
  outputs. Gemini and Voxtral were asked for diarization but returned no speaker fields
  in the probe. This does not establish the limits of every upstream route.
- **Local counterparts:** the additional local ASR pipelines use pyannote Community-1
  and Qwen forced alignment. Voxtral Small uses NF4 quantization, as its name indicates.
  Their diarization scores measure the shared pipeline, not native model diarization.
  See the [local pipeline protocol](../../docs/LOCAL_OPENWEIGHTS.md).
- **Scope:** four correlated meetings do not support claimed confidence intervals.
  Anonymous speaker labels do not test named-person identification or event annotation.

**Download and verify**

All **{verified_scores} clip scores** were recomputed from saved predictions and references.

- **Results:** [CSV](consolidated.csv) · [Full metrics](consolidated.json)
- **Provenance:** [Source selection](sources.json) ·
  [Original local run](../ami-4090-94clips-2026-09-27/sources.json) ·
  [Local counterparts](../local-openweights-2026-09-27/sources.json)
- **Audit:** [Request costs](costs.json) ·
  [Hosted capabilities](../../docs/OPENROUTER_CAPABILITY_AUDIT.md)

Rebuild and verify locally, without new model requests:

```bash
uv run python scripts/consolidate_results.py --verify
```

**References**

- [Open ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard): WER ranking.
- [MeetEval](https://arxiv.org/abs/2307.11394): speaker-attributed transcription metrics.
- [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/): diarization scoring.
- [Evaluation protocol](../../docs/PROTOCOL.md): normalization, timing, and subset rules.
- [AMI attribution and license](../../DATA_LICENSE.md): terms for the derived data.
"""
LABELS = {
    "moss": "MOSS 0.9B",
    "vibevoice": "VibeVoice offline",
    "qwen_pyannote": "Qwen3 + pyannote",
    "qwen_nemotron": "Qwen3 + Nemotron",
    "vibevoice_streaming": "VibeVoice Streaming",
    "parakeet_pyannote": "Parakeet TDT v3 + pyannote",
    "nemotron_asr_pyannote": "Nemotron 3.5 ASR + pyannote",
    "whisper_pyannote": "Whisper Large v3 + pyannote",
    "whisper_turbo_pyannote": "Whisper Turbo + pyannote",
    "qwen_small_pyannote": "Qwen3 ASR 0.6B + pyannote",
    "voxtral_mini_pyannote": "Voxtral Mini 3B + pyannote",
    "voxtral_small_nf4_pyannote": "Voxtral Small 24B NF4 + pyannote",
    "google--gemini-3.5-transcribe": "Gemini 3.5 Transcribe",
    "google--chirp-3": "Google Chirp 3",
    "microsoft--mai-transcribe-1.5": "MAI Transcribe 1.5",
    "microsoft--mai-transcribe-2": "MAI Transcribe 2",
    "deepgram--nova-3": "Deepgram Nova 3",
    "assemblyai--universal-3-5-pro": "AssemblyAI Universal 3.5 Pro",
    "nvidia--nemotron-3.5-asr-streaming-multilingual-0.6b": "NVIDIA Nemotron 3.5 ASR 0.6B",
    "nvidia--parakeet-tdt-0.6b-v3": "NVIDIA Parakeet TDT v3",
    "fish-audio--transcribe-1": "Fish Transcribe 1",
    "fish-audio--transcribe-1-pro": "Fish Transcribe 1 Pro",
    "meta--muse-voice-transcribe-1.0": "Meta Muse Voice 1.0",
    "mistralai--voxtral-mini-transcribe": "Voxtral Mini Transcribe",
    "mistralai--voxtral-small-24b-2507-stt": "Voxtral Small 24B",
    "mistralai--voxtral-mini-3b-2507": "Voxtral Mini 3B",
    "qwen--qwen3-asr-1.7b": "Qwen3 ASR 1.7B",
    "qwen--qwen3-asr-0.6b": "Qwen3 ASR 0.6B",
    "qwen--qwen3-asr-flash-2026-02-10": "Qwen3 ASR Flash",
    "openai--gpt-transcribe": "OpenAI GPT Transcribe",
    "openai--gpt-4o-transcribe": "GPT-4o Transcribe",
    "openai--gpt-4o-mini-transcribe": "GPT-4o Mini Transcribe",
    "openai--whisper-1": "Whisper 1",
    "openai--whisper-large-v3": "Whisper Large v3",
    "openai--whisper-large-v3-turbo": "Whisper Large v3 Turbo",
    "x-ai--grok-stt-1.0": "Grok STT 1.0",
}


LOCAL_MODEL_CARDS = {
    "moss": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "vibevoice": "microsoft/VibeVoice-ASR",
    "vibevoice_streaming": "microsoft/VibeVoice-ASR-Streaming-7B",
    "qwen_pyannote": "Qwen/Qwen3-ASR-1.7B",
    "qwen_nemotron": "Qwen/Qwen3-ASR-1.7B",
    "qwen_small_pyannote": "Qwen/Qwen3-ASR-0.6B",
    "parakeet_pyannote": "nvidia/parakeet-tdt-0.6b-v3",
    "nemotron_asr_pyannote": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "whisper_pyannote": "openai/whisper-large-v3",
    "whisper_turbo_pyannote": "openai/whisper-large-v3-turbo",
    "voxtral_mini_pyannote": "mistralai/Voxtral-Mini-3B-2507",
    "voxtral_small_nf4_pyannote": "mistralai/Voxtral-Small-24B-2507",
}


def model_card_url(model, track):
    if track == "api":
        return "https://openrouter.ai/" + model.replace("--", "/", 1)
    return "https://huggingface.co/" + LOCAL_MODEL_CARDS[model]


def medal(value, values, higher=False):
    """Competition ranks preserve ties; incomplete API runs never enter values."""
    if value is None or len(values) < 2:
        return ""
    rank = 1 + sum(v > value if higher else v < value for v in values)
    return {1: "🥇 ", 2: "🥈 ", 3: "🥉 "}.get(rank, "")


def shared_speaker_scores(groups, clip_ids=None):
    valid = [
        {r["id"] for r in rows if all(r["scores"][k] is not None
                                      for k in ("cpwer", "tcpwer", "der"))}
        for rows in groups.values()
    ]
    common = set.intersection(*valid) if clip_ids is None else set(clip_ids)
    if not common:
        raise ValueError("No common speaker-metric clips")
    result = {}
    for model, rows in groups.items():
        scores = [r["scores"] for r in rows if r["id"] in common]
        if {r["id"] for r in rows if r["id"] in common} != common:
            raise ValueError("Missing a shared speaker-metric clip")
        metrics = {}
        for key in ("cpwer", "tcpwer"):
            if any(s[key] is None for s in scores):
                metrics[key] = None
                continue
            denominator = sum(s[key]["length"] for s in scores)
            metrics[key] = sum(s[key]["errors"] for s in scores) / denominator
        metrics["der"] = None
        if all(s["der"] is not None for s in scores):
            denominator = sum(s["der"]["total"] for s in scores)
            metrics["der"] = sum(
                sum(s["der"][k] for k in ("false alarm", "missed detection", "confusion"))
                for s in scores
            ) / denominator
        result[model] = metrics
    return sorted(common), result


def validate_same_records(expected, actual):
    keys = ("id", "duration", "audio_sha256", "reference", "reference_activity")
    if len(actual) != len(expected) or any(
        any(a[k] != b[k] for k in keys) for a, b in zip(expected, actual, strict=True)
    ):
        raise ValueError("Local and API audio windows and references must match exactly")


def missing_cell(row, key):
    if row.get("excluded"):
        return "Excluded"
    if key == "cpwer":
        return "No labels"
    if key in {"tcpwer", "der"}:
        return "No speaker times" if row.get("cpwer") is not None else "No labels"
    return "—"


def reported_cost(row):
    metadata = row.get("prediction", {}).get("metadata", {})
    usage = (metadata.get("usage", {}) if row["status"] == "ok"
             else row.get("metadata", {}).get("response_usage", {}))
    cost = usage.get("cost")
    return cost if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 else None


def build(verify=False):
    records = [json.loads(line) for line in (BUNDLE / "manifest.jsonl").read_text().splitlines()]
    record_map = {r["id"]: r for r in records}
    sources = read_json(BUNDLE / "sources.json")
    rows, groups, checked = [], {}, 0
    for source in sources["models"]:
        model = source["model"]
        if source.get("excluded"):
            continue
        scores = read_json(BUNDLE / "scores" / f"{model}.json")
        if len(scores) != len(records) or {r["id"] for r in scores} != set(record_map):
            raise ValueError("Scores must cover each reference exactly once")
        if verify:
            for row in scores:
                record = dict(record_map[row["id"]])
                for key in ("reference", "reference_activity"):
                    record[key] = [Segment(**s) for s in record[key]]
                prediction = (
                    Prediction.from_dict(row["prediction"]) if row["status"] == "ok"
                    else Prediction([], timing="failed",
                                    metadata={"speaker_labels_available": False})
                )
                actual = score_record(
                    record, prediction, sources["tcp_collar"], sources["der_collar"]
                )
                if actual != row["scores"]:
                    raise ValueError(f"Saved score mismatch: {model}/{row['id']}")
                checked += 1
        summary = aggregate(model, scores, records)
        summary.update(track="api", eligible=summary["completed"] == len(records))
        summary.pop("speech_time_coverage", None)
        summary["rtf"] = summary["rtf_completed"]
        costs = [reported_cost(r) for r in scores]
        summary["api_reported_cost_usd"] = sum(c for c in costs if c is not None)
        summary["api_cost_complete"] = all(c is not None for c in costs)
        if any(r["scores"]["cpwer"] is not None for r in scores):
            groups[model] = scores
        rows.append(summary)
    common_ids = set.intersection(*(
        {r["id"] for r in scores if r["scores"]["cpwer"] is not None}
        for scores in groups.values()
    ))
    common, speaker_scores = shared_speaker_scores(groups, common_ids)
    local_sources = []
    local_bundles = [LOCAL_BUNDLE]
    if OPENWEIGHTS_BUNDLE.exists():
        local_bundles.append(OPENWEIGHTS_BUNDLE)
    for local_bundle in local_bundles:
        local_records = [json.loads(line) for line in
                         (local_bundle / "manifest.jsonl").read_text().splitlines()]
        validate_same_records(records, local_records)
        local_sources.extend(
            (local_bundle, source)
            for source in read_json(local_bundle / "sources.json")["models"]
        )
    local_checked = 0
    for local_bundle, source in local_sources:
        model = source["model"]
        if model in {r["model"] for r in rows}:
            raise ValueError("A model must appear only once in the consolidated comparison")
        scores = read_json(local_bundle / "scores" / f"{model}.json")
        if len(scores) != len(records) or {r["id"] for r in scores} != set(record_map):
            raise ValueError("Local scores must cover each reference exactly once")
        if verify:
            for row in scores:
                record = dict(record_map[row["id"]])
                for key in ("reference", "reference_activity"):
                    record[key] = [Segment(**s) for s in record[key]]
                prediction = (Prediction.from_dict(row["prediction"]) if row["status"] == "ok"
                              else Prediction([], timing="failed"))
                actual = score_record(
                    record, prediction, sources["tcp_collar"], sources["der_collar"]
                )
                if actual != row["scores"]:
                    raise ValueError(f"Saved score mismatch: {model}/{row['id']}")
                local_checked += 1
        summary = aggregate(model, scores, records)
        summary.update(track="local", eligible=all(
            r["status"] in {"ok", "invalid_output"} for r in scores
        ), rtf=summary["rtf_attempted"])
        summary.pop("speech_time_coverage", None)
        rows.append(summary)
        groups[model] = scores
    _, speaker_scores = shared_speaker_scores(groups, common)
    expected = {s["model"] for s in sources["models"] if not s.get("excluded")} | {
        s["model"] for _, s in local_sources
    }
    if len(rows) != len(expected) or {r["model"] for r in rows} != expected:
        raise ValueError("Consolidation must include every non-excluded source exactly once")
    for row in rows:
        row["full_dataset_metrics"] = {k: row[k] for k in ("wer", "cpwer", "tcpwer", "der")}
        if row["model"] in speaker_scores:
            row.update(speaker_scores[row["model"]], speaker_metric_clips=len(common))
    for row in rows:
        row["label"] = LABELS[row["model"]]
        row["model_url"] = model_card_url(row["model"], row["track"])
        row["run_on"] = {"api": "OpenRouter", "local": "Local 4090"}[row["track"]]
        memory = row.get("max_cuda_allocated_bytes")
        row["vram_gib"] = memory / 2**30 if memory is not None else None
    rows.sort(key=lambda r: r["wer"] if r["wer"] is not None else float("inf"))
    result = {"rows": rows, "shared_speaker_clip_ids": common, "verified_api_scores": checked,
              "verified_local_scores": local_checked}
    write_json(BUNDLE / "consolidated.json", result)
    columns = [
        ("wer", "WER ↓", ".2%"), ("cpwer", f"cpWER ({len(common)}) ↓", ".2%"),
        ("tcpwer", f"tcpWER ({len(common)}) ↓", ".2%"),
        ("der", f"DER ({len(common)}) ↓", ".2%"),
        ("rtf", "RTF ↓", ".3f"),
        ("api_reported_cost_usd", "API $ ↓", ".4f"), ("vram_gib", "VRAM GiB ↓", ".2f"),
    ]
    lines = ["| Model | Run on | " + " | ".join(c[1] for c in columns) + " | Clips |",
             "|---|---|" + "---:|" * (len(columns) + 1)]
    for row in rows:
        cells = [f"[{row['label']}]({row['model_url']})", row["run_on"]]
        for key, _, fmt in columns:
            value = row.get(key)
            values = [r[key] for r in rows if r["eligible"] and r.get(key) is not None]
            icon = (
                medal(value, values)
                if row["eligible"] else ""
            )
            cells.append(missing_cell(row, key) if value is None else icon + format(value, fmt))
        status = "Excluded" if row.get("excluded") else f"{row['completed']}/{row['recordings']}"
        if row["invalid_output"]:
            status += f"; {row['invalid_output']} invalid"
        cells.append(status)
        lines.append("| " + " | ".join(cells) + " |")
    fields = [
        "model", "label", "model_url", "run_on", "track", "eligible",
    ] + [c[0] for c in columns] + [
        "completed", "recordings", "speaker_metric_clips",
    ]
    with (BUNDLE / "consolidated.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    (BUNDLE / "table.md").write_text("\n".join(lines) + "\n")
    (BUNDLE / "README.md").write_text(
        INTRO.format(speaker_clips=len(common)) + METRICS + "\n".join(lines)
        + NOTES.format(verified_scores=f"{checked + local_checked:,}")
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = build(args.verify)
    print(
        f"Built {len(result['rows'])} rows; verified {result['verified_api_scores']} API "
        f"and {result['verified_local_scores']} local scores"
    )
