"""Build one comparison table from published scores, without new model requests."""

import argparse
import csv
import json
from pathlib import Path

from speaker_benchmark.runner import aggregate
from speaker_benchmark.schema import Prediction, Segment, read_json, write_json
from speaker_benchmark.scoring import score_record

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results/openrouter-2026-09-27"
LOCAL_BUNDLE = ROOT / "results/ami-4090-94clips-2026-09-27"
INTRO = """# Consolidated benchmark results

Sorted by **word error rate (WER), lower is better**, the general transcription metric
used for ranking by [Open ASR](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
Speaker attribution is additionally measured by cpWER and tcpWER, following
[MeetEval](https://arxiv.org/abs/2307.11394); DER measures speaker activity errors.
🥇🥈🥉 mark the three lowest measured values per column, except coverage (higher).
All systems use the same 94 clips and references: 92.28 minutes, up to 60 seconds
per clip. Speaker metrics use the same 89 labelled clips for every scored system.
— means unavailable, not zero. Incomplete API runs receive no medals.

"""
NOTES = """

RTF = processing seconds / audio seconds; lower is faster. Local inference uses one
RTX 4090 without artificial real-time waits. API timing includes network/provider time.
API $ and API timing cover selected successful responses, excluding failed requests
and recovery delays; local timing includes attempted inference. VRAM is peak allocated
GPU memory. Coverage is reference speech time overlapped by predicted speech, not an
accuracy score. Missing or invalid transcripts count as empty in dataset error rates.
The Clips column records incomplete runs and Chirp's one split-request recovery.
Missing speaker cells describe the tested route, not all upstream model capabilities.

[CSV](consolidated.csv) · [Full metrics](consolidated.json) · [Source selection](sources.json).
All 2,726 clip scores were recomputed from saved predictions and references.
[Local run provenance](../ami-4090-94clips-2026-09-27/sources.json) ·
[Chirp recovery](../openrouter-chirp-2026-09-27/README.md).
WER aggregates error counts over reference words; DER aggregates error durations.
No confidence intervals are claimed from four correlated meetings. The WER reference
orders overlapping words chronologically, so it is not an overlap-invariant measure.
Anonymous speaker labels do not test named-person identification or event annotation.

Rebuild and verify: `uv run python scripts/consolidate_results.py --verify`.
AMI-derived material retains its [data attribution and license](../../DATA_LICENSE.md).
"""
LABELS = {
    "moss": "MOSS 0.9B",
    "vibevoice": "VibeVoice offline",
    "qwen_pyannote": "Qwen3 + pyannote",
    "qwen_nemotron": "Qwen3 + Nemotron",
    "vibevoice_streaming": "VibeVoice Streaming",
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


def build(verify=False):
    records = [json.loads(line) for line in (BUNDLE / "manifest.jsonl").read_text().splitlines()]
    record_map = {r["id"]: r for r in records}
    sources = read_json(BUNDLE / "sources.json")
    rows, groups, checked = [], {}, 0
    for source in sources["models"]:
        model = source["model"]
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
        summary["rtf"] = summary["rtf_completed"]
        if model in {"microsoft--mai-transcribe-2", "deepgram--nova-3"}:
            groups[model] = scores
        rows.append(summary)
    common, speaker_scores = shared_speaker_scores(groups)
    local_records = [json.loads(line) for line in
                     (LOCAL_BUNDLE / "manifest.jsonl").read_text().splitlines()]
    validate_same_records(records, local_records)
    local_checked = 0
    for source in read_json(LOCAL_BUNDLE / "sources.json")["models"]:
        model = source["model"]
        scores = read_json(LOCAL_BUNDLE / "scores" / f"{model}.json")
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
        rows.append(summary)
        groups[model] = scores
    _, speaker_scores = shared_speaker_scores(groups, common)
    if len(rows) != len(LABELS) or {r["model"] for r in rows} != set(LABELS):
        raise ValueError("Consolidation requires all 24 API models and five local systems")
    for row in rows:
        row["full_dataset_metrics"] = {k: row[k] for k in ("wer", "cpwer", "tcpwer", "der")}
        if row["model"] in speaker_scores:
            row.update(speaker_scores[row["model"]], speaker_metric_clips=len(common))
    for row in rows:
        row["label"] = LABELS[row["model"]]
        memory = row.get("max_cuda_allocated_bytes")
        row["vram_gib"] = memory / 2**30 if memory is not None else None
    rows.sort(key=lambda r: r["wer"])
    result = {"rows": rows, "shared_speaker_clip_ids": common, "verified_api_scores": checked,
              "verified_local_scores": local_checked}
    write_json(BUNDLE / "consolidated.json", result)
    columns = [
        ("wer", "WER ↓", ".2%"), ("cpwer", f"cpWER ({len(common)}) ↓", ".2%"),
        ("tcpwer", f"tcpWER ({len(common)}) ↓", ".2%"),
        ("der", f"DER ({len(common)}) ↓", ".2%"),
        ("speech_time_coverage", "Coverage ↑", ".2%"), ("rtf", "RTF ↓", ".3f"),
        ("api_reported_cost_usd", "API $ ↓", ".4f"), ("vram_gib", "VRAM GiB ↓", ".2f"),
    ]
    lines = ["| Model | " + " | ".join(c[1] for c in columns) + " | Clips |",
             "|---|" + "---:|" * (len(columns) + 1)]
    for row in rows:
        cells = [row["label"]]
        for key, _, fmt in columns:
            value = row.get(key)
            values = [r[key] for r in rows if r["eligible"] and r.get(key) is not None]
            icon = (
                medal(value, values, higher=key == "speech_time_coverage")
                if row["eligible"] else ""
            )
            cells.append("—" if value is None else icon + format(value, fmt))
        status = f"{row['completed']}/{row['recordings']}"
        if row["model"] == "google--chirp-3":
            status += "; 1 split"
        if row["invalid_output"]:
            status += f"; {row['invalid_output']} invalid"
        cells.append(status)
        lines.append("| " + " | ".join(cells) + " |")
    fields = ["model", "label", "track", "eligible"] + [c[0] for c in columns] + [
        "completed", "recordings", "speaker_metric_clips",
    ]
    with (BUNDLE / "consolidated.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    (BUNDLE / "table.md").write_text("\n".join(lines) + "\n")
    (BUNDLE / "README.md").write_text(INTRO + "\n".join(lines) + NOTES)
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
