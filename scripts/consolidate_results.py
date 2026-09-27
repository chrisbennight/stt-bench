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
INTRO = """# Consolidated benchmark results

Sorted by **word error rate (WER), lower is better**, the general transcription metric
used for ranking by [Open ASR](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
Speaker attribution is additionally measured by cpWER and tcpWER, following
[MeetEval](https://arxiv.org/abs/2307.11394); DER measures speaker activity errors.
🥇🥈🥉 mark the three lowest measured values per column, except coverage (higher).
These are descriptive ranks across the documented protocol differences, not proof
of a controlled head-to-head win. — means unavailable, not zero.

"""
NOTES = """

* **\\*** Local RTX 4090: 25 windows up to 240 seconds. Hosted models: 94 windows up
  to 60 seconds. Both cover the same 92.28 minutes. Streaming was paced in real time;
  first text averaged 6.20 seconds, with 1.28-second p95 chunk-emission lag.
* **†** Chirp uses two 30-second requests for one recovered clip; all other clips
  retain the standard window. [Recovery details](../openrouter-chirp-2026-09-27/README.md).
* **‡** MAI 2 and Deepgram speaker metrics use the same 89 clips with labels from
  both models. WER and coverage use all 94. Other API routes were not tested with
  verified diarization options; missing cells do not establish model incapability.
* **§ / ¶** VibeVoice's one invalid output and DeepInfra's missing clips count as
  empty transcripts in full-dataset error rates. DeepInfra costs and RTF cover only
  completed clips and receive no medals. Their WER reflects service failures too.

RTF = processing seconds / audio seconds; lower is faster. API timing includes network
and provider time and excludes failed requests and recovery delays; local timing
includes attempted inference. API $ covers selected successful responses, not the
entire session bill or local electricity. VRAM is peak allocated GPU memory, not
total required capacity. Coverage measures overlap with reference speech time and
can be inflated by false speech; its medals indicate coverage only.

[CSV](consolidated.csv) · [Full metrics](consolidated.json) · [Source selection](sources.json).
All 2,256 API clip scores were recomputed from the bundled predictions and references.
The local scores retain their
[published provenance](../ami-4090-2026-09-27/comparison/comparison.json).
WER aggregates error counts over reference words; DER aggregates error durations.
No confidence intervals are claimed from four correlated meetings. The WER reference
orders overlapping words chronologically, so it is not an overlap-invariant measure.
Anonymous speaker labels do not test named-person identification or event annotation.

Rebuild and verify: `uv run python scripts/consolidate_results.py --verify`.
AMI-derived material retains its [data attribution and license](../../DATA_LICENSE.md).
"""
LABELS = {
    "moss": "MOSS 0.9B*",
    "vibevoice": "VibeVoice offline*§",
    "qwen_pyannote": "Qwen3 + pyannote*",
    "qwen_nemotron": "Qwen3 + Nemotron*",
    "vibevoice_streaming": "VibeVoice Streaming*",
    "google--gemini-3.5-transcribe": "Gemini 3.5 Transcribe",
    "google--chirp-3": "Google Chirp 3†",
    "microsoft--mai-transcribe-1.5": "MAI Transcribe 1.5",
    "microsoft--mai-transcribe-2": "MAI Transcribe 2‡",
    "deepgram--nova-3": "Deepgram Nova 3‡",
    "assemblyai--universal-3-5-pro": "AssemblyAI Universal 3.5 Pro",
    "nvidia--nemotron-3.5-asr-streaming-multilingual-0.6b": "NVIDIA Nemotron 3.5 ASR 0.6B",
    "nvidia--parakeet-tdt-0.6b-v3": "NVIDIA Parakeet TDT v3",
    "fish-audio--transcribe-1": "Fish Transcribe 1",
    "fish-audio--transcribe-1-pro": "Fish Transcribe 1 Pro",
    "meta--muse-voice-transcribe-1.0": "Meta Muse Voice 1.0",
    "mistralai--voxtral-mini-transcribe": "Voxtral Mini Transcribe",
    "mistralai--voxtral-small-24b-2507-stt": "Voxtral Small 24B¶",
    "mistralai--voxtral-mini-3b-2507": "Voxtral Mini 3B¶",
    "qwen--qwen3-asr-1.7b": "Qwen3 ASR 1.7B¶",
    "qwen--qwen3-asr-0.6b": "Qwen3 ASR 0.6B¶",
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


def shared_speaker_scores(groups):
    valid = [
        {r["id"] for r in rows if all(r["scores"][k] is not None
                                      for k in ("cpwer", "tcpwer", "der"))}
        for rows in groups.values()
    ]
    common = set.intersection(*valid)
    result = {}
    for model, rows in groups.items():
        scores = [r["scores"] for r in rows if r["id"] in common]
        metrics = {}
        for key in ("cpwer", "tcpwer"):
            denominator = sum(s[key]["length"] for s in scores)
            metrics[key] = sum(s[key]["errors"] for s in scores) / denominator
        denominator = sum(s["der"]["total"] for s in scores)
        metrics["der"] = sum(
            sum(s["der"][k] for k in ("false alarm", "missed detection", "confusion"))
            for s in scores
        ) / denominator
        result[model] = metrics
    return sorted(common), result


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
    for row in rows:
        if row["model"] in speaker_scores:
            row.update(speaker_scores[row["model"]], speaker_metric_clips=len(common))
    local = read_json(ROOT / "results/ami-4090-2026-09-27/comparison/comparison.json")
    for row in local["summary"]:
        row.update(track="local", eligible=True, rtf=row["rtf_attempted"])
        rows.append(row)
    for row in rows:
        row["label"] = LABELS[row["model"]]
        memory = row.get("max_cuda_allocated_bytes")
        row["vram_gib"] = memory / 2**30 if memory is not None else None
    rows.sort(key=lambda r: r["wer"])
    result = {"rows": rows, "shared_speaker_clip_ids": common, "verified_api_scores": checked}
    write_json(BUNDLE / "consolidated.json", result)
    columns = [
        ("wer", "WER ↓", ".2%"), ("cpwer", "cpWER ↓", ".2%"),
        ("tcpwer", "tcpWER ↓", ".2%"), ("der", "DER ↓", ".2%"),
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
        cells.append(f"{row['completed']}/{row['recordings']}")
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
    print(f"Built {len(result['rows'])} rows; verified {result['verified_api_scores']} API scores")
