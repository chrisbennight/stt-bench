"""Reparse saved streaming text into a separate, auditable result directory."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from speaker_benchmark.adapters import joint
from speaker_benchmark.runner import aggregate
from speaker_benchmark.schema import Prediction, Segment, digest, safe_id, write_json
from speaker_benchmark.scoring import normalize, score_record


def speaker_text(segments):
    grouped = defaultdict(list)
    for segment in segments:
        grouped[segment["speaker"]].append(segment["text"])
    return {speaker: normalize(" ".join(text)) for speaker, text in grouped.items()}


def reparse(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or source in destination.parents:
        raise ValueError("Destination must be new and outside the original run")
    plan = json.loads((source / "plan.json").read_text())
    records = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines()]
    identifiers = {safe_id(r["id"]) for r in records}
    if len(identifiers) != len(records):
        raise ValueError("Duplicate reference IDs")
    updates, audit = {}, []
    for model in plan["models"]:
        identifier = safe_id(model["id"])
        directory = source / identifier
        worker = json.loads((directory / "worker.json").read_text())
        paths = list((directory / "predictions").glob("*.json"))
        if worker["exit_code"] != 0 or {p.stem for p in paths} != identifiers:
            raise ValueError("Every model must have finished every recording")
        for path in paths:
            row = json.loads(path.read_text())
            if row["id"] != path.stem or row["status"] not in {"ok", "invalid_output"}:
                raise ValueError("Invalid or unattempted recording")
            if model["adapter"] != "vibevoice_streaming" or row["status"] != "ok":
                continue
            prediction = row["prediction"]
            events = prediction["events"]
            if "".join(e["text"] for e in events) != prediction["metadata"]["raw_output"]:
                raise ValueError("Streaming events do not match the saved raw text")
            segments, corrected = joint.parse_stream_events(events)
            parsed = [asdict(s) for s in segments]
            audit.append(
                {
                    "model": identifier,
                    "recording": row["id"],
                    "original_prediction_sha256": digest(path),
                    "speaker_text_changed": speaker_text(parsed)
                    != speaker_text(prediction["segments"]),
                    "speech_event_flags_changed": sum(
                        a["has_text"] != b["has_text"] for a, b in zip(events, corrected)
                    ),
                }
            )
            prediction["segments"] = parsed
            prediction["events"] = corrected
            prediction["metadata"]["stream_output_parser"] = "joined-chunks-v1"
            updates[path.relative_to(source)] = row
    if not audit:
        raise ValueError("No completed streaming predictions found")
    shutil.copytree(source, destination)
    for relative, row in updates.items():
        write_json(destination / relative, row)
    write_json(
        destination / "reprocessing.json",
        {
            "source_run": str(source),
            "source_plan_sha256": digest(source / "plan.json"),
            "parser_source_sha256": digest(joint.__file__),
            "parser": "joined-chunks-v1",
            "reason": "Sound annotations, speaker labels, and words can span streaming chunks",
            "inference_repeated": False,
            "raw_text_and_measured_times_preserved": True,
            "records": audit,
        },
    )
    for record in records:
        for key in ("reference", "reference_activity"):
            record[key] = [Segment(**s) for s in record[key]]
    summaries = []
    for model in plan["models"]:
        directory = destination / model["id"]
        rows = []
        for record in records:
            row = json.loads((directory / "predictions" / f"{record['id']}.json").read_text())
            prediction = (
                Prediction.from_dict(row["prediction"])
                if row["status"] == "ok"
                else Prediction([], timing="failed")
            )
            row["scores"] = score_record(
                record, prediction, plan["protocol"]["tcp_collar"], plan["protocol"]["der_collar"]
            )
            rows.append(row)
        write_json(directory / "scores.json", rows)
        summary = aggregate(model["id"], rows, records)
        worker = json.loads((directory / "worker.json").read_text())
        summary["sampled_process_tree_peak_rss_bytes"] = worker[
            "sampled_process_tree_peak_rss_bytes"
        ]
        summaries.append(summary)
    write_json(destination / "summary.json", summaries)
    with (destination / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = reparse(args.source, args.output)
    print(json.dumps({"reparsed": len(audit), "changes": audit}, indent=2))


if __name__ == "__main__":
    main()
