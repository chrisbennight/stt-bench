"""Combine completed runs of the same protocol and retain meeting-level results."""

import argparse
import csv
import json
from pathlib import Path

from speaker_benchmark.runner import aggregate
from speaker_benchmark.schema import safe_id


def compare(directories, selections=None):
    selections = {k: Path(v).resolve() for k, v in (selections or {}).items()}
    manifest_hash = protocol = records = None
    summaries, meetings, windows = [], [], []
    seen = set()
    for directory in map(Path, directories):
        plan = json.loads((directory / "plan.json").read_text())
        if manifest_hash is None:
            manifest_hash, protocol = plan["manifest_sha256"], plan["protocol"]
            records = [
                json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()
            ]
        elif plan["manifest_sha256"] != manifest_hash or plan["protocol"] != protocol:
            raise ValueError("Runs must use the same input manifest and scoring protocol")
        for model in plan["models"]:
            identifier = safe_id(model["id"])
            if identifier in selections and selections[identifier] != directory.resolve():
                continue
            if identifier in seen:
                raise ValueError("Duplicate system in comparison")
            seen.add(identifier)
            path = directory / identifier
            worker = json.loads((path / "worker.json").read_text())
            rows = json.loads((path / "scores.json").read_text())
            if worker["exit_code"] != 0 or any(
                row["status"] not in {"ok", "invalid_output"} for row in rows
            ):
                raise ValueError(f"{identifier}: incomplete run; resolve failures before ranking")
            if {row["id"] for row in rows} != {row["id"] for row in records}:
                raise ValueError("Prediction IDs do not match the reference set")
            summary = aggregate(identifier, rows, records)
            summary["sampled_process_tree_peak_rss_bytes"] = worker[
                "sampled_process_tree_peak_rss_bytes"
            ]
            summary["source_run"] = str(directory)
            summaries.append(summary)
            for meeting in sorted({r["source"]["recording"] for r in records}):
                selected = [r for r in records if r["source"]["recording"] == meeting]
                ids = {r["id"] for r in selected}
                subset = [row for row in rows if row["id"] in ids]
                meetings.append({"meeting": meeting, **aggregate(identifier, subset, selected)})
            for row in rows:
                score = row["scores"]
                windows.append(
                    {
                        "model": identifier,
                        "recording": row["id"],
                        "status": row["status"],
                        "reference_speakers": score["reference_speakers"],
                        "predicted_speakers": score["predicted_speakers"],
                        "cpwer_errors": score["cpwer"]["errors"],
                        "reference_words": score["cpwer"]["length"],
                        "wall_seconds": row["wall_seconds"],
                        "rtf": row["rtf"],
                    }
                )
    if set(selections) - seen:
        raise ValueError("Selected system was not found in its chosen run")
    return {
        "manifest_sha256": manifest_hash,
        "protocol": protocol,
        "audio_seconds": sum(r["duration"] for r in records),
        "recordings": len(records),
        "summary": summaries,
        "meetings": meetings,
        "windows": windows,
        "limits": [
            "Meeting-level variation is descriptive; correlated windows are not independent.",
            "AMI results do not establish general performance on other domains or languages.",
            "Speaker labels are anonymous identities within each independent audio window.",
            "Streaming DER, tcpWER, and coverage are unavailable without speech timestamps.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", action="append", default=[], metavar="MODEL=RUN_DIRECTORY")
    args = parser.parse_args()
    selections = {}
    for entry in args.select:
        identifier, separator, path = entry.partition("=")
        if not separator or not identifier or not path or identifier in selections:
            parser.error("Each selection must be a unique MODEL=RUN_DIRECTORY")
        selections[identifier] = path
    result = compare(args.runs, selections)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in ("summary", "meetings", "windows"):
        with (args.output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[name][0]))
            writer.writeheader()
            writer.writerows(result[name])
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
