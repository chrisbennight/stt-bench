"""Inspect saved output and recover Fish Pro labels without making network requests."""

import argparse
import json
from collections import Counter
from pathlib import Path

from speaker_benchmark.adapters.openrouter import parse_response
from speaker_benchmark.runner import aggregate
from speaker_benchmark.schema import Segment, digest, read_json, write_json
from speaker_benchmark.scoring import score_record


def audit(bundle, output):
    records = [json.loads(line) for line in (bundle / "manifest.jsonl").read_text().splitlines()]
    by_id = {r["id"]: r for r in records}
    output.mkdir(parents=True, exist_ok=False)
    report = []
    for path in sorted((bundle / "scores").glob("*.json")):
        rows = read_json(path)
        counts = Counter()
        for row in rows:
            counts[row["status"]] += 1
            for metric in ("wer", "cpwer", "tcpwer", "der", "coverage"):
                counts[metric] += row["status"] == "ok" and row["scores"][metric] is not None
        report.append({"model": path.stem, "source_sha256": digest(path), "counts": dict(counts)})
        if path.stem != "fish-audio--transcribe-1-pro":
            continue
        recovered = []
        for row in rows:
            if row["status"] != "ok":
                raise ValueError("Fish recovery requires an original successful response")
            record = dict(by_id[row["id"]])
            prediction = parse_response(
                row["prediction"]["metadata"]["raw_transcript"], record["duration"],
                model="fish-audio/transcribe-1-pro",
            )
            # Retain billed usage and request provenance from the original prediction.
            prediction.metadata = {**row["prediction"]["metadata"], **prediction.metadata,
                                   "usage": row["prediction"]["metadata"]["usage"]}
            for key in ("reference", "reference_activity"):
                record[key] = [Segment(**s) for s in record[key]]
            recovered.append({**row, "prediction": prediction.to_dict(),
                              "scores": score_record(record, prediction)})
        write_json(output / "fish-reparsed-scores.json", recovered)
        summary = aggregate(path.stem, recovered, records)
        write_json(output / "fish-reparsed-summary.json", summary)
    write_json(output / "audit.json", {"models": report, "additional_api_requests": 0,
                                       "additional_api_cost_usd": 0})
    print(f"Audited {len(report)} models; recovered Fish Pro from saved text; no API requests.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("results/openrouter-2026-09-27"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.bundle, args.output)
