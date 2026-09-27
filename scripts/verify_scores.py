"""Recompute every saved score without loading models or requiring original audio files."""

import argparse
import json
import logging
from pathlib import Path

from speaker_benchmark.schema import Prediction, Segment, digest, safe_id
from speaker_benchmark.scoring import score_record


def verify(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    records = {}
    for line in (directory / "manifest.jsonl").read_text().splitlines():
        record = json.loads(line)
        safe_id(record["id"])
        if record["id"] in records:
            raise ValueError("Duplicate reference recording")
        for key in ("reference", "reference_activity"):
            record[key] = [Segment(**s) for s in record[key]]
        records[record["id"]] = record
    checked, mismatches = 0, []
    score_hashes = {}
    for model in plan["models"]:
        identifier = safe_id(model["id"])
        path = directory / identifier / "scores.json"
        rows = json.loads(path.read_text())
        if len(rows) != len(records) or {r["id"] for r in rows} != set(records):
            raise ValueError(f"{identifier}: scores do not cover every recording exactly once")
        score_hashes[identifier] = digest(path)
        for row in rows:
            raw = json.loads(
                (directory / identifier / "predictions" / f"{row['id']}.json").read_text()
            )
            if raw != {k: v for k, v in row.items() if k != "scores"}:
                raise ValueError(f"{identifier}: scored output differs from saved prediction")
            if row["status"] == "ok":
                prediction = Prediction.from_dict(row["prediction"])
            elif row["status"] == "invalid_output":
                prediction = Prediction([], timing="failed")
            else:
                raise ValueError(f"{identifier}: not every recording was fully attempted")
            recomputed = score_record(
                records[row["id"]],
                prediction,
                plan["protocol"]["tcp_collar"],
                plan["protocol"]["der_collar"],
            )
            checked += 1
            if recomputed != row["scores"]:
                mismatches.append({"model": identifier, "recording": row["id"]})
    return {
        "checked_records": checked,
        "models": len(plan["models"]),
        "recordings_per_model": len(records),
        "score_mismatches": mismatches,
        "score_file_sha256": score_hashes,
        "verification": "Local recomputation from saved predictions and references",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    result = verify(args.run)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    if result["score_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
