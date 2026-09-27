"""Require saved output evidence before generating a full paid-run configuration."""

import argparse
import json
from pathlib import Path

from speaker_benchmark.schema import Prediction, Segment, read_json, write_json
from speaker_benchmark.scoring import score_record


def validate_probe(config, audit, plan, records, predictions):
    expected = {m["id"]: m for m in config["models"]}
    planned = {m["id"]: m for m in plan["models"]}
    if len(records) != 1 or set(expected) != set(planned):
        raise ValueError("Probe must use one shared clip and the exact candidate model set")
    for key in ("id", "duration", "audio_sha256", "reference", "reference_activity"):
        if records[0][key] != audit["probe_record"][key]:
            raise ValueError("Probe audio and references differ from the reviewed clip")
    requirements = {m["id"]: m["required_probe_metrics"] for m in audit["models"]}
    record = dict(records[0])
    for key in ("reference", "reference_activity"):
        record[key] = [Segment(**s) for s in record[key]]
    findings = []
    for identifier, model in expected.items():
        if planned[identifier]["options"] != model["options"]:
            raise ValueError("Probe options differ from the reviewed candidate")
        rows = predictions[identifier]
        if len(rows) != 1 or rows[0]["id"] != record["id"] or rows[0]["status"] != "ok":
            findings.append({"model": identifier, "reason": "probe_not_successful"})
            continue
        scores = score_record(record, Prediction.from_dict(rows[0]["prediction"]))
        missing = [metric for metric in requirements[identifier] if scores[metric] is None]
        if missing:
            findings.append({"model": identifier, "reason": "missing_requested_metrics",
                             "metrics": missing})
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json("configs/openrouter-capability-probe.json")
    audit = read_json("research/openrouter-capabilities.json")
    plan = read_json(args.run / "plan.json")
    records = [json.loads(line) for line in (args.run / "manifest.jsonl").read_text().splitlines()]
    predictions = {
        m["id"]: [read_json(p) for p in sorted((args.run / m["id"] / "predictions").glob("*.json"))]
        for m in config["models"]
    }
    findings = validate_probe(config, audit, plan, records, predictions)
    if findings:
        print(json.dumps({"ready": False, "findings": findings}, indent=2))
        raise SystemExit(2)
    for model in config["models"]:
        model["options"].update(max_requests=94, max_audio_seconds_total=5536.704)
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing full-run configuration")
    write_json(args.output, config)
    print("All required metrics present. Full configuration written; no API requests made.")
