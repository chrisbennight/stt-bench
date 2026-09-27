"""Rescore saved Deepgram responses offline into a separate, new run directory."""

import argparse
import json
import shutil
from pathlib import Path

from speaker_benchmark.adapters.openrouter import parse_response
from speaker_benchmark.runner import report
from speaker_benchmark.schema import digest, read_json, safe_id, write_json


def reparse(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source in output.parents:
        raise ValueError("Output must be a new directory outside the source run")
    plan = read_json(source / "plan.json")
    if len(plan["models"]) != 1:
        raise ValueError("Use the dedicated Deepgram run")
    model = plan["models"][0]
    if model["adapter"] != "openrouter" or model["options"]["model"] != "deepgram/nova-3":
        raise ValueError("Expected the Deepgram OpenRouter adapter")
    identifier = safe_id(model["id"])
    corrected, hashes = [], {}
    for line in (source / "manifest.jsonl").read_text().splitlines():
        record = json.loads(line)
        path = source / identifier / "predictions" / f"{safe_id(record['id'])}.json"
        row = read_json(path)
        hashes[path.name] = digest(path)
        if row["status"] == "ok":
            metadata = row["prediction"]["metadata"]
            data = {**metadata["raw_transcript"], "usage": metadata["usage"]}
        elif row["status"] == "invalid_output":
            metadata, data = row["metadata"], row["raw_output"]
        else:
            raise ValueError("Every source clip must have a saved response")
        prediction = parse_response(data, row["duration"], clip_timestamps=True)
        for key in ("model", "generation_id"):
            if key in metadata:
                prediction.metadata[key] = metadata[key]
        previous_status = row["status"]
        row = {k: v for k, v in row.items() if k not in {"metadata", "raw_output", "prediction"}}
        row.update(status="ok", prediction=prediction.to_dict(), original_status=previous_status)
        corrected.append((path.name, row))
    # Validate every response before writing; the source run is never modified.
    output.mkdir(parents=True)
    for name in ("plan.json", "manifest.jsonl"):
        shutil.copyfile(source / name, output / name)
    destination = output / identifier / "predictions"
    destination.mkdir(parents=True)
    for name, row in corrected:
        write_json(destination / name, row)
    write_json(output / "reparse-provenance.json", {
        "operation": "Offline Deepgram response parsing; no API requests",
        "source_predictions_sha256": hashes,
        "source_plan_sha256": digest(source / "plan.json"),
        "parser_sha256": digest(Path(__file__).resolve().parents[1]
                                / "src/speaker_benchmark/adapters/openrouter.py"),
        "script_sha256": digest(__file__),
        "inference_wall_times": "Preserved from source requests, excluding offline scoring",
    })
    return report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(reparse(args.source, args.output), indent=2))
