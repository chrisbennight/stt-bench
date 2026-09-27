"""Run each local pipeline once, checking its first clip before continuing."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from speaker_benchmark.runner import plan, report, run_workers
from speaker_benchmark.schema import Prediction, load_manifest, read_json, write_json
from speaker_benchmark.scoring import score_record


def input_signature(records):
    return {
        r["id"]: {
            "audio_sha256": r["audio_sha256"], "duration": r["duration"],
            "reference": [asdict(s) for s in r["reference"]],
            "reference_activity": [asdict(s) for s in r["reference_activity"]],
        }
        for r in records
    }


def run(config, manifest, baseline_manifest, output):
    blueprint, records = plan(config, manifest)
    if input_signature(records) != input_signature(load_manifest(baseline_manifest)):
        raise ValueError("Audio or references differ from the comparison baseline")
    recorded_hashes = {
        row["id"]: row.get("audio_sha256")
        for row in (json.loads(line) for line in Path(baseline_manifest).read_text().splitlines())
    }
    if recorded_hashes != {r["id"]: r["audio_sha256"] for r in records}:
        raise ValueError("Audio differs from the baseline's recorded hashes")
    for model in blueprint["models"]:
        if model["adapter"] == "openrouter":
            raise ValueError("This controller is for local models only")
        if not all(model["paths_exist"].values()) or not Path(model["python"]).is_file():
            raise ValueError(f"Missing runtime or snapshot for {model['id']}")
        if model["unsupported_recordings"]:
            raise ValueError("The full comparison must fit every adapter's duration limit")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    blueprint["protocol"]["parallel_models"] = 1
    blueprint["protocol"]["preflight"] = "first clip retained; remaining clips run once"
    write_json(output / "plan.json", blueprint)
    with (output / "manifest.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps({
                **record,
                "reference": [asdict(s) for s in record["reference"]],
                "reference_activity": [asdict(s) for s in record["reference_activity"]],
            }) + "\n")
    timeout = blueprint["protocol"]["worker_timeout_seconds"]
    for model in blueprint["models"]:
        directory = output / model["id"]
        directory.mkdir()
        job = {
            "adapter": model["adapter"], "options": model["options"],
            "output": str(directory),
            "records": [
                {k: r[k] for k in ("id", "audio", "duration", "language") if k in r}
                for r in records
            ],
        }
        write_json(directory / "job.json", job)
        first_job = directory / "preflight-job.json"
        write_json(first_job, {**job, "records": job["records"][:1]})
        run_workers([
            (directory, [model["python"], "-m", "speaker_benchmark.worker", str(first_job)])
        ], 1, timeout)
        worker = read_json(directory / "worker.json")
        write_json(directory / "preflight-worker.json", worker)
        result_path = directory / "predictions" / f"{records[0]['id']}.json"
        result = read_json(result_path) if result_path.exists() else {}
        valid = False
        if worker["exit_code"] == 0 and result.get("status") == "ok":
            prediction = Prediction.from_dict(result["prediction"])
            scores = score_record(
                records[0], prediction, blueprint["protocol"]["tcp_collar"],
                blueprint["protocol"]["der_collar"],
            )
            valid = bool(prediction.segments) and any(
                s.speaker != "unassigned" for s in prediction.segments
            ) and all(scores[k] is not None for k in ("wer", "cpwer", "tcpwer", "der"))
        write_json(directory / "preflight.json", {"passed": valid})
        if valid and len(records) > 1:
            write_json(directory / "preflight-environment.json", read_json(
                directory / "environment.json",
            ))
            remaining_job = directory / "remaining-job.json"
            write_json(remaining_job, {**job, "records": job["records"][1:]})
            run_workers([
                (directory, [model["python"], "-m", "speaker_benchmark.worker", str(remaining_job)])
            ], 1, timeout)
            remaining_worker = read_json(directory / "worker.json")
            remaining_worker["sampled_process_tree_peak_rss_bytes"] = max(
                worker["sampled_process_tree_peak_rss_bytes"],
                remaining_worker["sampled_process_tree_peak_rss_bytes"],
            )
            write_json(directory / "worker.json", remaining_worker)
        write_json(output / "progress.json", {"last_finished": model["id"]})
    summaries = report(output)
    write_json(output / "completion.json", {
        "all_complete": all(s["completed"] == len(records) for s in summaries),
    })
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.manifest, args.baseline_manifest, args.output)
