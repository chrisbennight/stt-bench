"""Controller, reproducible plans, and corpus-weighted reporting."""

import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from speaker_benchmark.adapters import BUILTINS
from speaker_benchmark.adapters.openrouter import OPTIONS as OPENROUTER_OPTIONS
from speaker_benchmark.adapters.openrouter import validate_options
from speaker_benchmark.schema import (
    Prediction,
    digest,
    load_manifest,
    read_json,
    safe_id,
    write_json,
)
from speaker_benchmark.scoring import score_record

OPTION_KEYS = {
    "model",
    "aligner",
    "tokenizer",
    "diarizer",
    "device",
    "max_new_tokens",
    "max_new_tokens_per_chunk",
    "asr_chunk_seconds",
    "realtime",
}


def plan(config_path, manifest_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    if set(config) - {"models", "protocol"}:
        raise ValueError("Unknown top-level configuration field")
    records = load_manifest(manifest_path)
    models, seen = [], set()
    for model in config["models"]:
        if set(model) - {"id", "adapter", "python", "options"}:
            raise ValueError("Unknown model configuration field")
        identifier = safe_id(model["id"])
        if identifier in seen:
            raise ValueError("Duplicate model ID")
        seen.add(identifier)
        options = dict(model.get("options", {}))
        remote = model["adapter"] == "openrouter"
        if set(options) - (OPENROUTER_OPTIONS if remote else OPTION_KEYS):
            raise ValueError("Unknown adapter option; credentials must not be stored in config")
        if remote:
            validate_options(options)
            if len(records) > options["max_requests"] or sum(
                r["duration"] for r in records
            ) > options["max_audio_seconds_total"] + 1e-6:
                raise ValueError("Manifest exceeds the configured OpenRouter allowance")
        required = (
            {"model", "aligner", "diarizer"} if model["adapter"].startswith("qwen_") else {"model"}
        )
        if model["adapter"] in BUILTINS and not required <= set(options):
            raise ValueError(f"{identifier}: required model paths are missing")
        for key in ("max_new_tokens", "max_new_tokens_per_chunk"):
            if key in options and (type(options[key]) is not int or options[key] <= 0):
                raise ValueError("Token limits must be positive integers")
        if "realtime" in options and type(options["realtime"]) is not bool:
            raise ValueError("realtime must be a boolean")
        if "asr_chunk_seconds" in options and not 1 <= options["asr_chunk_seconds"] <= 300:
            raise ValueError("ASR chunks must be between 1 and 300 seconds")
        paths, fingerprints = {}, {}
        for key in ("model", "aligner", "diarizer", "tokenizer"):
            if key not in options or remote:
                continue
            path = (config_path.parent / options[key]).resolve()
            options[key] = str(path)
            paths[key] = path.exists()
            if path.exists():
                if path.is_file():
                    fingerprints[key] = {"sha256": digest(path)}
                else:
                    fingerprints[key] = {
                        "snapshot_directory": path.name,
                        "config_sha256": {p.name: digest(p) for p in sorted(path.glob("*.json"))},
                        "weight_files": [
                            {"name": p.name, "bytes": p.stat().st_size}
                            for p in sorted(path.iterdir())
                            if p.suffix in {".safetensors", ".bin", ".nemo"}
                        ],
                    }
        executable = (
            os.path.abspath(config_path.parent / model["python"])
            if "python" in model
            else sys.executable
        )
        models.append(
            {
                "id": identifier,
                "adapter": model["adapter"],
                "python": executable,
                "options": options,
                "paths_exist": paths,
                "model_provenance": fingerprints,
                "unsupported_recordings": [
                    r["id"]
                    for r in records
                    if model["adapter"] in BUILTINS
                    and (
                        r["duration"] > options.get("max_audio_seconds", 60)
                        if remote else (
                            BUILTINS[model["adapter"]].max_duration is not None
                            and r["duration"] > BUILTINS[model["adapter"]].max_duration
                        )
                    )
                ],
            }
        )
    if not models:
        raise ValueError("No models configured")
    protocol = config.get("protocol", {})
    if set(protocol) - {"tcp_collar", "der_collar", "worker_timeout_seconds"}:
        raise ValueError("Unknown protocol field")
    for key, default in (
        ("tcp_collar", 5.0),
        ("der_collar", 0.0),
        ("worker_timeout_seconds", 7200),
    ):
        value = float(protocol.get(key, default))
        if not 0 <= value < float("inf") or (key == "worker_timeout_seconds" and value == 0):
            raise ValueError("Invalid protocol limit")
        protocol[key] = value
    protocol["random_seed_policy"] = "sha256-record-id-first-32-bits-v1"
    return {
        "models": models,
        "protocol": protocol,
        "manifest_sha256": digest(manifest_path),
        "config_sha256": digest(config_path),
    }, records


def run(config_path, manifest_path, output, parallel_models=1):
    if type(parallel_models) is not int or parallel_models < 1:
        raise ValueError("parallel_models must be a positive integer")
    blueprint, records = plan(config_path, manifest_path)
    blueprint["protocol"]["parallel_models"] = parallel_models
    for model in blueprint["models"]:
        if model["adapter"] == "openrouter" and model["unsupported_recordings"]:
            raise ValueError("Prepare shorter audio windows before running OpenRouter")
        if not all(model["paths_exist"].values()) or not Path(model["python"]).is_file():
            raise ValueError(
                f"{model['id']}: missing model snapshots or Python environment; run plan"
            )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", blueprint)
    write_json(
        output / "host.json",
        {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count()},
    )
    # Snapshot the references only in the controller output, not in the worker job.
    with (output / "manifest.jsonl").open("w") as stream:
        for row in records:
            saved = {
                **row,
                "reference": [asdict(s) for s in row["reference"]],
                "reference_activity": [asdict(s) for s in row["reference_activity"]],
            }
            stream.write(json.dumps(saved) + "\n")
    jobs = []
    for model in blueprint["models"]:
        directory = output / model["id"]
        directory.mkdir()
        job = {
            "adapter": model["adapter"],
            "options": model["options"],
            "output": str(directory),
            "records": [
                {k: r[k] for k in ("id", "audio", "duration", "language") if k in r}
                for r in records
            ],
        }
        write_json(directory / "job.json", job)
        argv = [model["python"], "-m", "speaker_benchmark.worker", str(directory / "job.json")]
        jobs.append((directory, argv))
    run_workers(jobs, parallel_models, blueprint["protocol"]["worker_timeout_seconds"])
    return report(output)


def run_workers(jobs, parallel_models, timeout):
    """Bound independent model processes; cancellation also stops their descendants."""
    pending = iter(jobs)
    active = {}
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < parallel_models and not exhausted:
                job = next(pending, None)
                if job is None:
                    exhausted = True
                    break
                directory, argv = job
                # Upstream output may expose credentials. Save structured diagnostics only.
                process = subprocess.Popen(
                    argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                active[process] = {"directory": directory, "started": time.monotonic(), "peak": 0}
            for process, state in list(active.items()):
                timed_out = False
                if process.poll() is None and time.monotonic() - state["started"] > timeout:
                    timed_out = True
                    stop_worker(process)
                if process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        state["peak"] = max(
                            state["peak"], sum(
                                p.memory_info().rss
                                for p in [parent] + parent.children(recursive=True)
                            ),
                        )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # A process may exit between enumeration and sampling.
                    continue
                process.wait()
                write_json(state["directory"] / "worker.json", {
                    "exit_code": process.returncode, "timed_out": timed_out,
                    "sampled_process_tree_peak_rss_bytes": state["peak"],
                })
                del active[process]
            if active:
                time.sleep(0.1)
    finally:
        for process in active:
            if process.poll() is None:
                stop_worker(process)


def stop_worker(process):
    import signal

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def report(output):
    output = Path(output)
    blueprint = read_json(output / "plan.json")
    records = load_manifest(output / "manifest.jsonl")
    summaries = []
    for model in blueprint["models"]:
        directory = output / model["id"]
        rows = []
        for record in records:
            path = directory / "predictions" / f"{record['id']}.json"
            row = read_json(path) if path.exists() else {"id": record["id"], "status": "not_run"}
            if row["status"] == "ok":
                prediction = Prediction.from_dict(row["prediction"])
            else:
                # Failed recordings count as empty hypotheses instead of vanishing from averages.
                prediction = Prediction([], timing="failed")
                if model.get("adapter") == "openrouter":
                    prediction.metadata["speaker_labels_available"] = False
            row["scores"] = score_record(
                record,
                prediction,
                blueprint["protocol"]["tcp_collar"],
                blueprint["protocol"]["der_collar"],
            )
            rows.append(row)
        write_json(directory / "scores.json", rows)
        summary = aggregate(model["id"], rows, records)
        worker_path = directory / "worker.json"
        worker = read_json(worker_path) if worker_path.exists() else {}
        summary["sampled_process_tree_peak_rss_bytes"] = worker.get(
            "sampled_process_tree_peak_rss_bytes"
        )
        summaries.append(summary)
    write_json(output / "summary.json", summaries)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for summary in summaries for key in summary
        )))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def aggregate(identifier, rows, records):
    ok = [r for r in rows if r["status"] == "ok"]
    attempted = [r for r in rows if r["status"] in {"ok", "invalid_output"}]
    summary = {
        "model": identifier,
        "recordings": len(rows),
        "completed": len(ok),
        "failed_or_unsupported": len(rows) - len(ok),
        "invalid_output": sum(r["status"] == "invalid_output" for r in rows),
    }
    for name in ("wer", "cpwer", "tcpwer"):
        scores = [r["scores"][name] for r in rows]
        length = sum(s["length"] for s in scores if s is not None)
        summary[name] = (
            sum(s["errors"] for s in scores) / length
            if all(s is not None for s in scores) and length
            else None
        )
    ders = [r["scores"]["der"] for r in rows]
    total = sum(d["total"] for d in ders if d)
    summary["der"] = (
        sum(d["false alarm"] + d["missed detection"] + d["confusion"] for d in ders) / total
        if all(d is not None for d in ders) and total
        else None
    )
    coverage = [r["scores"]["coverage"] for r in rows]
    total = sum(c["reference_seconds"] for c in coverage if c)
    summary["speech_time_coverage"] = (
        sum(c["covered_seconds"] for c in coverage) / total
        if all(c is not None for c in coverage) and total
        else None
    )
    summary["wall_seconds_completed"] = sum(r["wall_seconds"] for r in ok)
    durations = {r["id"]: r["duration"] for r in records}
    seconds = sum(durations[r["id"]] for r in ok)
    summary["rtf_completed"] = summary["wall_seconds_completed"] / seconds if seconds else None
    attempted_seconds = sum(durations[r["id"]] for r in attempted)
    summary["wall_seconds_attempted"] = sum(r["wall_seconds"] for r in attempted)
    summary["rtf_attempted"] = (
        summary["wall_seconds_attempted"] / attempted_seconds if attempted_seconds else None
    )
    summary["max_cuda_allocated_bytes"] = max(
        (d["allocated_bytes"] for r in attempted for d in r.get("cuda_peak") or []), default=None
    )
    summary["max_cuda_reserved_bytes"] = max(
        (d["reserved_bytes"] for r in attempted for d in r.get("cuda_peak") or []), default=None
    )
    summary["timing_sources"] = ",".join(sorted({r["scores"]["timing"] for r in ok}))
    remote_rows = [r for r in ok if r["prediction"]["metadata"].get("execution") == "remote_api"]
    if remote_rows:
        costs = [r["prediction"]["metadata"].get("usage", {}).get("cost") for r in remote_rows]
        summary["api_reported_cost_usd"] = sum(c for c in costs if c is not None)
        summary["api_cost_complete"] = (
            len(remote_rows) == len(rows) and all(c is not None for c in costs)
        )
    summary["generation_limit_hits"] = sum(
        r["prediction"].get("metadata", {}).get("generation_limit_hit") is True for r in ok
    ) + sum(
        r.get("metadata", {}).get("generation_limit_hit") is True
        for r in rows
        if r["status"] == "invalid_output"
    )
    events = [(r, e) for r in ok for e in r["prediction"].get("events", []) if e["has_text"]]
    summary["mean_first_text_seconds"] = None
    summary["p95_chunk_emission_lag_seconds"] = None
    paced_rows = [r for r in ok if r["prediction"]["metadata"].get("realtime_paced")]
    if paced_rows:
        import numpy as np

        first = [
            next((e["emitted_seconds"] for e in r["prediction"]["events"] if e["has_text"]), None)
            for r in paced_rows
        ]
        first = [t for t in first if t is not None]
        lag = [
            e["emitted_seconds"] - e["chunk_end_seconds"]
            for r, e in events
            if r["prediction"]["metadata"].get("realtime_paced")
        ]
        summary["mean_first_text_seconds"] = sum(first) / len(first) if first else None
        summary["p95_chunk_emission_lag_seconds"] = float(np.percentile(lag, 95)) if lag else None
    return summary
