"""One model per process, so runtimes and GPU allocations do not contaminate other systems."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import resource
import sys
import time
from pathlib import Path

from speaker_benchmark.adapters import create_adapter
from speaker_benchmark.adapters.common import InvalidModelOutput
from speaker_benchmark.adapters.openrouter import RemoteRequestError
from speaker_benchmark.schema import read_json, write_json

LOG = logging.getLogger(__name__)


def seed_record(identifier):
    """Stable per-record seeds make inference independent of input ordering and retries."""
    import numpy as np

    seed = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:4], "big")
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
    return seed


def cuda_stats(reset=False):
    # Avoid importing torch for plugins that do not use it.
    import sys

    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    result = []
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
        result.append(
            {
                "device": device,
                "name": torch.cuda.get_device_name(device),
                "allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    return result


def run_job(job_path):
    # Downloads must be an explicit preparation step, never hidden in timing measurements.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    job = read_json(job_path)
    directory = Path(job["output"])
    (directory / "predictions").mkdir(exist_ok=True)
    adapter = create_adapter(job["adapter"], job["options"])
    versions = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    # Record source commit IDs without recording potentially authenticated repository URLs.
    commits = {}
    for distribution in importlib.metadata.distributions():
        direct = distribution.read_text("direct_url.json")
        if direct:
            info = json.loads(direct).get("vcs_info", {})
            if info.get("commit_id"):
                commits[distribution.metadata["Name"]] = info["commit_id"]
    started = time.perf_counter()
    try:
        adapter.load()
    except Exception as exc:
        write_json(
            directory / "failure.json",
            {
                "stage": "model_load",
                "error_type": type(exc).__name__,
                "missing_module": exc.name if isinstance(exc, ModuleNotFoundError) else None,
            },
        )
        raise
    cuda_stats()
    write_json(
        directory / "environment.json",
        {
            "packages": versions,
            "source_commits": commits,
            "load_seconds": time.perf_counter() - started,
            "cuda": cuda_stats(),
            "warmup": "none; first recording includes cold inference effects",
        },
    )
    for record in job["records"]:
        result_path = directory / "predictions" / f"{record['id']}.json"
        base = {
            "id": record["id"],
            "duration": record["duration"],
            "seed": seed_record(record["id"]),
        }
        if adapter.max_duration and record["duration"] > adapter.max_duration:
            write_json(result_path, {**base, "status": "unsupported_duration"})
            continue
        cuda_stats(reset=True)
        started = time.perf_counter()
        # Completed generations with invalid output are isolated record failures.
        # Other failures stop the worker because CUDA or model state may be damaged.
        try:
            prediction = adapter.transcribe(
                record["audio"], record["duration"], record.get("language")
            )
            cuda = cuda_stats()
        except InvalidModelOutput as exc:
            elapsed = time.perf_counter() - started
            write_json(
                result_path,
                {
                    **base,
                    "status": "invalid_output",
                    "wall_seconds": elapsed,
                    "rtf": elapsed / record["duration"],
                    "cuda_peak": cuda_stats(),
                    "raw_output": exc.raw_output,
                    "metadata": exc.metadata,
                },
            )
            LOG.error("Invalid model output for %s", record["id"])
            continue
        except Exception as exc:
            write_json(
                result_path,
                {
                    **base, "status": "inference_failed", "error_type": type(exc).__name__,
                    **({"http_status": exc.http_status, "billing_status": "unknown"}
                       if isinstance(exc, RemoteRequestError) else {}),
                },
            )
            LOG.error("Inference failed for %s (%s)", record["id"], type(exc).__name__)
            raise
        elapsed = time.perf_counter() - started
        write_json(
            result_path,
            {
                **base,
                "status": "ok",
                "prediction": prediction.to_dict(),
                "wall_seconds": elapsed,
                "rtf": elapsed / record["duration"],
                "cuda_peak": cuda,
                "process_lifetime_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        run_job(args.job)
    except Exception as exc:
        # Upstream exception messages may contain paths, prompts, or credentials.
        LOG.error(
            "Worker stopped (%s). Check local snapshots and runtime versions.", type(exc).__name__
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
