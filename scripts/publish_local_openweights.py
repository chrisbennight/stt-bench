"""Package the completed local counterpart experiment with portable provenance."""

import argparse
import json
import shutil
from pathlib import Path

from consolidate_results import validate_same_records

from speaker_benchmark.schema import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def publish_diagnostics(main_run, output):
    initial = main_run / "nemotron_asr_pyannote"
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    write_json(attempts / "nemotron-initial.json", {
        "worker": read_json(initial / "worker.json"),
        "predictions": [read_json(p) for p in sorted((initial / "predictions").glob("*.json"))],
    })
    diagnostic = read_json(main_run.parent / "nemotron-diagnostic.json")
    write_json(attempts / "nemotron-diagnostic.json", {
        k: diagnostic[k] for k in ("status", "stage", "error_type", "message")
    })


def publish(main_run, small_run, nemotron_run, output):
    config = read_json(ROOT / "configs/local-openweights.json")
    baseline = [json.loads(line) for line in (
        ROOT / "results/ami-4090-94clips-2026-09-27/manifest.jsonl"
    ).read_text().splitlines()]
    prepared = []
    for configured in config["models"]:
        name = configured["id"]
        source = (small_run if name == "voxtral_small_nf4_pyannote" else
                  nemotron_run if name == "nemotron_asr_pyannote" else main_run)
        records = [json.loads(line) for line in
                   (source / "manifest.jsonl").read_text().splitlines()]
        validate_same_records(baseline, records)
        blueprint = read_json(source / "plan.json")
        model = next(m for m in blueprint["models"] if m["id"] == name)
        if any(blueprint["protocol"][k] != config["protocol"][k]
               for k in ("tcp_collar", "der_collar")):
            raise ValueError("Scoring protocol differs")
        directory = source / name
        rows = read_json(directory / "scores.json")
        worker = read_json(directory / "worker.json")
        if worker["exit_code"] != 0 or worker["timed_out"]:
            raise ValueError(f"Unfinished worker: {name}")
        if (len(rows) != len(baseline)
                or {r["id"] for r in rows} != {r["id"] for r in baseline}
                or any(
                    r["status"] != "ok" and not (
                        r["status"] == "invalid_output"
                        and r.get("metadata", {}).get("generation_limit_hit") is True
                    ) for r in rows
                )):
            raise ValueError(f"Resolve incomplete predictions before publishing: {name}")
        for row in rows:
            if read_json(directory / "predictions" / f"{row['id']}.json") != {
                k: v for k, v in row.items() if k != "scores"
            }:
                raise ValueError("Scores do not correspond to saved predictions")
        prepared.append((configured, source, directory, model))
    output.mkdir(parents=True, exist_ok=False)
    sources = []
    for configured, source, directory, model in prepared:
        name = configured["id"]
        for filename, destination in [
            ("scores.json", f"scores/{name}.json"),
            ("worker.json", f"workers/{name}.json"),
            ("environment.json", f"environments/{name}/environment.json"),
            ("preflight.json", f"preflights/{name}.json"),
            ("preflight-environment.json", f"environments/{name}/preflight/environment.json"),
        ]:
            if (directory / filename).exists():
                target = output / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / filename, target)
        repairs = source / "reprocessing" / name
        if repairs.exists():
            shutil.copytree(repairs, output / "reprocessing" / name)
        sources.append({
            "model": name, "adapter": model["adapter"], "options": configured["options"],
            "source_run": source.name,
            "source_scores_sha256": digest(directory / "scores.json"),
            "source_manifest_sha256": digest(source / "manifest.jsonl"),
            "model_provenance": model["model_provenance"],
        })
    portable = [{**r, "audio": f"audio/{r['id']}.wav"} for r in baseline]
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in portable))
    write_json(output / "sources.json", {
        "models": sources, "protocol": config["protocol"],
        "configuration": "configs/local-openweights.json", "gpu": "NVIDIA GeForce RTX 4090",
        "identical_api_audio_hashes_and_references": True,
        "selection": (
            "One transcription per successful clip; failed setup attempts retained separately."
        ),
        "timing": "Includes alignment recovery work where required; model loading excluded.",
        "reused_qwen_1_7b": "results/ami-4090-94clips-2026-09-27/scores/qwen_pyannote.json",
    })
    publish_diagnostics(main_run, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--small-run", type=Path, required=True)
    parser.add_argument("--nemotron-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    publish(args.main_run, args.small_run, args.nemotron_run, args.output)
