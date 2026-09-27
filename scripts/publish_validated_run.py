"""Package this experiment's completed hosted run without making any API calls."""

import hashlib
import json
from pathlib import Path

from consolidate_results import reported_cost, validate_same_records

from speaker_benchmark.adapters.openrouter import parse_response
from speaker_benchmark.schema import Segment, read_json, write_json
from speaker_benchmark.scoring import score_record

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/openrouter-validated-full"
BUNDLE = ROOT / "results/openrouter-validated-2026-09-27"
RECOVERY = ROOT / "runs/openrouter-chirp-missing"
PROBES = (
    "openrouter-capability-probe", "chirp-capability-diagnostic",
    "chirp-standard-timestamps-probe", "openrouter-nine-capability-probe",
    "openrouter-capability-variants", "openrouter-response-schema-probe",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def cost_entry(row):
    return {"clip": row["id"], "status": row["status"],
            "reported_cost_usd": reported_cost(row)}


def publish():
    plan = read_json(RUN / "plan.json")
    manifest = records(RUN / "manifest.jsonl")
    local = records(ROOT / "results/ami-4090-94clips-2026-09-27/manifest.jsonl")
    validate_same_records(local, manifest)
    if len(manifest) != 94 or len(plan["models"]) != 20:
        raise ValueError("Expected the approved 20-model, 94-clip run")
    ids = {r["id"] for r in manifest}
    record_map = {r["id"]: r for r in manifest}
    artifacts, sources, ledger = {}, [], []
    for model in plan["models"]:
        name = model["id"]
        directory = RUN / name
        worker = read_json(directory / "worker.json")
        recovering = name == "google--chirp-3"
        if not recovering and (worker["exit_code"] != 0 or worker["timed_out"]):
            raise ValueError(f"Worker did not finish successfully: {name}")
        scores = read_json(directory / "scores.json")
        ledger.extend({"run": RUN.name, "model": name, **cost_entry(r)}
                      for r in scores if r["status"] != "not_run")
        source = {"model": name, "api_model": model["options"]["model"],
                  "source_run": RUN.name, "options": model["options"],
                  "source_scores_sha256": digest(directory / "scores.json")}
        if recovering:
            recovery_plan = read_json(RECOVERY / "plan.json")
            recovery_manifest = records(RECOVERY / "manifest.jsonl")
            selected_ids = {r["id"] for r in recovery_manifest}
            expected = [r for r in manifest if r["id"] in selected_ids]
            validate_same_records(sorted(expected, key=lambda r: r["id"]),
                                  sorted(recovery_manifest, key=lambda r: r["id"]))
            recovery_worker = read_json(RECOVERY / name / "worker.json")
            if recovery_worker["exit_code"] not in (0, 1) or recovery_worker["timed_out"]:
                raise ValueError("Unexpected Chirp recovery termination")
            replacements = read_json(RECOVERY / name / "scores.json")
            if (len(replacements) != len(selected_ids)
                    or {r["id"] for r in replacements} != selected_ids
                    or any(r["status"] == "not_run" for r in replacements)):
                raise ValueError("Chirp recovery scores differ from its manifest")
            original_by_id = {r["id"]: r for r in scores}
            if any(original_by_id[r["id"]]["status"] == "ok" for r in replacements):
                raise ValueError("Recovery must not replace a successful response")
            artifacts["attempts/chirp-original.json"] = scores
            artifacts["attempts/chirp-recovery.json"] = replacements
            artifacts["workers/chirp-recovery.json"] = recovery_worker
            original_by_id.update({r["id"]: r for r in replacements})
            scores = [original_by_id[r["id"]] for r in manifest]
            source["recovery"] = {
                "run": RECOVERY.name, "clip_ids": sorted(selected_ids),
                "options": recovery_plan["models"][0]["options"],
                "scores_sha256": digest(RECOVERY / name / "scores.json"),
                "reason": "Client timeout; identical PCM samples, WAV transport, longer timeout",
            }
            ledger.extend({"run": RECOVERY.name, "model": name, **cost_entry(r)}
                          for r in replacements)
        corrected = []
        for index, row in enumerate(scores):
            if (row["status"] != "invalid_output"
                    or row.get("metadata", {}).get("reason") != "nonpositive_transcript_interval"):
                continue
            prediction = parse_response(row["raw_output"], row["duration"],
                                        clip_timestamps=True, model=model["options"]["model"])
            for key in ("model", "generation_id"):
                if key in row["metadata"]:
                    prediction.metadata[key] = row["metadata"][key]
            record = {**record_map[row["id"]]}
            for key in ("reference", "reference_activity"):
                record[key] = [Segment(**s) for s in record[key]]
            artifacts[f"reprocessing/{name}/{row['id']}.json"] = row
            replacement = {k: v for k, v in row.items()
                           if k not in {"raw_output", "metadata", "scores"}}
            replacement.update(status="ok", original_status=row["status"],
                               prediction=prediction.to_dict(), scores=score_record(
                                   record, prediction, plan["protocol"]["tcp_collar"],
                                   plan["protocol"]["der_collar"]))
            scores[index] = replacement
            corrected.append(row["id"])
        if corrected:
            source["offline_reparsed_clips"] = corrected
            source["reprocessing"] = (
                "Use valid native segments after reversed word times; no API call"
            )
        if len(scores) != 94 or {r["id"] for r in scores} != ids:
            raise ValueError(f"Incomplete score inventory: {name}")
        allowed = {"ok", "invalid_output", "inference_failed"} if recovering else {
            "ok", "invalid_output"
        }
        if any(r["status"] not in allowed for r in scores):
            raise ValueError(f"Unresolved requests: {name}")
        if recovering:
            failures = [r for r in scores if r["status"] == "inference_failed"]
            if len(failures) != 1 or failures[0].get("http_status") != 504:
                raise ValueError("Reconcile unexpected Chirp failure results")
        artifacts[f"scores/{name}.json"] = scores
        artifacts[f"environments/{name}/environment.json"] = read_json(
            directory / "environment.json"
        )
        artifacts[f"workers/{name}.json"] = worker
        sources.append(source)
    audit = read_json(ROOT / "research/openrouter-capabilities.json")
    sources.extend({"model": m["id"], "api_model": m["model"], "excluded": True,
                    "reason": "DeepInfra route excluded by user after earlier failures"}
                   for m in audit["models"] if m["status"] == "excluded_by_user")
    for run_name in PROBES:
        directory = ROOT / "runs" / run_name
        probe_plan = read_json(directory / "plan.json")
        for model in probe_plan["models"]:
            predictions = sorted((directory / model["id"] / "predictions").glob("*.json"))
            if len(predictions) != 1:
                raise ValueError(f"Expected one probe response: {run_name}/{model['id']}")
            row = read_json(predictions[0])
            artifacts[f"probes/{run_name}/{model['id']}.json"] = {
                "options": model["options"], "response": row,
                "source_prediction_sha256": digest(predictions[0]),
            }
            ledger.append({"run": run_name, "model": model["id"], **cost_entry(row)})
    if len(ledger) != 1880 + 42 + 1:
        raise ValueError("Unexpected request count; reconcile the ledger before publishing")
    artifacts["sources.json"] = {
        "models": sources, "tcp_collar": plan["protocol"]["tcp_collar"],
        "der_collar": plan["protocol"]["der_collar"],
        "protocol": plan["protocol"], "source_manifest_sha256": plan["manifest_sha256"],
        "config_sha256": plan["config_sha256"],
        "selection": "One corrected pass plus only missing Chirp clips; no score-based selection.",
        "publication_source_sha256": {str(p.relative_to(ROOT)): digest(p)
                                      for p in sorted((ROOT / "src").rglob("*.py"))},
    }
    artifacts["probe-selection.json"] = read_json(
        ROOT / "runs/openrouter-final-probe/selection.json"
    )
    final_probe = ROOT / "runs/openrouter-final-probe"
    probe_plan = read_json(final_probe / "plan.json")
    for model in probe_plan["models"]:
        model.pop("python", None)
        for path in (final_probe / model["id"] / "predictions").glob("*.json"):
            artifacts[f"validated-probe/{model['id']}/predictions/{path.name}"] = read_json(path)
    artifacts["validated-probe/plan.json"] = probe_plan
    artifacts["costs.json"] = {
        "scope": "Temporary recovery key: capability probes and corrected full pass only",
        "requests": ledger,
        "request_count": len(ledger),
        "reported_cost_usd": sum(r["reported_cost_usd"] or 0 for r in ledger),
        "unknown_billing_requests": sum(r["reported_cost_usd"] is None for r in ledger),
        "account_reconciliation": None,
        "note": "Unknown billing is not a zero charge. Prior benchmark uses a different key.",
    }
    if BUNDLE.exists():
        raise ValueError("Output exists; preserve published artifacts")
    for name, data in artifacts.items():
        (BUNDLE / name).parent.mkdir(parents=True, exist_ok=True)
        write_json(BUNDLE / name, data)
    portable = [{**r, "audio": f"audio/{r['id']}.wav"} for r in manifest]
    (BUNDLE / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in portable)
    )
    probe_records = [{**r, "audio": f"audio/{r['id']}.wav"}
                     for r in records(final_probe / "manifest.jsonl")]
    (BUNDLE / "validated-probe/manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in probe_records)
    )
    print(f"Published {len(sources)} model entries and {len(ledger)} request records")


if __name__ == "__main__":
    publish()
