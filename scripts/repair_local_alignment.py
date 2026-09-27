"""Recover alignment-rejected local transcripts without repeating ASR inference."""

import argparse
import time
from pathlib import Path

from speaker_benchmark.adapters.local_asr import AlignedPyannote
from speaker_benchmark.runner import report
from speaker_benchmark.schema import digest, read_json, write_json
from speaker_benchmark.worker import cuda_stats, seed_record


class SavedTranscript(AlignedPyannote):
    def load_asr(self):
        pass

    def recognize(self, audio_path, audio, language):
        return self.text, {"asr_mode": "retained_transcript_alignment_repair"}


def repair(run_path):
    run_path = Path(run_path)
    if not (run_path / "completion.json").exists():
        raise ValueError("Wait for the inference controller to finish before repairing outputs")
    if (run_path / "alignment-repairs.json").exists():
        raise FileExistsError("Alignment repairs have already been recorded")
    blueprint = read_json(run_path / "plan.json")
    repaired = []
    for model in blueprint["models"]:
        directory = run_path / model["id"]
        job = read_json(directory / "job.json")
        records = {r["id"]: r for r in job["records"]}
        candidates = []
        for path in sorted((directory / "predictions").glob("*.json")):
            row = read_json(path)
            if (row["status"] == "invalid_output"
                    and row.get("metadata", {}).get("reason") == "alignment_changed_transcript"):
                candidates.append(path)
        if not candidates:
            continue
        adapter = SavedTranscript(model["options"])
        adapter.load()
        for path in candidates:
            source_hash = digest(path)
            original = read_json(path)
            record = records[original["id"]]
            adapter.text = original["raw_output"]
            seed_record(record["id"])
            cuda_stats(reset=True)
            started = time.perf_counter()
            prediction = adapter.transcribe(
                record["audio"], record["duration"], record.get("language"),
            )
            cuda = cuda_stats()
            elapsed = time.perf_counter() - started
            if digest(path) != source_hash:
                raise RuntimeError("Prediction changed during repair; preserve the newer output")
            archive = run_path / "reprocessing" / model["id"] / path.name
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                raise FileExistsError("Original attempt archive already exists")
            write_json(archive, original)
            prediction.metadata.update({
                "asr_reused": True, "initial_attempt_seconds": original["wall_seconds"],
                "reprocessing_seconds": elapsed, "original_prediction_sha256": source_hash,
            })
            total = original["wall_seconds"] + elapsed
            write_json(path, {
                **{k: original[k] for k in ("id", "duration", "seed")},
                "status": "ok", "prediction": prediction.to_dict(),
                "wall_seconds": total, "rtf": total / record["duration"],
                "cuda_peak": [*(original.get("cuda_peak") or []), *(cuda or [])],
            })
            repaired.append({"model": model["id"], "clip": record["id"]})
        del adapter
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    write_json(run_path / "alignment-repairs.json", repaired)
    summaries = report(run_path)
    write_json(run_path / "completion.json", {
        "all_complete": all(s["completed"] == s["recordings"] for s in summaries),
    })
    return repaired


if __name__ == "__main__":
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    repair(parser.parse_args().run)
