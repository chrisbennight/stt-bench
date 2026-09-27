"""Command-line entry points."""

import argparse
import json
import logging
from pathlib import Path

from speaker_benchmark import datasets
from speaker_benchmark.runner import plan, report, run
from speaker_benchmark.schema import Prediction, Segment, load_manifest, read_json, write_json
from speaker_benchmark.scoring import score_record


def main():
    parser = argparse.ArgumentParser(description="Compare speaker-attributed speech systems")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = commands.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        if name == "run":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--parallel-models", type=int, default=1,
                           help="Concurrent model workers; each processes its clips sequentially")
    p = commands.add_parser("report")
    p.add_argument("output", type=Path)
    p = commands.add_parser("score")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSON object keyed by recording ID, each value a Prediction",
    )
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("fetch-ami")
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--meetings", nargs="+", default=["ES2004a"])
    p = commands.add_parser("prepare-ami")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--meetings", nargs="+", default=["ES2004a"])
    p.add_argument(
        "--window-seconds", type=float, default=240, help="Use 0 for complete recordings"
    )
    p = commands.add_parser("import-seglst")
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--audio-map", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p = commands.add_parser("smoke")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "plan":
        blueprint, records = plan(args.config, args.manifest)
        blueprint["recordings"] = len(records)
        blueprint["audio_seconds"] = sum(r["duration"] for r in records)
        print(json.dumps(blueprint, indent=2))
    elif args.command == "run":
        summaries = run(args.config, args.manifest, args.output, args.parallel_models)
        print(json.dumps(summaries, indent=2))
        if any(s["failed_or_unsupported"] for s in summaries):
            raise SystemExit(2)
    elif args.command == "report":
        print(json.dumps(report(args.output), indent=2))
    elif args.command == "score":
        predictions = read_json(args.predictions)
        records = load_manifest(args.manifest)
        if set(predictions) != {r["id"] for r in records}:
            raise ValueError("Predictions must contain every reference ID exactly once")
        write_json(
            args.output,
            {r["id"]: score_record(r, Prediction.from_dict(predictions[r["id"]])) for r in records},
        )
    elif args.command == "fetch-ami":
        datasets.fetch_ami(args.destination, args.meetings)
    elif args.command == "prepare-ami":
        datasets.prepare_ami(args.source, args.destination, args.meetings, args.window_seconds)
    elif args.command == "import-seglst":
        datasets.import_seglst(args.references, args.audio_map, args.destination)
    elif args.command == "smoke":
        # Deliberate synthetic fixture: exercises scoring, never presents model quality numbers.
        reference = [Segment("alice", "hello there", 0, 1), Segment("bob", "good morning", 2, 3)]
        record = {"reference": reference, "reference_activity": reference, "duration": 4}
        result = score_record(
            record,
            Prediction([Segment("one", "hello", 0, 1), Segment("one", "good morning", 2, 3)]),
        )
        write_json(args.output, {"synthetic_fixture_only": True, "scores": result})
        print("Synthetic scoring smoke check completed; no speech model was run.")


if __name__ == "__main__":
    main()
