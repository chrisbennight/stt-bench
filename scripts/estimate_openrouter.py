"""Discover STT models and estimate costs without credentials or inference requests."""

import argparse
import json
from pathlib import Path

from speaker_benchmark.openrouter_catalog import estimate, fetch_catalog, make_config
from speaker_benchmark.schema import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, help="Read a saved catalog instead of the API")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--requests-per-model", type=int, default=4)
    args = parser.parse_args()
    catalog = read_json(args.catalog) if args.catalog else fetch_catalog()
    report = estimate(catalog, args.seconds)
    if args.requests_per_model <= 0:
        parser.error("requests-per-model must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "catalog.json", catalog)
    write_json(args.output / "estimate.json", report)
    write_json(
        args.output / "config.json",
        make_config(
            catalog,
            args.requests_per_model,
            args.seconds,
        ),
    )
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise SystemExit("Estimate incomplete: review unpriced models before paid inference")


if __name__ == "__main__":
    main()
