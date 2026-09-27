# AMI / RTX 4090 result files

This directory records one five-system evaluation: 25 windows, four AMI test meeting groups,
5,536.704 seconds of audio, completed on 27 September 2026. See the
[main findings](../../README.md) and [full report](../../docs/RESULTS.md).

| Path | Contents |
| --- | --- |
| `corrected/` | Primary published predictions, per-window scores, references, summaries, and model/worker metadata |
| `original/` | The same measured trial before correcting a streaming chunk-boundary parsing issue |
| `comparison/` | Five-system JSON and CSV summaries, meeting/window tables, and supplementary checks |
| `provenance/packages/` | Recorded package versions for the controller and each model environment |
| `provenance/inference-source/` | Frozen source/configuration used during the measured inference run |
| `provenance/inference-source-sha256.json` | Hashes of that frozen source |
| `score-verification-*.json` | Independent recomputation: 125 checked scores per run, zero mismatches |
| `publication.json` | Description of filesystem-metadata changes made for publication |
| `checksums.json` | SHA-256 inventory of the published result files, excluding this checksum file |

All five workers attempted every window. Offline VibeVoice produced one invalid JSON
generation after repeating text to the token cap. Its raw output is in the corresponding
prediction file, and the failure is included in error and runtime totals. A benchmark command
exit status of 2 indicates that recorded failure; it does not mean windows were silently omitted.

The original streaming parser interpreted each chunk independently. One sound annotation
crossed chunk boundaries, so fragments incorrectly became speech. The corrected version parses
the exact concatenated text and preserves character-to-event positions. Only TS3003a-0 changed
in normalized speaker text or speech-event flags. The correction did not rerun a model or
change a seed, raw output, wall time, compute time, or emission timestamp. The
[reprocessing audit](corrected/reprocessing.json) records the transformation.

Original and corrected prediction files and per-model score files are byte-for-byte identical
to the respective measured/internal artifacts. Filesystem metadata was made portable for
publication: manifests point to `../../../data/ami-four/`, and archived plans/jobs use
`${BENCHMARK_ROOT}` and `${MODEL_CACHE}` in place of the original machine's checkout location. Manifest/source-plan
hashes were updated to identify the public copies. These archived plans are provenance, not
executable configurations; use `configs/five-systems.json` for a fresh run.

From the repository root:

```bash
uv sync --locked --python 3.12
uv run python scripts/verify_scores.py results/ami-4090-2026-09-27/corrected --output verification.json
uv run python scripts/verify_scores.py results/ami-4090-2026-09-27/original --output verification-original.json
```

Both commands work without audio, model weights, credentials, or a GPU. Output files must be
new paths. For new inference, follow [the running guide](../../docs/RUNNING.md).

The data attribution and separate license are in [DATA_LICENSE.md](../../DATA_LICENSE.md).
No model weights or source audio are redistributed.
