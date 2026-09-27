# Speaker benchmark

A local Python harness for comparing:

1. MOSS-Transcribe-Diarize 0.9B
2. VibeVoice-ASR offline
3. Qwen3-ASR-1.7B + Qwen forced alignment + pyannote Community-1
4. Qwen3-ASR-1.7B + Qwen forced alignment + Nemotron 3 Diarization
5. VibeVoice-ASR-Streaming-7B

The controller uses **MeetEval** for transcription metrics and **pyannote.metrics** for DER.
Each system runs in its own process and can use a different Python environment. The two
VibeVoice adapters share an environment but load different checkpoints in separate processes.
The adapters use a small common prediction format and existing scoring libraries.

## Status

The current consolidated evaluation uses **94 windows of at most 60 seconds**, identical
to OpenRouter, with one pass per local model. Reproduce it with existing environments:

```bash
uv run speaker-bench run --config configs/five-systems-openrouter.json \
  --manifest data/ami-openrouter/manifest.jsonl --output runs/local94 --parallel-models 1
```

Use the [OpenRouter dataset recipe](OPENROUTER.md) to prepare the shared manifest.
Streaming processes the supplied audio without real-time waits. The following description
and 240-second recipe document the historical local-only evaluation.

All five systems completed the 25-window comparison on the RTX 4090. There were
124 valid outputs and one invalid offline VibeVoice generation. All 125 final score records
have been independently recomputed with exact matches. See [the measured results](RESULTS.md).
The final comparison uses stable per-record random seeds because VibeVoice samples its
acoustic features even with greedy text decoding. See [the protocol](PROTOCOL.md).
Offline VibeVoice uses a separately pinned Qwen2.5-7B tokenizer. Both VibeVoice adapters
retain square-bracket sound annotations in raw output but exclude them from speech scoring.
Ordinary speech without a speaker label is retained under `unassigned`, including when
streaming speech arrives before its first speaker label.
Streaming output is parsed across chunk boundaries so split words, speaker markers, and
sound annotations retain their meaning. The saved raw run can be reparsed without inference
using `scripts/reparse_streaming_run.py`; see [the protocol](PROTOCOL.md) for provenance.
Nemotron requires the pinned official NeMo Speech source: the published 3.0.0 package does
not implement the rotary attention configuration in these weights.

## Quick CPU verification

```bash
uv sync --locked
uv run pytest -q
uv run speaker-bench smoke --output /tmp/speaker-smoke.json
```

The smoke command uses an intentionally incorrect synthetic transcript. Its numbers are
scorer checks, not model benchmark results.

## Benchmark tracks

| Track | Data | Purpose |
|---|---|---|
| Initial check | AMI ES2004a, Array1-01 microphone | One meeting, five windows, approximately 17.5 minutes total |
| Measured comparison | AMI ES2004a, IS1009a, TS3003a, EN2002a; 25 non-overlapping 240-second windows | Four meeting groups, 92.28 minutes, identical input clips for all five systems |
| Common comparison | All 16 meetings in the pinned AMI test list, non-overlapping 240-second windows | Identical input clips for all five systems; includes overlap and distant microphones |
| Long recordings | Complete AMI test meetings, offline systems only | Speaker consistency across long gaps, missing endings, and resource growth |
| Additional domain | NOTSOFAR-1 held-out single-channel meeting recordings | Different rooms and 4–8 participants; use the SegLST importer |

The 16-meeting, long-recording, and additional-domain tracks are supported recipes, not
completed measurements in this result bundle.

The windowed tracks reset speaker IDs at each window. They do not measure hour-long identity
consistency. Keep their results separate from the full-recording track. The streaming model
supports at most 480 seconds per session. MOSS and offline VibeVoice are limited to 5400 and
3600 seconds respectively; unsupported recordings remain visible in reports. Use a four-system
configuration without streaming for long recordings, and inspect any other duration exclusions.

### AMI preparation

```bash
uv run speaker-bench fetch-ami --destination data/ami-source --meetings ES2004a
uv run speaker-bench prepare-ami --source data/ami-source \
  --destination data/ami-common --meetings ES2004a --window-seconds 240
```

Preparation refuses to overwrite a destination. For the full test set, provide every ID in
`speaker_benchmark.datasets.AMI_TEST` to both commands. For full recordings, use a new destination
and `--window-seconds 0`. These are local benchmark protocols, not an official leaderboard submission.

The measured four-meeting comparison uses:

```bash
uv run speaker-bench fetch-ami --destination data/ami-source \
  --meetings ES2004a IS1009a TS3003a EN2002a
uv run speaker-bench prepare-ami --source data/ami-source --destination data/ami-four \
  --meetings ES2004a IS1009a TS3003a EN2002a --window-seconds 240
uv run speaker-bench run --config configs/five-systems.json \
  --manifest data/ami-four/manifest.jsonl --output runs/ami-four-seeded
```

Prepare the inference environments and model snapshots below before the run command.
Choose a new output directory for a repeated trial; existing measurements are preserved.

The recipe uses official AMI manual word annotations and pinned pyannote `only_words` RTTM
references. Punctuation-only and untimed XML elements are excluded. Words crossing a window
boundary are assigned by their midpoint exactly once and clipped to the window; speaker
activity is clipped into both windows when appropriate. Source hashes are recorded. AMI's
CC BY 4.0 attribution and source links are in [REFERENCES.md](REFERENCES.md).

### Other datasets and your own recordings

```bash
uv run speaker-bench import-seglst --references references.json \
  --audio-map audio-map.json --destination manifest.jsonl
```

`references.json` is MeetEval/CHiME SegLST: a JSON list of objects with `session_id`, `speaker`,
`words`, `start_time`, and `end_time`. `audio-map.json` maps each session to an audio path,
language, and optional RTTM file:

```json
{"meeting1": {"audio": "meeting1.wav", "language": "English", "rttm": "meeting1.rttm"}}
```

Use official NOTSOFAR conversion/export tools to produce SegLST, then import it here. This
project does not silently run the upstream baseline or download the entire NOTSOFAR corpus.
Without RTTM, DER uses transcript intervals as speaker activity; record that difference when
comparing datasets. Files must be mono. Pick the same microphone for every system; do not mix
headset references, beamformed audio, and distant microphones in one aggregate.

## Inference setup

Model runtimes are deliberately separate: Qwen currently pins Transformers 4.57.6, while
MOSS requires Transformers 5.x. Use Python 3.12 and a compatible NVIDIA driver/PyTorch build.
The scripts use standard PyPI packages and pinned official Git sources, without privileged
containers. They create local environments and refuse to replace existing ones.

```bash
python scripts/setup_runtime.py moss
python scripts/setup_runtime.py vibevoice
python scripts/setup_runtime.py qwen_pyannote
python scripts/setup_runtime.py qwen_nemotron
```

These install substantial GPU dependencies. The core `uv.lock` covers the CPU harness;
GPU dependency resolutions are recorded separately in each environment's
`resolved-packages.json`. GPU transitive dependencies are not yet supplied as tested lockfiles.

Download weights explicitly before measurements:

```bash
.venv-moss/bin/python scripts/fetch_models.py moss
.venv-vibevoice/bin/python scripts/fetch_models.py vibevoice vibevoice_tokenizer vibevoice_streaming
.venv-qwen_pyannote/bin/python scripts/fetch_models.py qwen aligner pyannote
.venv-qwen_nemotron/bin/python scripts/fetch_models.py nemotron
```

`model-revisions.json` pins the requested checkpoints. Scripts use the normal authenticated
Hugging Face client and its existing credential cache; no credentials belong in configs or
command arguments. Community-1 requires accepting its provider terms. The pyannote preparation
step also populates nested model dependencies. The worker sets Hugging Face/Transformers offline
mode, so a missing dependency fails instead of downloading during a measured run.

MOSS loads custom Python model code. Pinned revisions provide traceability, not a security
audit of these new model repositories. Review that code before running on a host with secrets.

## Run and inspect

```bash
uv run speaker-bench plan --config configs/five-systems.json \
  --manifest data/ami-common/manifest.jsonl
uv run speaker-bench run --config configs/five-systems.json \
  --manifest data/ami-common/manifest.jsonl --output runs/ami-first
```

Start with a manifest containing one short recording on the GPU host. Inspect its actual
transcript and speaker turns before expanding to the full suite. The plan reports missing
snapshots without importing or loading GPU models. The run command uses a new output directory
and exits with status 2 if any recording fails or is unsupported. It does not resume or reuse
old predictions automatically.

Outputs include `summary.csv`, `summary.json`, per-recording scores, predictions, streaming
events, package versions, model path/config fingerprints, dataset hashes, and worker outcomes.
Model download time is excluded; model load time is reported separately. First-recording
inference is cold: there is no hidden warmup. Run repeated trials into new directories to study
variance. Avoid other GPU workloads during measurement.

```bash
uv run speaker-bench report runs/ami-first
```

The report can be regenerated after an interrupted worker. Missing predictions count as
empty hypotheses for transcription scoring, and completion/failure counts remain explicit.
Runtime reports include both successful recordings and all attempted recordings, including
completed generations with invalid output. Both denominators are labelled. Never compare
speed without checking completion counts. Upstream console output is discarded to avoid
persisting credentials or transcripts in logs; structured predictions retain the transcript,
and unexpected failures retain their exception class. Invalid structured outputs retain their
raw text and generation-limit metadata. Handle output directories as transcript data.

To verify a finished result bundle without the audio files or GPU models:

```bash
uv run python scripts/verify_scores.py runs/ami-four-seeded --output score-verification.json
```

This checks every model/recording, verifies that scored outputs match saved predictions,
and recomputes the scores from the saved references. It rejects unattempted recordings.

## Metrics and interpretation

- **WER:** chronological, speaker-agnostic word errors. Secondary for overlap, where there is
  no uniquely correct interleaving of speakers' words.
- **cpWER:** MeetEval speaker-permutation-invariant transcription errors; primary accuracy measure.
- **tcpWER:** MeetEval timing-constrained errors, default 5-second collar. Segment-level
  timestamps use MeetEval's documented pseudo-word timing, not invented exact word boundaries.
- **DER:** pyannote.metrics, zero forgiveness collar by default, overlapping speech included,
  evaluated across the entire recording. Pipeline activity intervals are retained independently
  of recognized words, preserving simultaneous speakers. Optional 0.25-second collar must be
  a separately identified run.
- **Runtime/memory:** wall time, RTF, worker process-tree sampled RSS, and PyTorch peak allocated
  and reserved CUDA bytes. CUDA figures exclude non-PyTorch allocations; RSS is not VRAM.
- **Streaming:** first nonempty attributed text and p95 chunk-end-to-emission lag under
  real-time audio pacing. These are chunk-level latency measures, not word-aligned latency.
  Per-chunk events and compute seconds are retained. Setting `realtime: false` measures
  unpaced throughput; latency aggregates are then unavailable.

VibeVoice streaming emits `Speaker k:` text without timestamps. **Native DER and tcpWER
are null for that adapter**, with an explicit reason. Adding a forced
aligner would create a different augmented system and should have its own configuration;
this harness does not mislabel chunk edges as word times.

English scoring uses Unicode normalization, case folding, punctuation removal, and whitespace
tokens while keeping apostrophes, fillers, repetitions and spoken numbers. There is no LLM
cleanup or oracle speaker count. This initial recipe is English-only: multilingual/CJK
comparisons need an explicit language-appropriate CER/tokenization protocol before use.
Corpus aggregates use total errors / total reference words (or durations), not averages of
recording percentages. The reporting code does not silently omit failed files.

## Adapter plugins

Implement `load()` and `transcribe(audio_path, duration, language) -> Prediction`, with an
optional `max_duration` in seconds. Only audio metadata crosses the worker boundary; reference
transcripts, speaker counts, and speaker names are never passed to the adapter. Worker isolation
prevents accidental API leakage, but is not a filesystem security sandbox for untrusted plugins.

Register an installed package entry point:

```toml
[project.entry-points."speaker_benchmark.adapters"]
my_model = "my_package:MyAdapter"
```

The configuration can then select `adapter: "my_model"` and its Python executable. Options
use the allowlisted model/path/device/generation fields in `runner.py`; extend that schema
explicitly when adding new options. Return native timestamps or mark `timing="unavailable"`.
Use `activity` for separate overlap-aware diarization intervals. Empty valid output is distinct
from parser/model failure. Supplied built-ins are in `src/speaker_benchmark/adapters/`.
