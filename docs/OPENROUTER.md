# OpenRouter transcription comparison

The adapter supports models served by OpenRouter's dedicated
[`/api/v1/audio/transcriptions` endpoint](https://openrouter.ai/docs/guides/overview/multimodal/stt).
The public [transcription catalog](https://openrouter.ai/api/v1/models?output_modalities=transcription)
returned **24 model IDs on 27 September 2026**. This includes every model in the
[speech-to-text collection](https://openrouter.ai/collections/speech-to-text-models) at discovery.
It does not include general audio-chat models served through Chat Completions.

**Status: implementation tested offline; live evaluation is underway.** Hosted-model
results have not yet been published. Availability depends on the account's privacy and
provider settings; check the authenticated `/api/v1/models/user?output_modalities=transcription`
catalog as well as the public catalog before running.
The original local-model results remain archived; the current local pass uses this exact
94-clip manifest.

## Cost before running

| One trial per model | Audio per model | Requests across 24 models | Most expensive model: Chirp 3 | All 24 models |
| --- | --- | --- | --- | --- |
| Recommended screening pass | 4 minutes | 96 | $0.064 | **$0.34** |
| Full audio duration, in 60-second windows | 92.2784 minutes | 2,256 | $1.48 | **$7.85** |

The full count is 94 clips from these four meetings when prepared in 60-second windows.
The short pass uses the interval 60–120 seconds from each meeting, selected before seeing
model output. One trial, sequential requests, no automatic retries, no warmup calls, and
no repeated sampling. Four minutes is a screening test, not a reliable accuracy ranking.
A planning allowance of **$1 for screening** or **$10 for the full pass** provides some
headroom; neither is an API-enforced spending guarantee.

These estimates use current base prices and exclude tax, credit-purchase fees, rounding,
provider minimum charges, optional feature surcharges, and failed requests that are billed.
The current local comparison uses the exact 94 clips, audio hashes, and references from
this API benchmark. The historical 240-second local results are excluded from consolidation.

The [saved catalog](../results/openrouter-cost-estimate-2026-09-27/catalog.json),
[per-model cost table](../results/openrouter-cost-estimate-2026-09-27/README.md),
[short estimate](../results/openrouter-cost-estimate-2026-09-27/short.json), and
[full estimate](../results/openrouter-cost-estimate-2026-09-27/full.json) contain the inputs,
per-model calculations, and retrieval time. They contain no inference measurements.

The catalog's `pricing.prompt` is not always a price per token. Duration models use a
per-second rate, while MAI models use a per-hour rate. We review billing units explicitly
against the collection and model pages. Newly discovered models remain unpriced until
their units are reviewed; the estimator does not silently assume a unit.

GPT-4o Transcribe and Mini use OpenAI's published approximate blended rates of
[$0.006 and $0.003 per minute](https://developers.openai.com/api/docs/pricing), with a
price-consistency check against the OpenRouter catalog. Gemini 3.5 Transcribe uses its
model-specific [Google pricing assumptions](https://ai.google.dev/gemini-api/docs/pricing):
25 audio tokens/second and 175 output tokens/minute. At OpenRouter's $2/$12 per million
input/output tokens, this is $0.0051/minute. Actual transcript lengths vary. The generic
Gemini audio-understanding token rate is not substituted for the transcription model's rate.

## What the adapter measures

The adapter sends mono 16 kHz PCM16 WAV, a model ID, and the recording's language code.
It never sends reference transcripts, speaker counts, names, or vocabulary hints. API keys
come only from `OPENROUTER_API_KEY` at runtime; keys are not accepted in model configuration.
No new package dependency is required.

The initial configuration requests plain JSON from most models. For MAI-Transcribe 2 and
Deepgram Nova-3 it requests verbose output and enables the documented diarization option
in the same request. This tests their speaker capability without an extra inference pass.
Other models may advertise diarization or timestamps upstream, but their OpenRouter option
shapes have not been verified here. Missing features in this first configuration are not
evidence that the underlying model cannot provide them.

Transcripts with native speaker labels can receive speaker-attributed word error and
diarization scores. Models returning only text receive chronological WER; cpWER, tcpWER,
DER, and predicted speaker count are unavailable. Timestamps alone do not imply diarization.
Chronological WER remains imperfect on overlapping meeting speech, where word order is
ambiguous. Inline speaker markers or sound annotations in plain-text responses require
model-specific review before treating their WER as a clean lexical comparison.

Wall time includes local WAV encoding, upload, queueing, and the remote response. It is
client-observed latency, not provider GPU execution time. Local GPU memory is unavailable
and must not be compared with the 4090 memory measurements. Native response usage and a
generation ID, when provided, are saved with each prediction. `api_reported_cost_usd` sums
reported costs from valid completed records; `api_cost_complete` is false when any record
failed or omitted cost. Failed requests may still be billed: consult OpenRouter activity
for the final charge rather than assuming an unreported cost is zero.

The endpoint does not honor provider routing preferences such as `order` or `only`.
We record the requested model but do not claim a pinned serving backend. The model named
Nemotron Streaming is called through the synchronous transcription endpoint here; this
does not measure streaming latency.

## Reproduce the estimate

From the CPU environment described in [Running](RUNNING.md):

```bash
uv run python scripts/estimate_openrouter.py --output runs/openrouter-estimate
```

This public GET request requires no key and makes no inference calls. It writes the
catalog, estimate, and a four-request-per-model configuration. To reproduce the saved
full-duration estimate offline:

```bash
uv run python scripts/estimate_openrouter.py \
  --catalog results/openrouter-cost-estimate-2026-09-27/catalog.json \
  --seconds 5536.704 --requests-per-model 94 \
  --output runs/openrouter-full-estimate
```

## Prepare the screening manifest

Use the existing downloaded AMI source, or fetch it using [Running](RUNNING.md). Prepare
60-second windows in a new directory:

```bash
uv run speaker-bench prepare-ami --source data/ami-source \
  --destination data/ami-openrouter \
  --meetings ES2004a IS1009a TS3003a EN2002a --window-seconds 60
uv run python - <<'PY'
import json
from pathlib import Path
p = Path('data/ami-openrouter')
rows = [json.loads(line) for line in (p / 'manifest.jsonl').read_text().splitlines()]
selected = [row for row in rows if row['source']['offset_seconds'] == 60]
assert len(selected) == 4
with (p / 'screening.jsonl').open('x') as stream:
    for row in selected:
        stream.write(json.dumps(row) + '\n')
PY
uv run speaker-bench plan --config configs/openrouter.json \
  --manifest data/ami-openrouter/screening.jsonl
```

References use the existing word-midpoint boundary rule and clipped activity intervals.
All models receive exactly the same audio. No automatic splitting inside an adapter can
silently change context or speaker identity scope.

## Run only after reviewing costs

Provide `OPENROUTER_API_KEY` through your runtime's secret injection, then explicitly set
`OPENROUTER_ALLOW_PAID_REQUESTS=1`. Do not put a key in commands, source files, or a JSON
configuration. The supplied configuration caps each model at four requests and 240 seconds
total audio; the controller validates the complete manifest before sending any requests.
`--parallel-models N` runs up to N independent model workers at once (default: 1).
Each worker still transcribes one clip at a time. This does not duplicate requests.

```bash
OPENROUTER_ALLOW_PAID_REQUESTS=1 uv run speaker-bench run \
  --config configs/openrouter.json \
  --manifest data/ami-openrouter/screening.jsonl \
  --output runs/openrouter-screening
```

For the full pass on all 24 models, use the full 60-second-window manifest and the
94-request configuration. This permits at most 24 concurrent API requests:

```bash
OPENROUTER_ALLOW_PAID_REQUESTS=1 uv run speaker-bench run \
  --config configs/openrouter-full.json \
  --manifest data/ami-openrouter/manifest.jsonl \
  --parallel-models 24 --output runs/openrouter-full
```

The full configuration permits 5,536.704 seconds per model, one trial only. Parallelism
is recorded in the run plan. Shared provider limits can still cause failures; retries
remain disabled. Concurrent execution also works for local adapters, but sharing a GPU
changes latency and memory contention, so use the default serial mode for controlled
local-model performance measurements.

The output directory must be new. An HTTP error or timeout stops that model's worker and
leaves the other models eligible to run. Remaining clips for the failed model are reported
as not run. No failed request is retried automatically; its server-side outcome or cost may
be unknown. The controller reports failures instead of dropping them from the denominator.
HTTP status is retained without recording an arbitrary upstream error body.

## Chirp transport recovery

The first Chirp recovery stopped on one failing clip before attempting the remaining
clips. A subsequent isolated recovery completed 28 of the 29 missing clips; one request
took 111 seconds, while the remaining failing request returned HTTP 504 after 181 seconds.
Longer client timeouts cannot repair an upstream gateway timeout. Failed requests now
record elapsed time and distinguish HTTP errors, client timeouts, and other transport
errors without retaining arbitrary error messages.

The OpenRouter adapter accepts `audio_format: "flac"` as a lossless alternative to its
default WAV transport. Both encodings preserve the same mono 16 kHz PCM16 samples;
regression tests decode the submitted payload and check sample equality. This option
does not trim, resample, or otherwise change the benchmark audio.

FLAC also timed out on the remaining clip. Two nonoverlapping 30-second WAV requests
completed it, giving **94/94 scored clips and 38.68% WER**. The
[published results](../results/openrouter-chirp-2026-09-27/README.md) include all predictions,
failure history, scores, and verification. This is explicitly a recovery variant:
one clip has shorter context than the standard 60-second protocol. The cost of the
additional successful recovery requests was approximately **$0.46**.

## Deepgram timestamp handling

A diagnostic replay reproduced a Deepgram word ending at 61.314938 seconds in a
60-second clip. The original strict parser rejected the entire transcript. For
`deepgram/nova-3`, intervals crossing an audio boundary now use their intersection with
`[0, audio duration]`. Word text and speaker labels remain unchanged. Original timestamps
remain in `raw_transcript`, each adjustment is recorded in `timestamp_adjustments`, and
affected predictions have `timing: native_clipped`. Positive intervals entirely after
the audio retain their reported timestamps and speaker labels, with their indices
recorded in `intervals_after_audio`. WER, cpWER, and tcpWER retain those words, including
any insertion errors; DER uses only the actual audio duration as its evaluation region.
This avoids deleting words or inventing timestamps to satisfy validation. Intervals
entirely before zero, reversed or zero-length intervals, and non-finite times remain
invalid. The timestamp policy is applied without consulting reference transcripts.

Saved responses can be reparsed without another paid request. The destination must
not exist; original predictions remain untouched:

```bash
uv run python scripts/reparse_deepgram_run.py \
  runs/openrouter-deepgram runs/openrouter-deepgram-corrected
```

The corrected 94-clip run has WER **40.60%**. Five responses lack speaker labels,
so full-run cpWER, tcpWER, and DER remain unavailable. On the **89 clips with speaker
labels**, weighted cpWER is **66.68%**, tcpWER **67.98%**, and DER **56.38%**.
These subset scores must not be compared directly against other models' full-run scores.
The previously rejected boundary clip now contributes to all four metrics.

Rejected JSON transcripts now retain allowlisted text, words, segments, and usage with
a stable validation reason. Unknown fields, server error bodies, and request headers
are excluded; the active API key is redacted if echoed in a retained field. This allows
future parser corrections to use saved responses without new inference. Malformed JSON
is recorded as `invalid_json` without retaining an arbitrary body.

The full Deepgram-only rerun uses the same audio and inference options:

```bash
OPENROUTER_ALLOW_PAID_REQUESTS=1 uv run speaker-bench run \
  --config configs/openrouter-deepgram.json \
  --manifest data/ami-openrouter/manifest.jsonl \
  --output runs/openrouter-deepgram
```

Its estimated cost is about $0.40 for 94 requests. Keep it separate from the first pass;
do not select the better transcript from repeated calls. The initial DeepInfra failures
remain recorded as incomplete evaluations, without an additional rerun.

The request/audio limits bound the workload, **not dollars**. The key's provider-enforced
credit limit is the spending boundary. A 90-second client timeout is configured per request;
OpenRouter documents a shorter upstream processing timeout. A 60-second clip can still fail
if its provider is slow. Check failed request IDs and actual charges before authorizing a
selective retry. The completed evaluation and four partial DeepInfra runs appear in the
[single consolidated table](../results/openrouter-2026-09-27/README.md), together with the
local systems rerun on the same manifest. It includes per-metric medals; metric coverage
and recovery status are shown directly in the table.
