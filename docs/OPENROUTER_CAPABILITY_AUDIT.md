# Hosted output audit and recovery

The first hosted run did not validate the speaker and timestamp options for every route.
Its missing metric cells must not be interpreted as model capability findings. The current
comparison is provisional until corrected configurations have been tested and run.

The [machine-readable audit](../research/openrouter-capabilities.json) covers all 24 model
IDs, provider tags, source links, observed output counts, proposed request options, and
base-price estimates. All queried endpoints returned empty `supported_parameters` lists;
those lists are not evidence that timestamps or speaker labels are unsupported.

The [live probe results](../results/openrouter-capability-probe-2026-09-27/README.md) now
validate ten routes. Nine others are absent from the authenticated model list, and Chirp
rejects verbose output even without provider options. Total observed probe spend is about
$0.0315. The full-run check failed as intended; no full rerun has started.

## What the audit established

- MAI 2 and Deepgram already returned structured timing and speakers, although some clips
  lacked speaker labels. The other 22 saved routes have no structured timestamps to recover.
- Fish Pro has inline speaker markers in all 94 saved transcripts. The previous parser
  counted markers and event annotations as words. The corrected parser preserves raw text,
  removes bracketed non-speech annotations, and assigns text to the documented inline labels.
  An unfinished trailing annotation is excluded and recorded. Empty spoken output counts
  as deletions. This correction requires no API call.
  [Saved correction results](../results/openrouter-capability-audit-2026-09-27/README.md)
  give WER 31.54% and full-dataset cpWER 43.20%.
- Gemini, Grok, Meta, and Mistral document speaker options in their own APIs. Whether the
  exact OpenRouter route forwards each candidate option remains unverified.
- Chirp documents diarization, but its descriptions of synchronous support conflict.
  Its candidate configuration requires a successful output check, not an assumption.
- AssemblyAI's Sync route documents word timestamps. Its asynchronous API's speaker option
  must not be assumed to work on the Sync route.
- Whisper and Parakeet expose timing upstream. Qwen's local aligner is a separate component;
  the hosted Qwen route must be checked rather than credited with local pipeline features.
- MAI 1.5 and GPT-4o transcription routes are classified as text-only for this validation.
  GPT Transcribe and the remaining uncertain routes retain an explicit pending-probe status.

Sources: [OpenRouter request and forwarding rules](https://openrouter.ai/docs/guides/overview/multimodal/stt),
[model collection](https://openrouter.ai/collections/speech-to-text-models),
[Fish output semantics](https://docs.fish.audio/features/speech-to-text),
[Gemini configuration](https://ai.google.dev/gemini-api/docs/transcribe),
[Grok diarization](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text),
[Meta speaker turns](https://dev.meta.ai/docs/speech-to-text),
[Mistral transcription](https://docs.mistral.ai/api/endpoint/audio/transcriptions),
[Chirp support](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3), and
[AssemblyAI Sync timestamps](https://www.assemblyai.com/docs/sync-stt/getting-started/transcribe-a-short-audio-file).

Fish's timed segments may cross speaker turns and omit labels. The parser therefore does
not invent speaker timestamps from word order. Timed speech can support coverage while
cpWER uses inline speaker text; tcpWER and DER remain unavailable without aligned speakers.

## Validation before another full pass

The [candidate probe configuration](../configs/openrouter-capability-probe.json) makes at
most one request per active model on one shared 60-second clip. The four DeepInfra routes
previously excluded by the user remain excluded: Qwen 0.6B and 1.7B, Voxtral Mini 3B, and
Voxtral Small 24B. No retries are automatic, including after timeouts.

Current base estimates are **$0.0804 for the 20-model probe** and **$7.4222 for a full
20-model pass**. Including all 24 would be $0.0851 and $7.8513 respectively. Chirp is the
most expensive duration-priced route at approximately $1.4765 for the full audio.
Feature surcharges, billing rounding, token variability, and failed requests are not a
guaranteed zero. These estimates are not provider-enforced spending limits.

The probe clip and its audio hash and references are in the audit JSON. Credentials belong
in the runtime environment as `OPENROUTER_API_KEY`, never in configuration or results.
Use the existing runner with `OPENROUTER_ALLOW_PAID_REQUESTS=1` only after reviewing costs.
The controller must use the shared probe manifest and an unused run directory.

After the probe, validate saved output before generating any full-run configuration:

```bash
uv run python scripts/validate_openrouter_probe.py runs/openrouter-capability-probe \
  --output configs/openrouter-validated-full.json
```

This command makes no API calls. It checks exact candidate options, clip identity, audio
hash, references, successful response status, and actual score availability. A response
that silently omits a requested capability blocks generation of the full configuration.
Resolve each failed or missing capability with evidence; do not relax requirements just
to obtain a passing check. A single probe establishes request/output shape, not accuracy
or reliability across the full dataset. Inspect all subsequent clip outputs too.

Inspect the old outputs and recover Fish Pro locally without changing published raw data:

```bash
uv run python scripts/audit_openrouter_outputs.py --output runs/openrouter-output-audit
```

The resulting audit and corrected Fish scores are separate from the original run. A new
full run must retain request costs and failed/unknown billing states; neither discarded
requests nor validation requests may be silently omitted from the session cost report.
