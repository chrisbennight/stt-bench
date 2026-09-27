# Hosted output audit and recovery

The first hosted run did not validate the speaker and timestamp options for every route.
Its missing metric cells must not be interpreted as model capability findings. The corrected
configuration was generated only after inspecting live output for every included route.
The original results remain archived so the extra requests and corrections are visible.

The [follow-up request and raw-response review](../results/openrouter-diarization-review-2026-09-27/README.md)
checks all 20 output sets and ten small live diagnostics. It finds no current parser loss
for the missing labels, but does not establish failure of the underlying diarization
models. Gemini, Voxtral, and Meta still return no labels with the tested native options.

The [machine-readable audit](../research/openrouter-capabilities.json) covers all 24 model
IDs, provider tags, source links, observed output counts, proposed request options, and
base-price estimates. All queried endpoints returned empty `supported_parameters` lists;
those lists are not evidence that timestamps or speaker labels are unsupported.

The [first probe](../results/openrouter-capability-probe-2026-09-27/README.md) validated ten
routes, while privacy settings blocked nine others and Chirp rejected verbose output.
After those settings changed, the authenticated catalog exposed all 24 IDs. Bounded
follow-up probes tested the nine routes, rejected-format alternatives, and actual returned
field names. There were **42 probe requests in total**, including failed attempts.
Twenty configurations passed the output check; four DeepInfra routes remained excluded.
The [corrected results](../results/openrouter-validated-2026-09-27/README.md) contain 19
complete hosted runs and Chirp's 93 successful clips plus one scored failure. Native
speaker scores use the same 77 labelled clips across every compared system. The
[request ledger](../results/openrouter-validated-2026-09-27/costs.json) records every probe
and full-pass attempt, including the failed Chirp retry.

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
- Grok returned timed speaker labels with its provider-specific diarization option.
  An earlier response using the same options contained empty text; both observations
  remain in the probe record.
- Gemini and Mistral document speaker options upstream. With the tested OpenRouter
  options they returned timed words and segments but no speaker fields. Instrumentation
  retained the actual response field names, confirming that the parser did not discard
  a hidden speaker field. Gemini's alternative option shape also failed to produce labels.
  This does not establish that every option or serving route lacks the capability.
- Meta rejected verbose output and returned text only in JSON mode even with its
  `DIARIZATION` provider option. Its upstream speaker features remain unverified here.
- Chirp rejected verbose output both with and without provider options; JSON text output
  passed. Its upstream diarization capability is not measured by this integration.
- AssemblyAI's Sync route documents word timestamps. Its asynchronous API's speaker option
  must not be assumed to work on the Sync route.
- Whisper, Parakeet, Nemotron, AssemblyAI Sync, and Fish standard returned timing without
  speaker labels. Those responses support WER and coverage, but not speaker error metrics.
- Hosted Qwen Flash rejected verbose output and passed in JSON mode. Qwen's local aligner
  is a separate component, not a feature added to this hosted route.
- MAI 1.5 and GPT-4o transcription routes are classified as text-only for this validation.
  GPT Transcribe also rejected verbose output and passed in JSON mode.

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

## Validation before the corrected full pass

The [validated probe configuration](../configs/openrouter-capability-probe.json) makes at
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

After assembling the reviewed probe observations, validate saved output before generating
the [full-run configuration](../configs/openrouter-validated-full.json):

```bash
uv run python scripts/validate_openrouter_probe.py \
  results/openrouter-validated-2026-09-27/validated-probe \
  --output runs/verified-openrouter-full.json
```

This command makes no API calls. It checks exact candidate options, clip identity, audio
hash, references, successful response status, and actual score availability. A response
that silently omits a requested capability blocks generation of the full configuration.
Resolve each failed or missing capability with evidence; do not relax requirements just
to obtain a passing check. The audit retains original requested metrics, observed metrics,
tested options, and explicit limitations for routes that returned less than requested.
Passing this check means the selected output shape was observed, not that all upstream
features were successfully exposed. A single probe establishes request/output shape, not accuracy
or reliability across the full dataset. Inspect all subsequent clip outputs too.

Inspect the old outputs and recover Fish Pro locally without changing published raw data:

```bash
uv run python scripts/audit_openrouter_outputs.py --output runs/openrouter-output-audit
```

The resulting audit and corrected Fish scores are separate from the original run. A new
full run must retain request costs and failed/unknown billing states; neither discarded
requests nor validation requests may be silently omitted from the session cost report.
