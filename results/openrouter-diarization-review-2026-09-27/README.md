# Review of missing hosted speaker labels

**The missing labels are already absent from the tested OpenRouter responses, before
our parser runs. This does not establish that the underlying models failed at diarization.**
The evidence supports a limitation somewhere in request forwarding, upstream behavior,
or response conversion. We cannot locate that internal boundary without provider-side
traces or a direct-provider comparison.

**What was checked**

- **All 20 tested routes:** inspected saved transcript fields and the recorded field names
  from the 94-clip pass. These historical transcript copies are filtered projections,
  not complete HTTP response bodies; field-name instrumentation provides additional evidence.
- **Provider tags:** queried the public endpoints API and matched the option keys to its
  actual tags. Mistral has both `mistral` and `mistral/eu`; the request covered both.
  Its endpoint name identifies the 2602 version, not the older 2507 transcription model.
- **Ten diagnostic requests:** used the same 60-second AMI clip, `ES2004a-4800000`,
  containing four reference speakers. Captured complete successful JSON responses before
  applying our transcript parser. Audio and reference speaker identities were not modified
  or supplied as hints. No successful benchmark recording was replaced with a diagnostic.
- **Actual serialization:** intercepted the benchmark adapter with a fake transport and
  compared its request options with the four enabled diagnostic requests. All matched.
- **Parser replay:** parsed the new responses offline and inspected every nested JSON key
  for speaker, diarization, turn, or utterance fields. No unhandled speaker fields were found.

**Findings by route**

- **Gemini 3.5 Transcribe:** `google-ai-studio` and the native
  `generation_config.transcription_config.mode.diarization_mode="speaker"` shape match
  the documented interfaces. The response contains timed words and segments with no
  speaker fields. Replacing the mode with an invalid value still returned HTTP 200 and
  identical text, words, and segments. This is consistent with an ignored option, but does
  not prove exactly where it was ignored. OpenRouter's model page advertises diarization.
  [OpenRouter model](https://openrouter.ai/google/gemini-3.5-transcribe) ·
  [Google configuration](https://ai.google.dev/gemini-api/docs/transcribe)
- **Voxtral Mini Transcribe:** `diarize=true` under both serving provider tags returned
  timed output without labels. An invalid object in place of the boolean produced the
  same text, words, and segments. Mistral documents a language/timestamp incompatibility;
  removing the language hint, then requesting only segment timestamps, also returned no
  labels. Those configuration changes did not repair the result.
  [Native parameters](https://docs.mistral.ai/api/endpoint/audio/transcriptions) ·
  [Language/timestamp restriction](https://docs.mistral.ai/studio/audio/speech_to_text/offline_transcription)
- **Meta Muse Voice:** valid `mode="DIARIZATION"` returned only `text` and `usage`.
  An invalid mode returned HTTP 400, so the mode is recognized somewhere on the remote
  request path. Meta documents speaker-labelled `turns` in its native response, but there
  were no such fields in the OpenRouter body for our parser to recover. The earlier
  verbose-output probe was rejected. The exact point where speaker output is lost remains
  unconfirmed. [Meta's native request and response](https://dev.meta.ai/docs/speech-to-text)
- **Deepgram, as a control:** the enabled request returned native `speaker` fields and
  our parser retained them. The invalid-object request returned HTTP 200 without labels.
  This also shows why accepting an invalid option is not sufficient proof that all options
  are ignored. The control returned only one labelled speaker; it checks transport and
  parsing, not diarization accuracy on the four-speaker clip.
- **Fish Pro:** its native contract provides inline speaker markers and separate unlabelled
  timing. Our corrected parser preserves the markers as labels. Its missing tcpWER/DER
  is an alignment limitation documented by Fish, not evidence that it failed to label text.
  [Fish output semantics](https://docs.fish.audio/features/speech-to-text)
- **Other routes:** absence of speaker labels is not, by itself, a failed diarization test.
  For example, AssemblyAI's Sync response documents words and timing, while other endpoints
  have different features. Chirp's tested OpenRouter route rejects verbose output; that
  does not disprove Google's native diarization support. No fresh request was needed for
  these observations. [AssemblyAI Sync](https://www.assemblyai.com/docs/sync-stt/getting-started/transcribe-a-short-audio-file) ·
  [Google Chirp](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3)

**What this means for the benchmark**

- No new parser or serialization defect was established by this review. Earlier parser
  defects, including Fish markers and boundary timestamps, remain documented separately.
- The main table measures **output delivered by these OpenRouter configurations**, not
  every feature of the upstream models or their direct APIs. Its numbers are unchanged.
- A missing metric should be read as **not available in the tested output**, not as a
  measured high diarization error or proof of general model incapability.
- OpenRouter documents that provider integrations may silently drop unsupported options.
  A successful HTTP response therefore does not establish that a requested feature ran.
  [OpenRouter forwarding and output rules](https://openrouter.ai/docs/guides/overview/multimodal/stt)

**Evidence and cost**

- [Complete successful diagnostic JSON responses and request options](responses/)
- [All-route field audit, parser checks, and diagnostic summary](summary.json)
- [Serving provider tags and endpoint names](endpoints.json)
- [Cost and balance observations](costs.json)

The ten requests reported **$0.029404** in response charges. One rejected request did not
report usage; its individual billing is unknown. These diagnostics are separate from the
benchmark and its 42 earlier capability probes. No full benchmark was rerun.

Reproduce the inspection without network requests, credentials, or source audio:

```bash
uv run python scripts/audit_openrouter_diarization.py
```

The next decisive evidence would be a provider trace for the recorded generation IDs,
showing forwarded options and native upstream speaker fields, or a same-audio request to
the direct provider API. No support message was sent and no direct-provider credential
was used in this review.

AMI-derived transcripts retain their [data attribution and license](../../DATA_LICENSE.md).
