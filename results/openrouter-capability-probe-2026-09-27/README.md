# Live output validation

One shared 60-second AMI clip was sent once to each of the 20 active routes. Two additional
Chirp requests investigated its structured-output rejection: one captured sanitized error
diagnostics; one removed all provider-specific options and requested only word timestamps.
Neither was a full benchmark rerun. The four previously excluded DeepInfra routes were
not called.

Ten routes produced valid transcripts after offline parser corrections. The
[summary](summary.json) records metric availability and failures per route. These scores
are format-validation evidence, not an accuracy ranking from one clip.

- MAI 2 and Deepgram returned words, speakers, and timing for all five metrics.
- Fish Pro returned inline speaker labels and separate word timings. It supports cpWER
  here, but no verified association between those timings and speaker turns.
- AssemblyAI, Fish standard, Nemotron, Parakeet, Whisper Large v3, and Whisper Turbo returned
  timing without speaker labels, supporting WER.
- MAI 1.5 returned text only, as expected for the tested route.
- Nine routes were missing from the authenticated model list. Six returned 404; three
  rejected the request with 400. Access settings and those request shapes remain unresolved.
- Chirp returned 400 for verbose output, including a standard timestamp-only request without
  provider options. This establishes a rejection on the tested OpenRouter route, not an
  absence of diarization in Google's own API.

The account's observed credit change was approximately **$0.0314663**, matching reported
costs for the ten returned transcripts. Billing can settle asynchronously; failed requests
retain their unknown billing status rather than being assigned a fabricated zero cost.
The full run has **not** started.

`original/` preserves original scored responses, including four parser rejections.
`reparsed/` preserves their offline corrections. Native zero-duration word timestamps are
retained as points: their text counts in word metrics, but they add no speech duration.
Intervals crossing the audio boundary are intersected with it; late words remain in word
scoring, while DER is restricted to the clip. Raw timestamps and adjustment metadata remain
available. [Correction provenance](parser-corrections.json) identifies each source file.
`diagnostics/` records Chirp's HTTP status and fixed diagnostic keywords without upstream
messages, credentials, or headers.

The [full-run validation check](../../scripts/validate_openrouter_probe.py) rejected the
incomplete probe and did not produce a runnable full configuration. See the
[capability audit](../../docs/OPENROUTER_CAPABILITY_AUDIT.md) and
[data attribution](../../DATA_LICENSE.md). No audio is redistributed.
