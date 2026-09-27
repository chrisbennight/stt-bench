# Consolidated benchmark results

Sorted by **word error rate (WER), lower is better**, the general transcription metric
used for ranking by [Open ASR](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
Speaker attribution is additionally measured by cpWER and tcpWER, following
[MeetEval](https://arxiv.org/abs/2307.11394); DER measures speaker activity errors.
🥇🥈🥉 mark the three lowest measured values per column, except coverage (higher).
All systems use the same 94 clips and references: 92.28 minutes, up to 60 seconds
per clip. Speaker metrics use the same 77 labelled clips for every scored system.
Missing cells state the output limitation; these describe the tested route, not every
upstream capability. Four previously excluded DeepInfra routes receive no scores or medals.

| Model | WER ↓ | cpWER (77) ↓ | tcpWER (77) ↓ | DER (77) ↓ | Coverage ↑ | RTF ↓ | API $ ↓ | VRAM GiB ↓ | Clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAI Transcribe 2 | 🥇 26.10% | 🥈 35.13% | 🥈 35.97% | 58.76% | 55.04% | 0.145 | 0.1538 | — | 94/94 |
| MAI Transcribe 1.5 | 🥈 26.41% | No labels | No labels | No labels | No times | 0.148 | 0.5538 | — | 94/94 |
| MOSS 0.9B | 🥉 30.92% | 🥇 26.29% | 🥇 27.10% | 🥇 23.57% | 🥉 88.83% | 🥉 0.073 | — | 🥇 1.92 | 94/94 |
| Fish Transcribe 1 Pro | 31.97% | 40.28% | No speaker times | No speaker times | 53.84% | 0.157 | 0.5538 | — | 94/94 |
| NVIDIA Parakeet TDT v3 | 32.76% | No labels | No labels | No labels | 29.54% | 0.140 | 0.1384 | — | 94/94 |
| Qwen3 ASR Flash | 33.34% | No labels | No labels | No labels | No times | 0.164 | 0.1937 | — | 94/94 |
| Gemini 3.5 Transcribe | 33.62% | No labels | No labels | No labels | 60.46% | 0.166 | 0.2770 | — | 94/94 |
| Qwen3 + pyannote | 34.17% | 48.13% | 50.03% | 🥈 26.83% | 🥈 92.99% | 🥈 0.033 | — | 🥉 7.11 | 94/94 |
| Qwen3 + Nemotron | 34.17% | 🥉 39.54% | 🥉 40.76% | 🥉 29.95% | 77.97% | 🥇 0.032 | — | 🥈 6.13 | 94/94 |
| VibeVoice offline | 34.82% | 39.58% | 40.77% | 34.27% | 🥇 94.74% | 0.140 | — | 17.09 | 94/94 |
| Meta Muse Voice 1.0 | 35.05% | No labels | No labels | No labels | No times | 0.255 | 0.2768 | — | 94/94 |
| Voxtral Mini Transcribe | 36.19% | No labels | No labels | No labels | 50.64% | 0.171 | 0.2738 | — | 94/94 |
| OpenAI GPT Transcribe | 38.84% | No labels | No labels | No labels | No times | 0.154 | 0.4153 | — | 94/94 |
| Whisper 1 | 39.09% | No labels | No labels | No labels | 73.00% | 0.167 | 0.5538 | — | 94/94 |
| GPT-4o Mini Transcribe | 39.66% | No labels | No labels | No labels | No times | 0.150 | 0.1330 | — | 94/94 |
| Google Chirp 3 | 40.16% | No labels | No labels | No labels | No times | 0.151 | 1.4608 | — | 93/94 |
| Fish Transcribe 1 | 40.24% | No labels | No labels | No labels | 55.95% | 0.152 | 0.5538 | — | 94/94 |
| Deepgram Nova 3 | 40.58% | 65.96% | 67.24% | 54.87% | 71.20% | 0.138 | 0.3968 | — | 94/94 |
| AssemblyAI Universal 3.5 Pro | 42.97% | No labels | No labels | No labels | 46.44% | 0.142 | 0.3460 | — | 94/94 |
| VibeVoice Streaming | 43.71% | 49.38% | No speaker times | No speaker times | No times | 0.115 | — | 16.36 | 94/94 |
| NVIDIA Nemotron 3.5 ASR 0.6B | 45.62% | No labels | No labels | No labels | 45.97% | 0.156 | 🥇 0.0184 | — | 94/94 |
| Whisper Large v3 Turbo | 47.68% | No labels | No labels | No labels | 63.50% | 0.164 | 🥈 0.0226 | — | 94/94 |
| Whisper Large v3 | 48.57% | No labels | No labels | No labels | 55.54% | 0.183 | 🥉 0.0538 | — | 94/94 |
| GPT-4o Transcribe | 57.16% | No labels | No labels | No labels | No times | 0.151 | 0.2254 | — | 94/94 |
| Grok STT 1.0 | 57.55% | 62.83% | 65.88% | 65.96% | 61.61% | 0.158 | 0.1538 | — | 94/94 |
| Voxtral Small 24B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |
| Voxtral Mini 3B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |
| Qwen3 ASR 1.7B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |
| Qwen3 ASR 0.6B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |

RTF = processing seconds / audio seconds; lower is faster. Local inference uses one
RTX 4090 without artificial real-time waits. API timing includes network/provider time.
API $ covers reported charges for returned responses, including parser-rejected responses.
Failed HTTP requests have unknown billing unless reconciled in the [cost ledger](costs.json).
API timing covers completed responses; local timing includes attempted inference.
VRAM is peak allocated GPU memory. Coverage is reference speech time overlapped by
predicted speech, not an accuracy score. Missing or invalid transcripts count as empty
in dataset error rates.
For routes with timing, empty speech output covers zero reference speech; a nonempty
transcript without timing prevents a full-dataset coverage score.
No external diarizer or invented timestamps are added to hosted outputs. Gemini and
Voxtral were asked for diarization but did not return speaker fields in the probe;
their upstream speaker capabilities remain distinct from this tested integration.

[CSV](consolidated.csv) · [Full metrics](consolidated.json) · [Source selection](sources.json).
All 2,350 clip scores were recomputed from saved predictions and references.
[Local run provenance](../ami-4090-94clips-2026-09-27/sources.json) ·
[Capability audit](../../docs/OPENROUTER_CAPABILITY_AUDIT.md).
WER aggregates error counts over reference words; DER aggregates error durations.
No confidence intervals are claimed from four correlated meetings. The WER reference
orders overlapping words chronologically, so it is not an overlap-invariant measure.
Anonymous speaker labels do not test named-person identification or event annotation.

Rebuild and verify: `uv run python scripts/consolidate_results.py --verify`.
AMI-derived material retains its [data attribution and license](../../DATA_LICENSE.md).
