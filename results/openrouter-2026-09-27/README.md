# Consolidated benchmark results

Sorted by **word error rate (WER), lower is better**, the general transcription metric
used for ranking by [Open ASR](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
Speaker attribution is additionally measured by cpWER and tcpWER, following
[MeetEval](https://arxiv.org/abs/2307.11394); DER measures speaker activity errors.
🥇🥈🥉 mark the three lowest measured values per column, except coverage (higher).
All systems use the same 94 clips and references: 92.28 minutes, up to 60 seconds
per clip. Speaker metrics use the same 89 labelled clips for every scored system.
— means unavailable, not zero. Incomplete API runs receive no medals.

| Model | WER ↓ | cpWER (89) ↓ | tcpWER (89) ↓ | DER (89) ↓ | Coverage ↑ | RTF ↓ | API $ ↓ | VRAM GiB ↓ | Clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAI Transcribe 2 | 🥇 26.10% | 🥈 36.35% | 🥈 37.37% | 59.71% | 55.09% | 0.102 | 0.1538 | — | 94/94 |
| MAI Transcribe 1.5 | 🥈 26.54% | — | — | — | — | 0.106 | 0.5538 | — | 94/94 |
| MOSS 0.9B | 🥉 30.92% | 🥇 27.76% | 🥇 28.66% | 🥇 25.31% | 🥉 88.83% | 0.073 | — | 🥇 1.92 | 94/94 |
| NVIDIA Parakeet TDT v3 | 32.78% | — | — | — | — | 0.091 | 0.1384 | — | 94/94 |
| AssemblyAI Universal 3.5 Pro | 33.06% | — | — | — | — | 0.098 | 0.3460 | — | 94/94 |
| Qwen3 ASR Flash | 33.34% | — | — | — | — | 0.101 | 0.1937 | — | 94/94 |
| Qwen3 + pyannote | 34.17% | 49.24% | 51.49% | 🥈 28.37% | 🥈 92.99% | 🥉 0.033 | — | 🥉 7.11 | 94/94 |
| Qwen3 + Nemotron | 34.17% | 🥉 40.81% | 🥉 42.27% | 🥉 31.36% | 77.97% | 🥈 0.032 | — | 🥈 6.13 | 94/94 |
| Meta Muse Voice 1.0 | 34.29% | — | — | — | — | 0.243 | 0.2768 | — | 94/94 |
| VibeVoice offline | 34.82% | 41.40% | 42.62% | 35.96% | 🥇 94.74% | 0.140 | — | 17.09 | 94/94 |
| Gemini 3.5 Transcribe | 35.48% | — | — | — | — | 0.107 | 0.2770 | — | 94/94 |
| Voxtral Mini Transcribe | 36.69% | — | — | — | — | 0.082 | 0.2738 | — | 94/94 |
| OpenAI GPT Transcribe | 38.63% | — | — | — | — | 0.084 | 0.4153 | — | 94/94 |
| Google Chirp 3 | 38.68% | — | — | — | — | 0.175 | 1.4768 | — | 94/94; 1 split |
| Whisper 1 | 39.64% | — | — | — | — | 0.105 | 0.5538 | — | 94/94 |
| GPT-4o Mini Transcribe | 39.89% | — | — | — | — | 0.080 | 0.1330 | — | 94/94 |
| Fish Transcribe 1 | 40.19% | — | — | — | — | 0.114 | 0.5538 | — | 94/94 |
| Deepgram Nova 3 | 40.60% | 66.68% | 67.98% | 56.38% | 71.21% | 🥇 0.017 | 0.3968 | — | 94/94 |
| VibeVoice Streaming | 43.71% | 51.93% | — | — | — | 0.115 | — | 16.36 | 94/94 |
| Whisper Large v3 | 43.78% | — | — | — | — | 0.135 | 🥉 0.1081 | — | 94/94 |
| NVIDIA Nemotron 3.5 ASR 0.6B | 45.62% | — | — | — | — | 0.118 | 🥇 0.0184 | — | 94/94 |
| Whisper Large v3 Turbo | 47.91% | — | — | — | — | 0.128 | 🥈 0.0212 | — | 94/94 |
| Grok STT 1.0 | 56.16% | — | — | — | — | 0.081 | 0.1538 | — | 94/94 |
| GPT-4o Transcribe | 57.57% | — | — | — | — | 0.082 | 0.2252 | — | 94/94 |
| Fish Transcribe 1 Pro | 69.81% | — | — | — | — | 0.120 | 0.5538 | — | 94/94 |
| Qwen3 ASR 0.6B | 75.82% | — | — | — | — | 0.128 | 0.0087 | — | 44/94 |
| Voxtral Mini 3B | 78.21% | — | — | — | — | 0.131 | 0.0315 | — | 32/94 |
| Voxtral Small 24B | 85.86% | — | — | — | — | 0.194 | 0.0645 | — | 22/94 |
| Qwen3 ASR 1.7B | 88.41% | — | — | — | — | 0.140 | 0.0079 | — | 18/94 |

RTF = processing seconds / audio seconds; lower is faster. Local inference uses one
RTX 4090 without artificial real-time waits. API timing includes network/provider time.
API $ and API timing cover selected successful responses, excluding failed requests
and recovery delays; local timing includes attempted inference. VRAM is peak allocated
GPU memory. Coverage is reference speech time overlapped by predicted speech, not an
accuracy score. Missing or invalid transcripts count as empty in dataset error rates.
The Clips column records incomplete runs and Chirp's one split-request recovery.
Missing speaker cells describe the tested route, not all upstream model capabilities.

[CSV](consolidated.csv) · [Full metrics](consolidated.json) · [Source selection](sources.json).
All 2,726 clip scores were recomputed from saved predictions and references.
[Local run provenance](../ami-4090-94clips-2026-09-27/sources.json) ·
[Chirp recovery](../openrouter-chirp-2026-09-27/README.md).
WER aggregates error counts over reference words; DER aggregates error durations.
No confidence intervals are claimed from four correlated meetings. The WER reference
orders overlapping words chronologically, so it is not an overlap-invariant measure.
Anonymous speaker labels do not test named-person identification or event annotation.

Rebuild and verify: `uv run python scripts/consolidate_results.py --verify`.
AMI-derived material retains its [data attribution and license](../../DATA_LICENSE.md).
