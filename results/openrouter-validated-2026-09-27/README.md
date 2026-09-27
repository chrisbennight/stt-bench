# Consolidated benchmark results

**One comparison, the same audio for every system.**

- **Test audio:** 94 clips, 92.28 minutes total, up to 60 seconds per clip.
- **Speaker comparison:** the same 77 labelled clips for every scored system.
- **Sort order:** WER, lowest first, following the general transcription ranking used by Open ASR.
- **Medals:** 🥇🥈🥉 mark the three lowest measured values per column.
- **Missing results:** cells identify missing labels or timing in the tested output.
  Four excluded DeepInfra routes receive no scores or medals.

Metric definitions, interpretation notes, and references follow the table.

| Model | WER ↓ | cpWER (77) ↓ | tcpWER (77) ↓ | DER (77) ↓ | RTF ↓ | API $ ↓ | VRAM GiB ↓ | Clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MAI Transcribe 2 | 🥇 26.10% | 🥈 35.13% | 🥈 35.97% | 58.76% | 0.145 | 0.1538 | — | 94/94 |
| MAI Transcribe 1.5 | 🥈 26.41% | No labels | No labels | No labels | 0.148 | 0.5538 | — | 94/94 |
| MOSS 0.9B | 🥉 30.92% | 🥇 26.29% | 🥇 27.10% | 🥇 23.57% | 0.073 | — | 🥇 1.92 | 94/94 |
| Fish Transcribe 1 Pro | 31.97% | 40.28% | No speaker times | No speaker times | 0.157 | 0.5538 | — | 94/94 |
| NVIDIA Parakeet TDT v3 | 32.76% | No labels | No labels | No labels | 0.140 | 0.1384 | — | 94/94 |
| Parakeet TDT v3 + pyannote | 33.33% | 55.61% | 60.49% | 🥈 26.83% | 🥇 0.011 | — | 5.72 | 94/94 |
| Qwen3 ASR Flash | 33.34% | No labels | No labels | No labels | 0.164 | 0.1937 | — | 94/94 |
| Gemini 3.5 Transcribe | 33.62% | No labels | No labels | No labels | 0.166 | 0.2770 | — | 94/94 |
| Qwen3 + pyannote | 34.17% | 48.13% | 50.03% | 🥈 26.83% | 0.033 | — | 7.11 | 94/94 |
| Qwen3 + Nemotron | 34.17% | 🥉 39.54% | 🥉 40.76% | 29.95% | 0.032 | — | 6.13 | 94/94 |
| VibeVoice offline | 34.82% | 39.58% | 40.77% | 34.27% | 0.140 | — | 17.09 | 94/94 |
| Meta Muse Voice 1.0 | 35.05% | No labels | No labels | No labels | 0.255 | 0.2768 | — | 94/94 |
| Voxtral Mini 3B + pyannote | 35.82% | 58.48% | 63.02% | 🥈 26.83% | 0.053 | — | 12.02 | 92/94; 2 invalid |
| Voxtral Mini Transcribe | 36.19% | No labels | No labels | No labels | 0.171 | 0.2738 | — | 94/94 |
| Voxtral Small 24B NF4 + pyannote | 37.12% | 58.50% | 62.85% | 27.93% | 0.079 | — | 16.84 | 93/94; 1 invalid |
| Qwen3 ASR 0.6B + pyannote | 37.45% | 49.35% | 51.16% | 🥈 26.83% | 0.030 | — | 🥈 4.77 | 94/94 |
| OpenAI GPT Transcribe | 38.84% | No labels | No labels | No labels | 0.154 | 0.4153 | — | 94/94 |
| Whisper 1 | 39.09% | No labels | No labels | No labels | 0.167 | 0.5538 | — | 94/94 |
| Whisper Large v3 + pyannote | 39.35% | 59.01% | 63.59% | 🥈 26.83% | 0.043 | — | 6.18 | 94/94 |
| GPT-4o Mini Transcribe | 39.66% | No labels | No labels | No labels | 0.150 | 0.1330 | — | 94/94 |
| Google Chirp 3 | 40.16% | No labels | No labels | No labels | 0.151 | 1.4608 | — | 93/94 |
| Fish Transcribe 1 | 40.24% | No labels | No labels | No labels | 0.152 | 0.5538 | — | 94/94 |
| Deepgram Nova 3 | 40.58% | 65.96% | 67.24% | 54.87% | 0.138 | 0.3968 | — | 94/94 |
| AssemblyAI Universal 3.5 Pro | 42.97% | No labels | No labels | No labels | 0.142 | 0.3460 | — | 94/94 |
| VibeVoice Streaming | 43.71% | 49.38% | No speaker times | No speaker times | 0.115 | — | 16.36 | 94/94 |
| Nemotron 3.5 ASR + pyannote | 43.80% | 59.57% | 64.38% | 🥈 26.83% | 🥈 0.015 | — | 5.76 | 94/94 |
| NVIDIA Nemotron 3.5 ASR 0.6B | 45.62% | No labels | No labels | No labels | 0.156 | 🥇 0.0184 | — | 94/94 |
| Whisper Turbo + pyannote | 46.95% | 64.49% | 69.80% | 🥈 26.83% | 🥉 0.017 | — | 🥉 5.05 | 94/94 |
| Whisper Large v3 Turbo | 47.68% | No labels | No labels | No labels | 0.164 | 🥈 0.0226 | — | 94/94 |
| Whisper Large v3 | 48.57% | No labels | No labels | No labels | 0.183 | 🥉 0.0538 | — | 94/94 |
| GPT-4o Transcribe | 57.16% | No labels | No labels | No labels | 0.151 | 0.2254 | — | 94/94 |
| Grok STT 1.0 | 57.55% | 62.83% | 65.88% | 65.96% | 0.158 | 0.1538 | — | 94/94 |
| Voxtral Small 24B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |
| Voxtral Mini 3B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |
| Qwen3 ASR 1.7B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |
| Qwen3 ASR 0.6B | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded | Excluded |

**What the metrics mean**

- **WER — word error rate ↓:** substituted, deleted, and inserted words divided by
  reference words. Measures transcription accuracy; ignores speaker identity.
- **cpWER — concatenated minimum-permutation WER ↓:** measures words and speaker
  attribution, allowing anonymous speaker labels to be renamed for the best match.
- **tcpWER — time-constrained cpWER ↓:** adds timing constraints to speaker-attributed
  word matching. This benchmark uses a five-second collar.
- **DER — diarization error rate ↓:** missed speech, false alarms, and speaker confusion
  divided by reference speaker time. Uses zero collar and includes overlapping speech.
- **RTF — real-time factor ↓:** processing seconds divided by audio seconds.
  For example, 0.1 means processing took one tenth of the audio duration.
- **API $ ↓:** reported response charges in US dollars, including parser-rejected responses.
  Failed requests have unknown billing unless reconciled in the cost ledger.
- **VRAM GiB ↓:** peak allocated GPU memory during local inference.
- **Clips:** successful outputs / total clips. Missing or invalid transcripts count as
  empty hypotheses in dataset error rates.

**Reading the results fairly**

- **Timing:** local inference uses one RTX 4090 without artificial real-time waits.
  API timing includes network/provider time and covers completed responses;
  local timing includes attempted inference.
- **Aggregation:** WER sums word-error counts before division; DER sums error durations.
  Overlapping reference words are ordered chronologically for WER, which makes it
  sensitive to word ordering during overlap.
- **Output limitations:** no external diarizer or invented timestamps are added to hosted
  outputs. Gemini and Voxtral were asked for diarization but returned no speaker fields
  in the probe. This does not establish the limits of every upstream route.
- **Local counterparts:** the additional local ASR pipelines use pyannote Community-1
  and Qwen forced alignment. Voxtral Small uses NF4 quantization, as its name indicates.
  Their diarization scores measure the shared pipeline, not native model diarization.
  See the [local pipeline protocol](../../docs/LOCAL_OPENWEIGHTS.md).
- **Scope:** four correlated meetings do not support claimed confidence intervals.
  Anonymous speaker labels do not test named-person identification or event annotation.

**Download and verify**

All **3,008 clip scores** were recomputed from saved predictions and references.

- **Results:** [CSV](consolidated.csv) · [Full metrics](consolidated.json)
- **Provenance:** [Source selection](sources.json) ·
  [Original local run](../ami-4090-94clips-2026-09-27/sources.json) ·
  [Local counterparts](../local-openweights-2026-09-27/sources.json)
- **Audit:** [Request costs](costs.json) ·
  [Hosted capabilities](../../docs/OPENROUTER_CAPABILITY_AUDIT.md)

Rebuild and verify locally, without new model requests:

```bash
uv run python scripts/consolidate_results.py --verify
```

**References**

- [Open ASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard): WER ranking.
- [MeetEval](https://arxiv.org/abs/2307.11394): speaker-attributed transcription metrics.
- [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/): diarization scoring.
- [Evaluation protocol](../../docs/PROTOCOL.md): normalization, timing, and subset rules.
- [AMI attribution and license](../../DATA_LICENSE.md): terms for the derived data.
