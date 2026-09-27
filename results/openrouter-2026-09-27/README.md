# Consolidated benchmark results

Sorted by **word error rate (WER), lower is better**, the general transcription metric
used for ranking by [Open ASR](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
Speaker attribution is additionally measured by cpWER and tcpWER, following
[MeetEval](https://arxiv.org/abs/2307.11394); DER measures speaker activity errors.
🥇🥈🥉 mark the three lowest measured values per column, except coverage (higher).
These are descriptive ranks across the documented protocol differences, not proof
of a controlled head-to-head win. — means unavailable, not zero.

| Model | WER ↓ | cpWER ↓ | tcpWER ↓ | DER ↓ | Coverage ↑ | RTF ↓ | API $ ↓ | VRAM GiB ↓ | Clips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAI Transcribe 2‡ | 🥇 26.10% | 🥈 36.35% | 🥈 37.37% | 59.71% | 55.09% | 0.102 | 0.1538 | — | 94/94 |
| MAI Transcribe 1.5 | 🥈 26.54% | — | — | — | — | 0.106 | 0.5538 | — | 94/94 |
| MOSS 0.9B* | 🥉 30.29% | 🥇 29.39% | 🥇 30.62% | 🥈 26.50% | 🥉 89.44% | 0.077 | — | 🥇 2.41 | 25/25 |
| NVIDIA Parakeet TDT v3 | 32.78% | — | — | — | — | 0.091 | 0.1384 | — | 94/94 |
| AssemblyAI Universal 3.5 Pro | 33.06% | — | — | — | — | 0.098 | 0.3460 | — | 94/94 |
| Qwen3 ASR Flash | 33.34% | — | — | — | — | 0.101 | 0.1937 | — | 94/94 |
| Qwen3 + pyannote* | 34.04% | 44.38% | 47.32% | 🥇 24.68% | 🥇 93.29% | 🥉 0.033 | — | 🥉 7.11 | 25/25 |
| Qwen3 + Nemotron* | 34.04% | 40.78% | 42.38% | 🥉 31.16% | 78.33% | 🥈 0.027 | — | 🥈 6.13 | 25/25 |
| Meta Muse Voice 1.0 | 34.29% | — | — | — | — | 0.243 | 0.2768 | — | 94/94 |
| Gemini 3.5 Transcribe | 35.48% | — | — | — | — | 0.107 | 0.2770 | — | 94/94 |
| Voxtral Mini Transcribe | 36.69% | — | — | — | — | 0.082 | 0.2738 | — | 94/94 |
| VibeVoice offline*§ | 37.33% | 🥉 40.57% | 🥉 41.64% | 37.94% | 🥈 89.49% | 0.257 | — | 19.76 | 24/25 |
| OpenAI GPT Transcribe | 38.63% | — | — | — | — | 0.084 | 0.4153 | — | 94/94 |
| Google Chirp 3† | 38.68% | — | — | — | — | 0.175 | 1.4768 | — | 94/94 |
| Whisper 1 | 39.64% | — | — | — | — | 0.105 | 0.5538 | — | 94/94 |
| GPT-4o Mini Transcribe | 39.89% | — | — | — | — | 0.080 | 0.1330 | — | 94/94 |
| Fish Transcribe 1 | 40.19% | — | — | — | — | 0.114 | 0.5538 | — | 94/94 |
| Deepgram Nova 3‡ | 40.60% | 66.68% | 67.98% | 56.38% | 71.21% | 🥇 0.017 | 0.3968 | — | 94/94 |
| VibeVoice Streaming* | 42.83% | 54.33% | — | — | — | 1.002 | — | 16.51 | 25/25 |
| Whisper Large v3 | 43.78% | — | — | — | — | 0.135 | 🥉 0.1081 | — | 94/94 |
| NVIDIA Nemotron 3.5 ASR 0.6B | 45.62% | — | — | — | — | 0.118 | 🥇 0.0184 | — | 94/94 |
| Whisper Large v3 Turbo | 47.91% | — | — | — | — | 0.128 | 🥈 0.0212 | — | 94/94 |
| Grok STT 1.0 | 56.16% | — | — | — | — | 0.081 | 0.1538 | — | 94/94 |
| GPT-4o Transcribe | 57.57% | — | — | — | — | 0.082 | 0.2252 | — | 94/94 |
| Fish Transcribe 1 Pro | 69.81% | — | — | — | — | 0.120 | 0.5538 | — | 94/94 |
| Qwen3 ASR 0.6B¶ | 75.82% | — | — | — | — | 0.128 | 0.0087 | — | 44/94 |
| Voxtral Mini 3B¶ | 78.21% | — | — | — | — | 0.131 | 0.0315 | — | 32/94 |
| Voxtral Small 24B¶ | 85.86% | — | — | — | — | 0.194 | 0.0645 | — | 22/94 |
| Qwen3 ASR 1.7B¶ | 88.41% | — | — | — | — | 0.140 | 0.0079 | — | 18/94 |

* **\*** Local RTX 4090: 25 windows up to 240 seconds. Hosted models: 94 windows up
  to 60 seconds. Both cover the same 92.28 minutes. Streaming was paced in real time;
  first text averaged 6.20 seconds, with 1.28-second p95 chunk-emission lag.
* **†** Chirp uses two 30-second requests for one recovered clip; all other clips
  retain the standard window. [Recovery details](../openrouter-chirp-2026-09-27/README.md).
* **‡** MAI 2 and Deepgram speaker metrics use the same 89 clips with labels from
  both models. WER and coverage use all 94. Other API routes were not tested with
  verified diarization options; missing cells do not establish model incapability.
* **§ / ¶** VibeVoice's one invalid output and DeepInfra's missing clips count as
  empty transcripts in full-dataset error rates. DeepInfra costs and RTF cover only
  completed clips and receive no medals. Their WER reflects service failures too.

RTF = processing seconds / audio seconds; lower is faster. API timing includes network
and provider time and excludes failed requests and recovery delays; local timing
includes attempted inference. API $ covers selected successful responses, not the
entire session bill or local electricity. VRAM is peak allocated GPU memory, not
total required capacity. Coverage measures overlap with reference speech time and
can be inflated by false speech; its medals indicate coverage only.

[CSV](consolidated.csv) · [Full metrics](consolidated.json) · [Source selection](sources.json).
All 2,256 API clip scores were recomputed from the bundled predictions and references.
The local scores retain their
[published provenance](../ami-4090-2026-09-27/comparison/comparison.json).
WER aggregates error counts over reference words; DER aggregates error durations.
No confidence intervals are claimed from four correlated meetings. The WER reference
orders overlapping words chronologically, so it is not an overlap-invariant measure.
Anonymous speaker labels do not test named-person identification or event annotation.

Rebuild and verify: `uv run python scripts/consolidate_results.py --verify`.
AMI-derived material retains its [data attribution and license](../../DATA_LICENSE.md).
