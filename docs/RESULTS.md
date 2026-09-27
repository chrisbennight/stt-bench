# Historical 25-window local benchmark results

These results are archived. The [current consolidated comparison](../results/openrouter-2026-09-27/README.md)
uses the same 94-clip manifest for local and hosted models.

Completed 27 September 2026 using one RTX 4090.

**MOSS is the strongest choice for speaker-attributed meeting transcripts in this test.**
It has the lowest cpWER on every meeting, the lowest allocated GPU memory, and valid output
on all 25 windows. Qwen/pyannote has the lowest diarization timing error; Qwen/Nemotron is
the fastest offline pipeline.

The comparison covers 25 identical windows from four AMI test meeting groups: 92.28 minutes
of single-channel distant-microphone English audio, including overlapping speech. Models run
sequentially in BF16 without quantization or an oracle speaker count. This is one seeded trial,
not a general model leaderboard.

## Main comparison

Lower errors are better. cpWER measures words and speaker attribution; DER measures who spoke
when. Runtime includes failed generations and excludes loading. GPU allocation is the PyTorch
peak, not total device memory.

| System | cpWER | DER | Attempted runtime | Peak CUDA allocation | Valid outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOSS 0.9B | 29.39% | 26.50% | 7.06 min | 2.41 GiB | 25/25 |
| VibeVoice offline | 40.57% | 37.94% | 23.74 min | 19.76 GiB | 24/25 |
| Qwen3 + pyannote | 44.38% | 24.68% | 3.06 min | 7.11 GiB | 25/25 |
| Qwen3 + Nemotron | 40.78% | 31.16% | 2.52 min | 6.13 GiB | 25/25 |
| VibeVoice Streaming | 54.33% | — | 92.43 min paced | 16.51 GiB | 25/25 |

All 125 model/window attempts finished: 124 valid outputs and one invalid offline VibeVoice
generation. Every worker exited successfully. The overall command returned status 2 because
it deliberately flags any recorded failure.

## Practical interpretation

- **MOSS:** best transcription with speaker attribution on this workload, at about 13.1 times
  real time and 2.41 GiB allocated GPU memory. Its DER is slightly worse than pyannote.
- **Qwen/pyannote:** best speaker activity timing, with 24.68% DER, but 44.38% cpWER. Diarization
  quality and complete transcription-and-assignment quality are different measures.
- **Qwen/Nemotron:** fastest offline option, at about 36.6 times real time. It has worse DER
  than pyannote but better speaker-attributed transcription here. Both Qwen pipelines produced
  identical raw ASR text on all 25 windows.
- **VibeVoice offline:** higher memory use and one long repetition failure. Its 40.57% cpWER
  is close to Qwen/Nemotron's 40.78%; this trial does not establish a meaningful difference
  between those two values.
- **VibeVoice Streaming:** completed every window, but its 54.33% cpWER is the highest in
  this comparison. It provides live anonymous speaker labels and text without native speech
  timestamps. DER, tcpWER, and coverage are therefore unavailable.

## Streaming timing

The paced run took **92.43 minutes** for 92.28 minutes of audio. Model computation totalled
**10.55 minutes** (compute RTF 0.114); the remaining time largely comes from deliberate
real-time audio pacing. This compute figure is supplementary, not an independently measured
unpaced run.

The **p95 chunk-end-to-emission delay was 1.28 seconds**. Mean first speech-text emission was
**6.20 seconds** from each window's start, including initial silence and buffering. Chunk
delay includes lookahead and generation; neither number is word-aligned latency.

One `[Environmental Sounds]` annotation crossed chunk boundaries in TS3003a-0. The original
parser counted its fragments as speech. The corrected parser interprets the exact combined
text and retains the measured event timestamps. Only that window changed in normalized
speaker text or speech-event flags. Streaming cpWER changed from 54.3371% to 54.3304%; mean
first-text time changed from 5.59 to 6.20 seconds. The original run, frozen inference source,
and per-record reprocessing audit are retained. See [protocol details](PROTOCOL.md).

## Failure and sensitivity check

On EN2002a-19200000, offline VibeVoice repeated a phrase until the 32,768-token cap, taking
672.66 seconds and producing invalid JSON. This was a completed model generation, not an
infrastructure interruption. It counts as an empty hypothesis, and its runtime remains
included. Successful-window runtime alone was 12.52 minutes; the main table includes all
23.74 minutes.

Restricting every system to the same 24 windows with valid output is a supplementary
sensitivity check, not the primary ranking:

| System | cpWER on the common 24 windows |
| --- | ---: |
| MOSS | 29.84% |
| VibeVoice offline | 37.11% |
| Qwen3 + pyannote | 44.88% |
| Qwen3 + Nemotron | 41.17% |
| VibeVoice Streaming | 55.70% |

## Per-meeting cpWER

| Meeting | MOSS | VibeVoice offline | Qwen/pyannote | Qwen/Nemotron | VibeVoice Streaming |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN2002a | 29.97% | 45.14% | 49.40% | 44.46% | 51.86% |
| ES2004a | 34.60% | 38.90% | 39.13% | 36.64% | 55.07% |
| IS1009a | 22.60% | 27.75% | 36.27% | 33.55% | 61.10% |
| TS3003a | 27.62% | 38.75% | 41.24% | 39.78% | 55.60% |

## Supplementary quality metrics

| System | Chronological WER | tcpWER, 5-second collar | Speech-time coverage |
| --- | ---: | ---: | ---: |
| MOSS | 30.29% | 30.62% | 89.44% |
| VibeVoice offline | 37.33% | 41.64% | 89.49% |
| Qwen3 + pyannote | 34.04% | 47.32% | 93.29% |
| Qwen3 + Nemotron | 34.04% | 42.38% | 78.33% |
| VibeVoice Streaming | 42.83% | — | — |

Chronological WER is secondary because overlapping speakers have no unique word interleaving.
Coverage is a diagnostic and can be inflated by hallucinated spans. DER has a zero-second
collar and includes overlap. The Qwen pipelines use forced alignment; joint models use native
segment times. Their tcpWER timestamp sources are therefore different.

## Verification and reproducibility

All 125 original score records and all 125 corrected score records were independently
recomputed locally, with zero mismatches. Raw predictions match the scored records.
Additional checks confirmed all 125 seeds and preserved measurements, and identical Qwen
ASR text on all 25 paired windows. Regression tests cover chunk-boundary parsing and
preservation of original output and timing; publication checks also validate result hashes.

- [Final CSV summary](../results/ami-4090-2026-09-27/comparison/summary.csv) and
  [complete comparison JSON](../results/ami-4090-2026-09-27/comparison/comparison.json)
- [Final verification record](../results/ami-4090-2026-09-27/score-verification-final.json) and
  [supplementary checks](../results/ami-4090-2026-09-27/comparison/supplemental.json)
- [Corrected predictions and references](../results/ami-4090-2026-09-27/corrected/) and
  [original measured run](../results/ami-4090-2026-09-27/original/)
- [Reprocessing audit](../results/ami-4090-2026-09-27/corrected/reprocessing.json)
- [Recorded runtime packages](../results/ami-4090-2026-09-27/provenance/packages/) and
  [original inference source hashes](../results/ami-4090-2026-09-27/provenance/inference-source-sha256.json)
- [Runnable setup and adapter instructions](RUNNING.md), [full protocol](PROTOCOL.md),
  [benchmark research](REFERENCES.md), and [security review scope](../SECURITY.md)

To verify delivered scores without models or original audio:

```bash
uv sync --locked
uv run python scripts/verify_scores.py results/ami-4090-2026-09-27/corrected --output /tmp/speaker-score-verification.json
```

To reproduce the parsing correction from the preserved raw run:

```bash
uv run python scripts/reparse_streaming_run.py results/ami-4090-2026-09-27/original --output /tmp/speaker-reparsed
```

Choose new output paths if those already exist. Full inference setup, model revisions, and
the exact four-meeting preparation commands are in [the running guide](RUNNING.md).
No model weights or source audio are included in this repository.

## Limits and next steps

These are four correlated English meeting groups, with speaker identities reset every four
minutes. There is no repeated-trial uncertainty estimate, multilingual evaluation, long-session
identity test, or annotation-event accuracy score. AMI may appear in model training or
development. GPU allocations exclude non-PyTorch memory, loading is excluded, and the first
inference is cold. The Qwen aligner has a small documented token-merging effect.

For this workload, start with MOSS. Before a production choice, the most informative next
comparison is a held-out sample of your actual audio, followed by a different meeting corpus
such as NOTSOFAR and repeated trials. Those are recommendations, not additional measurements
claimed here.
