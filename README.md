# Speech transcription with speakers: a five-system benchmark

**MOSS produced the most accurate speaker-attributed transcripts in this test.** Qwen3 with
pyannote had the lowest diarization timing error, and Qwen3 with NVIDIA Nemotron was the
fastest offline pipeline. VibeVoice Streaming ran in real time but made more transcription
and speaker-attribution errors on these recordings.

This repository compares five downloadable speech systems on the **same 92.28 minutes of
English meeting audio**, using one **RTX 4090**. It includes the small Python harness,
model adapters, recorded package versions, pinned model revisions, predictions, references,
and independently verified scores. The measurements were completed on **27 September 2026**.

The question is practical: **which system gets the words and speakers right, how fast does
it run, and how much GPU memory does it use?** This is a controlled, small meeting-audio
comparison—not an OpenASR leaderboard reproduction or a universal model ranking.

**OpenRouter extension:** the harness also supports hosted models through OpenRouter's
speech-to-text API. The [implementation and cost estimate](docs/OPENROUTER.md) cover all
24 transcription model IDs discovered on 27 September 2026. A four-minute screening pass
is estimated at **$0.34 total**; the same audio duration as the full local benchmark is
estimated at **$7.85 total**, using shorter API-compatible windows. These are prospective
costs, not measured model results. OpenRouter evaluation is underway; its results have not
yet been published.

## What is being compared?

There are three approaches here: joint models that generate words and speaker labels together;
pipelines that combine speech recognition, forced alignment, and a separate diarizer; and
a streaming joint model that emits text as audio arrives. Speaker labels are anonymous IDs
within a clip. They do not identify a person by name.

| System | Components and output |
| --- | --- |
| **MOSS 0.9B** | [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize): joint transcription, speaker labels, and segment timestamps. |
| **VibeVoice offline** | [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR): joint structured transcripts with speakers and timestamps. |
| **Qwen3 + pyannote** | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), and [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1). Words are assigned to speakers by temporal overlap. |
| **Qwen3 + Nemotron** | The same Qwen recognition/alignment stages, with [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) supplying speaker activity. |
| **VibeVoice Streaming** | [VibeVoice-ASR-Streaming-7B](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B): incremental speaker-labelled text. This adapter receives no native speech timestamps. |

Some models can also describe sound events. **Annotation/event accuracy is not evaluated**
here; bracketed VibeVoice sound annotations are retained in raw output but excluded from
speech-word scoring. All five systems use local inference; no paid transcription service is
part of the measurement. Check each provider's model license and access terms separately.

The scoring libraries are [MeetEval](https://github.com/fgnt/meeteval) and
[pyannote.metrics](https://pyannote.github.io/pyannote-metrics/). Model cards describe upstream
capabilities; the tables below are this repository's own measurements. More authoritative
sources and related evaluations are collected in [References](docs/REFERENCES.md).

## The test

We use the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/), with official
[manual annotations, release 1.6.2](https://groups.inf.ed.ac.uk/ami/download/), and the pinned
[pyannote AMI speaker-activity references](https://github.com/pyannote/AMI-diarization-setup).

| Setting | Measured configuration |
| --- | --- |
| Meetings | ES2004a, IS1009a, TS3003a, EN2002a: four AMI test meeting groups |
| Input | Array1-01 distant microphone, mono 16 kHz English audio; overlapping speech included |
| Windows | 25 nonoverlapping clips, each at most 240 seconds; 5,536.704 seconds total |
| Hardware | One NVIDIA RTX 4090, 24 GB; one model process at a time |
| Precision | BF16, no quantization; no oracle speaker counts or reference text supplied to models |
| Timing | Model loading excluded; no warmup; first inference includes cold effects |
| Randomness | One trial; per-window seed derived from the recording ID, identical policy for all systems |
| Streaming | Audio paced in real time, including lookahead; speaker state resets per window |

Text normalization preserves fillers, repetitions, and number spellings. No LLM cleans up
the references or hypotheses. Errors are summed before division; the results are not an
average of per-clip percentages. See the [full protocol](docs/PROTOCOL.md), including exact
boundary handling, decoding settings, and pipeline limitations.

## Results

**Lower error rates are better.** cpWER measures transcription and speaker attribution while
allowing anonymous labels to be renamed. DER measures who spoke when: missed speech, false
alarms, and speaker confusion. It uses **zero collar and includes overlap** here. These are
different objectives, so their rankings need not agree.

| System | cpWER | DER | Total inference time | Peak CUDA allocation | Valid outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| **MOSS 0.9B** | **29.39%** | 26.50% | 7.06 min | **2.41 GiB** | 25/25 |
| VibeVoice offline | 40.57% | 37.94% | 23.74 min | 19.76 GiB | 24/25 |
| Qwen3 + pyannote | 44.38% | **24.68%** | 3.06 min | 7.11 GiB | 25/25 |
| Qwen3 + Nemotron | 40.78% | 31.16% | **2.52 min** | 6.13 GiB | 25/25 |
| VibeVoice Streaming | 54.33% | — | 92.43 min, paced | 16.51 GiB | 25/25 |

Runtime includes failed generations. CUDA allocation is the PyTorch peak, not all device
memory. Streaming wall time deliberately includes waiting for audio; its model-compute total
was **10.55 minutes**. Its **p95 chunk-end-to-emission delay was 1.28 seconds** and mean first
speech-text emission was **6.20 seconds**, including initial silence. These are chunk-level
measurements, not word-aligned latency. Native DER, tcpWER, and coverage are unavailable for
that adapter because its output has no speech timestamps.

### Per-meeting cpWER

| Meeting | MOSS | VibeVoice offline | Qwen/pyannote | Qwen/Nemotron | VibeVoice Streaming |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN2002a | **29.97%** | 45.14% | 49.40% | 44.46% | 51.86% |
| ES2004a | **34.60%** | 38.90% | 39.13% | 36.64% | 55.07% |
| IS1009a | **22.60%** | 27.75% | 36.27% | 33.55% | 61.10% |
| TS3003a | **27.62%** | 38.75% | 41.24% | 39.78% | 55.60% |

[Full results](docs/RESULTS.md) include WER, timing-constrained cpWER, coverage, failure details,
and a sensitivity check on the 24 windows where every system returned valid output.

## Findings and limitations

- **MOSS is the best starting point for this workload.** It has the lowest cpWER on all four
  meetings, uses the least GPU memory, and processes the audio about 13 times faster than real time.
- **Pyannote wins the timing-based diarization measure, not the complete transcription pipeline.**
  Both Qwen pipelines produce identical raw ASR text; their speaker attribution differs.
  Nemotron is the fastest offline combination here, at about 36.6 times real time.
- **Offline VibeVoice had a real generation failure.** One window repeated text to the
  32,768-token cap and produced invalid JSON. It is scored as an empty hypothesis, its raw output
  is retained, and its runtime is included. Removing that same window from every system does
  not change MOSS's lead. VibeVoice's and Qwen/Nemotron's aggregate cpWER values are close;
  this trial does not establish a meaningful difference between them.
- **Streaming trades accuracy for incremental output in this configuration.** It completed
  all clips, but had the highest cpWER. A split sound annotation exposed a parser bug; the
  published correction reparses saved text without rerunning inference or changing timing.
  [The original records and the correction audit are both included](results/ami-4090-2026-09-27/README.md).

These are four correlated English meeting groups and one seeded trial. AMI may appear in
model training or development. This does not measure multilingual accuracy, long-session
speaker consistency, named-person recognition, or sound-event annotation quality. Native
segment times and forced alignment also provide different timestamp sources. Testing on
held-out recordings from the intended application, a second corpus such as NOTSOFAR, and
repeated trials would be the most useful extensions.

## Inspect or reproduce

The checked-in results require **no GPU, model download, token, or audio download** to verify:

```bash
uv sync --locked --python 3.12
uv run pytest -q
uv run python scripts/verify_scores.py results/ami-4090-2026-09-27/corrected --output verification.json
```

The verifier recomputes all 125 scores from saved predictions and references and checks that
scored outputs match the prediction files. The published original and corrected runs each
have zero score mismatches. [Running the models](docs/RUNNING.md) covers environments, access
terms, downloads, the exact four-meeting recipe, and the simple adapter interface.

- [Result files and provenance guide](results/ami-4090-2026-09-27/README.md)
- [CSV summary](results/ami-4090-2026-09-27/comparison/summary.csv),
  [per-meeting CSV](results/ami-4090-2026-09-27/comparison/meetings.csv), and
  [complete JSON comparison](results/ami-4090-2026-09-27/comparison/comparison.json)
- [Pinned model revisions](model-revisions.json) and [upstream source revisions](upstream-revisions.json)
- [Security and publication scope](SECURITY.md)

## Licenses and attribution

Original harness code and documentation are [MIT licensed](LICENSE). AMI-derived references
and transcript material retain their separate [CC BY 4.0 attribution and terms](DATA_LICENSE.md).
Model weights and source audio are not redistributed. Each upstream model and library retains
its own license; this repository's license does not override those terms.
