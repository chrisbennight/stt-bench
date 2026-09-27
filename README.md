# Speech transcription benchmark: local models and OpenRouter

**One consolidated comparison using the same 94 audio clips and references.** The
[results table](results/openrouter-validated-2026-09-27/README.md) compares transcription accuracy,
speaker attribution, diarization, speed, cost, and GPU memory, with per-metric medals.
**MAI Transcribe 2 leads WER at 26.10%; MOSS leads the local models at 30.92% WER
and leads all three speaker error metrics on the shared 77-clip subset.**

This repository compares 12 local speech systems and 20 OpenRouter routes on
**92.28 minutes of English meeting audio**. Local models use one **RTX 4090**. It includes
the small Python harness,
model adapters, recorded package versions, pinned model revisions, predictions, references,
and independently verified scores. Four additional DeepInfra routes are excluded
after earlier failures. The measurements are from **27 September 2026**.

The question is practical: **which system gets the words and speakers right, how fast does
it run, and how much GPU memory does it use?** This small meeting-audio evaluation is
not an OpenASR leaderboard reproduction or a universal model ranking.

The first hosted pass did not validate each route's speaker and timestamp options.
The corrected pass follows **42 bounded capability probes**, with tested request options,
actual response fields, and limitations documented in the
[capability audit](docs/OPENROUTER_CAPABILITY_AUDIT.md). The
[cost ledger](results/openrouter-validated-2026-09-27/costs.json) includes those probes,
the corrected pass, and failed attempts. The original five local systems were each run once on
the exact same manifest and their valid results are reused. Seven additional
[local OpenRouter counterparts](results/local-openweights-2026-09-27/README.md) use
pyannote Community-1 as their default diarizer. No external diarizer or
invented timestamps are added to hosted outputs.

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
| **Local OpenRouter counterparts + pyannote** | Parakeet, Nemotron ASR, Whisper large-v3 and Turbo, Qwen 0.6B, and Voxtral Mini/Small, with Qwen forced alignment and pyannote Community-1. Voxtral Small uses NF4 quantization. [Weights, configuration, and protocol](docs/LOCAL_OPENWEIGHTS.md). |

Some models can also describe sound events. **Annotation/event accuracy is not evaluated**
here; bracketed VibeVoice sound annotations are retained in raw output but excluded from
speech-word scoring. Names ending in **+ pyannote** identify local pipelines; the hosted
routes retain their provider model names. Check each provider's model license and access terms
separately.

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
| Windows | 94 nonoverlapping clips, each at most 60 seconds; 5,536.704 seconds total |
| Hardware | Local: one NVIDIA RTX 4090, 24 GB, one model at a time; hosted: provider-controlled |
| Precision | Local: BF16 or NeMo checkpoint default; Voxtral Small: NF4; hosted: provider-defined; no oracle speakers or reference text |
| Timing | Local: loading excluded, no warmup; API: request wall time, including network/provider time |
| Randomness | One local trial with per-window seeds from recording IDs; API generation seeds are not controlled |
| Streaming | The same supplied audio, processed without artificial waits; state resets per window |

Text normalization preserves fillers, repetitions, and number spellings. No LLM cleans up
the references or hypotheses. Errors are summed before division; the results are not an
average of per-clip percentages. See the [full protocol](docs/PROTOCOL.md), including exact
boundary handling, decoding settings, and pipeline limitations.

## Results

**Lower error rates are better.** cpWER measures transcription and speaker attribution while
allowing anonymous labels to be renamed. DER measures who spoke when: missed speech, false
alarms, and speaker confusion. It uses **zero collar and includes overlap** here. These are
different objectives, so their rankings need not agree.

The [single results table](results/openrouter-validated-2026-09-27/README.md) contains all 32 scored systems,
sorted by WER. WER uses 94 clips. Speaker metrics use one shared set of clips
where all four speaker-labelled hosted routes supplied usable labels, including for
the local systems; the count is in the column headings.
The [full JSON](results/openrouter-validated-2026-09-27/consolidated.json) also retains each system's
full-dataset metrics. Streaming has no native speech timestamps, so its DER and tcpWER
remain unavailable. The historical 25-window local run is retained in
[the archive](docs/RESULTS.md) and is excluded from the current comparison.

## Findings and limitations

- MAI Transcribe 2 (26.10%), MAI Transcribe 1.5 (26.41%), and MOSS (30.92%) have
  the lowest WER. On the shared 77 labelled clips, MOSS has 26.29% cpWER,
  27.10% tcpWER, and 23.57% DER. This speaker subset is conditional on the hosted
  systems returning labels; it is not an estimate for all 94 clips.
- The original five local systems and five additional pipelines returned valid output
  on all 94 clips. Voxtral Mini returned 92/94; Voxtral Small NF4 returned 93/94.
  Their three repeating, token-limited generations are retained as failures and
  count in dataset error rates.
- Among the seven added local pipelines, **Parakeet + pyannote** has the lowest WER
  (**33.33%**) and fastest measured processing (**0.011 RTF**). MOSS remains the best
  local system by WER and the shared speaker-error metrics.
- The added ASR pipelines share pyannote diarization, so equal DER values largely
  reflect the same component. Their cpWER and tcpWER also measure transcription and
  alignment errors. These are not native diarization scores for the ASR weights.
- MAI 2, Deepgram, and Grok returned timed speaker labels. Fish Pro returned inline
  speaker-labelled text and separate speech timing, supporting cpWER
  without enough information for tcpWER or DER.
- Gemini and Voxtral returned timestamps but no speaker fields with the tested
  diarization options. Their upstream capabilities are not disproved by this result.
  The [request and raw-response review](results/openrouter-diarization-review-2026-09-27/README.md)
  also checks Meta, confirms the missing fields precede our parser, and tests alternative
  options. These are hosted integration observations, not proof of model incapability.
- Transcription accuracy, speaker attribution, and diarization timing are distinct measures;
  their winners need not agree. Missing metric cells mean the tested output could not support
  that score, rather than proving that every upstream route lacks the capability.
- Four DeepInfra routes remain excluded and receive no scores or medals. The earlier
  incomplete attempts remain in the archive.
- Chirp returned 93/94 clips. One 60-second clip timed out, then returned HTTP 504
  with a longer timeout and lossless WAV transport. It counts as an empty transcript
  in the 94-clip WER; the benchmark does not shorten that window. Failed attempts and
  unknown billing remain in the ledger. Three Gemini timestamp errors were repaired
  offline using valid native segment times from the same saved responses.
- Local runs are sequential on one GPU. API timing includes network and provider processing;
  the displayed speed is observed throughput, not a hardware-normalized model comparison.

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
uv run python scripts/consolidate_results.py --verify
```

The verifier recomputes all 2,350 scores from saved predictions and references and checks
that the local and API clip hashes and references match. [Running the models](docs/RUNNING.md)
covers environments, access
terms, downloads, the exact four-meeting recipe, and the simple adapter interface.

- [CSV summary](results/openrouter-validated-2026-09-27/consolidated.csv) and
  [complete JSON comparison](results/openrouter-validated-2026-09-27/consolidated.json)
- [Original 25-window archive](results/ami-4090-2026-09-27/README.md)
- [Pinned model revisions](model-revisions.json) and [upstream source revisions](upstream-revisions.json)
- [Security and publication scope](SECURITY.md)

## Licenses and attribution

Original harness code and documentation are [MIT licensed](LICENSE). AMI-derived references
and transcript material retain their separate [CC BY 4.0 attribution and terms](DATA_LICENSE.md).
Model weights and source audio are not redistributed. Each upstream model and library retains
its own license; this repository's license does not override those terms.
