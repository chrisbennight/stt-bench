# Speech transcription benchmark: local models and OpenRouter

**One consolidated comparison using the same 94 audio clips and references.** The
[results table](results/openrouter-2026-09-27/README.md) compares transcription accuracy,
speaker attribution, diarization, speed, cost, and GPU memory, with per-metric medals.

This repository compares five downloadable speech systems and 24 OpenRouter model IDs on
**92.28 minutes of English meeting audio**. Local models use one **RTX 4090**. It includes
the small Python harness,
model adapters, recorded package versions, pinned model revisions, predictions, references,
and independently verified scores. The measurements were completed on **27 September 2026**.

The question is practical: **which system gets the words and speakers right, how fast does
it run, and how much GPU memory does it use?** This small meeting-audio evaluation is
not an OpenASR leaderboard reproduction or a universal model ranking.

**OpenRouter extension:** the harness also supports hosted models through OpenRouter's
speech-to-text API. The [implementation and cost estimate](docs/OPENROUTER.md) cover all
24 transcription model IDs discovered on 27 September 2026. A four-minute screening pass
is estimated at **$0.34 total**; the same audio duration as the full local benchmark is
estimated at **$7.85 total**, using the shared 60-second windows. These are prospective
costs, not measured model results. The
[consolidated results table](results/openrouter-2026-09-27/README.md) now includes all
24 hosted models and five local systems, sorted by WER with top-three metric badges.
The five local systems were each run once on the exact API manifest. Speaker-metric
coverage is in the column headings; incomplete runs and Chirp's one split-request
recovery are identified in the Clips column.

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
speech-word scoring. These five systems use local inference; the additional 24 model IDs
use OpenRouter. Check each provider's model license and access terms separately.

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
| Precision | Local: BF16, no quantization; hosted: provider-defined; no oracle speakers or reference text |
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

The [single results table](results/openrouter-2026-09-27/README.md) contains all 29 systems,
sorted by WER. WER and coverage use 94 clips. Speaker metrics use the same 89 clips where
both tested diarizing API routes supplied speaker labels, including for the local systems.
The [full JSON](results/openrouter-2026-09-27/consolidated.json) also retains each system's
full-dataset metrics. Streaming has no native speech timestamps, so its DER and tcpWER
remain unavailable. The historical 25-window local run is retained in
[the archive](docs/RESULTS.md) and is excluded from the current comparison.

## Findings and limitations

- MAI Transcribe 2 had the lowest WER (26.10%), followed by MAI Transcribe 1.5
  (26.54%) and MOSS (30.92%). MOSS led the speaker error metrics on the shared
  89-clip subset: cpWER 27.76%, tcpWER 28.66%, and DER 25.31%.
- All five local systems returned valid output on all 94 clips in one pass each.
- Transcription accuracy, speaker attribution, and diarization timing are distinct measures;
  their winners need not agree. Missing metric cells mean the tested output could not support
  that score, rather than proving that every upstream route lacks the capability.
- Four DeepInfra routes stopped early. Their missing transcripts count as deletions, and
  they receive no medals. Chirp completed after one clip was processed in two halves;
  its recovery is retained explicitly in the published records.
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

The verifier recomputes all 2,726 scores from saved predictions and references and checks
that the local and API clip hashes and references match. [Running the models](docs/RUNNING.md)
covers environments, access
terms, downloads, the exact four-meeting recipe, and the simple adapter interface.

- [CSV summary](results/openrouter-2026-09-27/consolidated.csv) and
  [complete JSON comparison](results/openrouter-2026-09-27/consolidated.json)
- [Original 25-window archive](results/ami-4090-2026-09-27/README.md)
- [Pinned model revisions](model-revisions.json) and [upstream source revisions](upstream-revisions.json)
- [Security and publication scope](SECURITY.md)

## Licenses and attribution

Original harness code and documentation are [MIT licensed](LICENSE). AMI-derived references
and transcript material retain their separate [CC BY 4.0 attribution and terms](DATA_LICENSE.md).
Model weights and source audio are not redistributed. Each upstream model and library retains
its own license; this repository's license does not override those terms.
