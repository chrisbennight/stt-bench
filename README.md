# Speech transcription benchmark: OpenRouter and local RTX 4090

**32 systems, the same 94 audio clips, one [results table](results/openrouter-validated-2026-09-27/README.md).**
We compare **20 OpenRouter speech-to-text routes** and **12 local model pipelines** on
92.28 minutes of English meeting audio, measuring transcription accuracy, who said what,
processing speed, API cost, and GPU memory.

- **Best transcription in this sample:** MAI Transcribe 2, **26.10% WER**.
- **Best local transcription and speaker metrics:** MOSS, **30.92% WER**, with the
  lowest cpWER, tcpWER, and DER on the shared 77-clip speaker subset.
- **Fastest measured local pipeline:** Parakeet + pyannote, **0.011 RTF** and **33.33% WER**.

The [results page](results/openrouter-validated-2026-09-27/README.md) explains each metric
before the table, identifies **OpenRouter** or **Local 4090** for every row, and links each
model to its OpenRouter page or Hugging Face model card. It is sorted by WER, with medals
for the three lowest measured values in each metric column.

- **Benchmark data — AMI Meeting Corpus:** [Paper](https://doi.org/10.1007/11677482_3) ·
  [Dataset](https://groups.inf.ed.ac.uk/ami/corpus/) ·
  [Diarization references](https://github.com/pyannote/AMI-diarization-setup).
- **Transcription scoring — MeetEval:** [Paper](https://arxiv.org/abs/2307.11394) ·
  [GitHub](https://github.com/fgnt/meeteval). DER uses
  [pyannote.metrics](https://github.com/pyannote/pyannote-metrics).

This repository includes the Python harness, adapters, pinned model revisions, runtime
versions, saved predictions, references, scores, and verification tools. Measurements are
from **27 September 2026**. This is a small controlled meeting-audio comparison, not an
OpenASR leaderboard reproduction or a universal model ranking.

## What is being compared?

The comparison includes hosted transcription services, joint models that return words and
speaker labels, and local pipelines that add alignment and diarization to an ASR model.
Speaker labels are anonymous IDs within each clip; they do not identify a person by name.

- **OpenRouter — 20 routes:** MAI, GPT Transcribe, GPT-4o, Whisper, Gemini, Chirp,
  Deepgram, AssemblyAI, Fish, Muse Voice, Voxtral, Qwen Flash, Parakeet, Nemotron, and Grok.
  We score the output actually returned by the hosted endpoint. No external diarizer or
  forced aligner is added to hosted responses.
- **Joint local models — 3 systems:** [MOSS](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize),
  [VibeVoice offline](https://huggingface.co/microsoft/VibeVoice-ASR), and
  [VibeVoice Streaming](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B).
  Streaming returns speaker-labelled text without native speech timestamps.
- **Local ASR pipelines — 9 systems:** Qwen3-ASR 1.7B with either pyannote or Nemotron
  diarization; Qwen3-ASR 0.6B, Parakeet, Nemotron ASR, Whisper large-v3 and Turbo, and
  Voxtral Mini and Small with pyannote. They use Qwen forced alignment. Voxtral Small
  runs with NF4 quantization to fit the 4090. See the
  [pipeline components and model cards](docs/LOCAL_OPENWEIGHTS.md).

Local ASR counterparts let us compare downloadable weights with hosted routes, but serving
settings and releases can differ. Adding pyannote measures a complete local pipeline, not
native diarization in the ASR weights. Hosted Voxtral Mini Transcribe and Qwen Flash are
not the same releases as the downloadable Voxtral and Qwen models tested here.

Some models can also describe sound events. **Annotation/event accuracy is not evaluated**
here; bracketed VibeVoice sound annotations are retained in raw output but excluded from
speech-word scoring. Names ending in **+ pyannote** identify local pipelines; the hosted
routes retain their provider model names. Check each provider's model license and access terms
separately.

The scoring libraries are [MeetEval](https://github.com/fgnt/meeteval) and
[pyannote.metrics](https://pyannote.github.io/pyannote-metrics/). Model cards describe upstream
capabilities; the results table contains this repository's own measurements. More authoritative
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
- Ten local systems returned valid output on all 94 clips. Voxtral Mini returned 92/94;
  Voxtral Small NF4 returned 93/94.
  Their three repeating, token-limited generations are retained as failures and
  count in dataset error rates.
- Eight local ASR pipelines share pyannote diarization, so equal DER values largely
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
- Four failed DeepInfra routes are omitted from the results table. All four have local
  counterparts in the comparison; the earlier hosted attempts remain in the archive.
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

The verifier recomputes all **3,008 clip scores** (1,880 hosted and 1,128 local) from saved
predictions and references, and checks that local and API clip hashes and references match. [Running the models](docs/RUNNING.md)
covers environments, access terms, downloads, the shared four-meeting recipe, and the
adapter interface. Verification makes no inference calls; new hosted runs require a cost
estimate and explicit paid-request opt-in.

- [CSV summary](results/openrouter-validated-2026-09-27/consolidated.csv) and
  [complete JSON comparison](results/openrouter-validated-2026-09-27/consolidated.json)
- [Run local models](docs/RUNNING.md) and [additional local pipelines](docs/LOCAL_OPENWEIGHTS.md)
- [Run OpenRouter models and estimate costs](docs/OPENROUTER.md)
- [Evaluation protocol](docs/PROTOCOL.md) and [hosted capability audit](docs/OPENROUTER_CAPABILITY_AUDIT.md)
- [Request cost ledger](results/openrouter-validated-2026-09-27/costs.json)
- [Original 25-window archive](results/ami-4090-2026-09-27/README.md)
- [Pinned model revisions](model-revisions.json) and [upstream source revisions](upstream-revisions.json)
- [Security and publication scope](SECURITY.md)

## Licenses and attribution

Original harness code and documentation are [MIT licensed](LICENSE). AMI-derived references
and transcript material retain their separate [CC BY 4.0 attribution and terms](DATA_LICENSE.md).
Model weights and source audio are not redistributed. Each upstream model and library retains
its own license; this repository's license does not override those terms.
