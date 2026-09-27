# Research and benchmark selection

Research dates: 26–27 September 2026. Implementation details were checked against official
repositories. The source revisions and model checkpoint revisions are stored
in `upstream-revisions.json` and `model-revisions.json`.

## Framework decision

[MeetEval](https://github.com/fgnt/meeteval) already implements cpWER, tcpWER and standard
interchange formats. [pyannote.metrics](https://github.com/pyannote/pyannote-metrics) supplies
duration-based DER including overlap and configurable collars. Both are published, established
research packages. Reusing their scorers avoids inventing approximate speaker matching.

[SURE](https://arxiv.org/html/2605.30899) is a broader speech-understanding framework with
scenario suites and joint-versus-pipeline comparisons. Its paper links an anonymized evaluation
repository and itself uses MeetEval for speaker-attributed transcription. A small adapter layer
over MeetEval is a more direct fit for this five-system comparison; this is not a claim that
SURE cannot support it.

## Dataset sources

- [AMI official downloads](https://groups.inf.ed.ac.uk/ami/download/): manual annotation
  release 1.6.2 and Array1-01 audio. [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).
  Attribution: AMI Meeting Corpus, AMI consortium; see Carletta et al., *The AMI Meeting Corpus:
  A Pre-announcement*. Data distributed under CC BY 4.0.
- [pyannote AMI setup](https://github.com/pyannote/AMI-diarization-setup): test recording list
  and `only_words` speaker-activity references, pinned to a Git revision. This harness's
  240-second window recipe is a custom comparison protocol, not a reproduction of every AMI paper.
- [NOTSOFAR dataset](https://huggingface.co/datasets/microsoft/NOTSOFAR) and
  [official challenge implementation](https://github.com/microsoft/NOTSOFAR1-Challenge):
  useful additional far-field meetings with speaker-attributed timing references. Select
  held-out single-channel recordings and preserve official split identities. Approximately
  six-minute meetings make many recordings suitable for the shared short-session track;
  verify durations instead of assuming all fit the streaming limit.

AMI is common in model training/evaluation histories. Add NOTSOFAR and personally representative
held-out recordings before drawing broad conclusions about real-world accuracy.

## Adapter sources

- [MOSS inference helper](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/moss_transcribe_diarize/inference_utils.py)
- [VibeVoice offline example](https://github.com/microsoft/VibeVoice/blob/main/demo/vibevoice_asr_inference_from_file.py)
- [VibeVoice streaming server](https://github.com/microsoft/VibeVoice/blob/main/demo/vibevoice_asr_streaming_fastapi_demo.py)
- [Streaming output format and limitations](https://arxiv.org/html/2609.02812v2)
- [Qwen official inference](https://github.com/QwenLM/Qwen3-ASR)
- [Community-1 output and offline loading](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Nemotron 3 NeMo loading and offline configuration](https://huggingface.co/nvidia/Nemotron-3-Diarization)

The streaming file example pre-encodes all audio chunks before generating text. This adapter
instead uses `init_streaming_state`, `encode_speech` on the current bounded window, and
`streaming_generate_step`, following the live server. This matters for meaningful first-output
latency measurements.

## Existing comparisons worth retaining

- [G-STAR revision 2](https://arxiv.org/html/2603.10468v2): compares joint recognition with
  FireRedASR, forced alignment and Sortformer combined after inference.
- [TagSpeech](https://arxiv.org/html/2601.06896v1): includes Whisper-large-v3 + pyannote 3.1.
- [TranscrIA field evaluation](https://github.com/Martossien/transcria/blob/main/docs/STT_BENCHMARK_REAL_MEETINGS.md):
  includes Qwen and joint models but lacks cpWER and mixes explicitly different serving paths.

No published controlled result for all five requested systems under one protocol was located.
This harness is intended to produce that comparison locally, rather than combine incomparable
numbers from those papers into a synthetic leaderboard.
