# Local counterparts of the hosted models

This extension runs the downloadable counterparts of the OpenRouter models on the
same 94 AMI clips, with pyannote Community-1 providing speaker diarization.
It adds seven pipelines. The completed Qwen3-ASR-1.7B + pyannote result already uses
these clips and is reused rather than run a second time.

All seven additional pipelines completed the 94-clip pass on the RTX 4090. Five returned
94 valid outputs; Voxtral Mini returned 92 and Voxtral Small NF4 returned 93. Results are in the
[single consolidated table](../results/openrouter-validated-2026-09-27/README.md);
the original Qwen 1.7B + pyannote row remains the comparison for that model.

The two existing Qwen pipelines use [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
and [Qwen forced alignment](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), with either
[pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) or
[Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization). Together with
the seven pipelines below, these make up the nine local ASR-plus-diarizer systems.

| Pipeline | Transcription weights | Precision |
|---|---|---|
| Parakeet + pyannote | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | NeMo checkpoint default |
| Nemotron ASR + pyannote | [Nemotron 3.5 ASR 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | NeMo checkpoint default |
| Whisper + pyannote | [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | BF16 |
| Whisper Turbo + pyannote | [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | BF16 |
| Qwen Small + pyannote | [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | BF16 |
| Voxtral Mini + pyannote | [Voxtral Mini 3B 2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | BF16 |
| Voxtral Small + pyannote | [Voxtral Small 24B 2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | NF4, BF16 computation |

The Qwen and Voxtral models include the four failed DeepInfra routes. Those hosted
runs do not provide complete comparison baselines. The newer hosted Voxtral
Transcribe and Qwen Flash models are different releases.

## Pipeline and comparability

- **Transcription:** each model receives the full original clip. Whisper uses its
  sequential long-form implementation. Qwen retains the existing internal 30-second
  ASR chunks. NeMo runs offline transcription; this is not a real-time latency test.
- **Word timestamps:** the new Whisper, Voxtral, and NeMo pipelines use
  [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) on the
  transcript, without reference text. Alignment must preserve every normalized word.
  Raw transcription is retained for audit. Point timestamps remain points; valid
  spans are intersected with the audio window, with the number changed recorded.
- **Speaker labels:** [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  processes the entire clip. Exclusive activity assigns words by greatest overlap;
  regular activity preserves overlapping speakers for DER. Unmatched words remain
  explicitly unassigned. No reference speaker count is supplied.
- **Metrics:** WER, cpWER, tcpWER, and DER use the existing scorer and
  collars. Runtime includes transcription, alignment, and diarization; downloads
  and model loading are excluded. These are pipeline results, not claims that the
  transcription weights natively produce speaker labels.
  Identical diarization scores across successful pipelines are expected because
  they share the same diarizer; cpWER and tcpWER also depend on transcription and
  word alignment. Token-limit failures count as empty hypotheses under the existing
  protocol, retain their raw generation, and remain in the denominator.
- **Memory:** run models sequentially on one GPU. Voxtral Small uses
  [bitsandbytes NF4](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
  because its documented BF16 requirement exceeds 24GB. Its name must retain the
  quantization label in published results.

## Preparation and execution

The configuration is [local-openweights.json](../configs/local-openweights.json).
Reuse the existing gated pyannote snapshot after accepting its upstream terms.
Credentials belong only in the authorized download environment, never in configs.

```bash
python scripts/setup_runtime.py local_asr
python scripts/setup_runtime.py local_nemo
.venv-local_asr/bin/python scripts/fetch_models.py whisper whisper_turbo qwen_small voxtral_mini voxtral_small
.venv-local_nemo/bin/python scripts/fetch_models.py parakeet nemotron_asr
```

The sequential controller checks the first clip from each model for nonempty,
timed, speaker-labelled output and all four accuracy metrics. It retains
that first result and runs the remaining clips once. It verifies audio hashes and
references against the baseline before starting and refuses to overwrite a run:

```bash
uv run python scripts/run_local_preflight.py \
  --config configs/local-openweights.json \
  --manifest data/ami-openrouter/manifest.jsonl \
  --baseline-manifest runs/local94/manifest.jsonl \
  --output runs/local-openweights
```

The baseline manifest must contain the recorded audio hashes and point to available
audio files. Model loading is repeated after the retained preflight and excluded
from inference timing. Each recording has the same deterministic seed in both phases.

## Compatibility fixes and retained failures

- **Nemotron language input:** the pinned NeMo Lhotse loader requires `lang` in
  each manifest record and `prompt_mode=langID` for deterministic language-conditioned
  inference. Its ordinary file-list path lost the requested language and defaulted
  to a training-time random prompt mode. The adapter supplies both fields through
  a temporary manifest, without patching NeMo or substituting another checkpoint.
- **Whisper currency symbols:** the forced aligner strips symbols attached to numbers,
  such as the currency sign in `£17`. The adapter proves a one-to-one token match and
  restores the original token at its measured timestamp. Six rejected Whisper outputs
  were recovered from their saved transcripts; no ASR inference was repeated. Recovery
  time and the original attempts are retained.
- **Voxtral repetition:** Mini produced two repeating generations and Small NF4 produced
  one. Each reached the 4,096-token limit. These model outputs were retained as failures,
  without retries to select a better transcription.
