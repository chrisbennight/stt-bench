# Local OpenRouter counterparts on the RTX 4090

Seven additional local pipelines were evaluated on the **same 94 AMI clips and
references** as the hosted benchmark. Each uses pyannote Community-1 for diarization.
The previously completed Qwen3-ASR-1.7B + pyannote result is reused.

**Results are in the [single consolidated table](../openrouter-validated-2026-09-27/README.md).**
These are local pipeline measurements, not native diarization capabilities of the
transcription models or a reproduction of provider decoding settings.

- **Complete outputs:** Parakeet, Nemotron ASR, both Whisper variants, and Qwen 0.6B
  each produced 94/94 valid predictions.
- **Retained model failures:** Voxtral Mini produced 92/94 valid predictions; Voxtral
  Small NF4 produced 93/94. The three other generations repeated words until the
  4,096-token limit. They remain in the denominator as empty hypotheses under the
  existing failure protocol; their raw outputs are preserved in the score files.
- **Quantization:** Voxtral Small used bitsandbytes NF4 with BF16 computation. Its
  maximum measured allocated GPU memory was approximately 16.84 GiB.
- **Alignment recovery:** six Whisper transcripts were recovered from saved text
  after fixing preservation of currency symbols. Their ASR was not repeated; timing
  includes the initial attempt and recovery. Original records are in [reprocessing](reprocessing).
- **Nemotron setup:** two failed setup attempts preceded the corrected pass. The
  NeMo loader required language and inference prompt mode in the manifest. The
  [initial attempt](attempts/nemotron-initial.json) and
  [diagnostic](attempts/nemotron-diagnostic.json) are retained.

The first clip from each pipeline served as a preflight and was retained in the full
pass. Models ran sequentially. Loading and downloads are excluded from inference
timing. No additional OpenRouter requests were made for this extension.

See [sources and configuration](sources.json), [per-clip predictions and scores](scores),
[runtime versions](environments), [worker outcomes](workers), and
[the pipeline protocol and upstream model links](../../docs/LOCAL_OPENWEIGHTS.md).
The shared input is recorded in [manifest.jsonl](manifest.jsonl); hashes and references
match the prior hosted and local runs. Audio and weights are not included.

The derived AMI material retains its [attribution and license](../../DATA_LICENSE.md).
