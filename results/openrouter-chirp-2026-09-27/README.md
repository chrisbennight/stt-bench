# Chirp 3 completion and recovery

Google Chirp 3 through OpenRouter completed **94/94 clips**, covering 92.28 minutes
of AMI meeting audio. Word error rate is **38.68%**. This JSON transcription route
returned no speaker labels or native timestamps, so speaker and timing metrics are
unavailable.

| Measure | Result |
|---|---:|
| WER, all reference words included | 38.68% |
| Completed clips | 94/94 |
| Cost reported for selected successful responses | $1.4768 |

This is a recovered run: 65 predictions came from the original pass and 28 from
isolated recovery requests. The final clip, `EN2002a-6720000`, repeatedly timed out
as a single 60-second request, including an HTTP 504 after 181 seconds. Lossless FLAC
also timed out. Two consecutive, nonoverlapping 30-second WAV requests succeeded in
2.92 and 4.57 seconds; their transcripts were concatenated in audio order. All samples
were retained without resampling or signal changes. No references were sent to the API.

**One of 94 clips therefore has a different context window.** Label this as a recovery
variant when comparing models; it is not a uniform 60-second protocol. Successful
predictions were never rerun or selected by score. The split clip's component responses
are retained in its prediction metadata. The original failed outputs remain in the
[recovery history](recovery-history.json).

[Summary](summary.json), [CSV](summary.csv), [per-clip scores](google--chirp-3/scores.json),
[source hashes](merge-provenance.json), and [verification](verification.json) accompany
the predictions. Every saved score was independently recomputed with no mismatches.
Costs and inference times in the summary cover successful selected responses; they
exclude failed requests, which may have been billed, and the time spent recovering.
They do not describe end-to-end recovery latency.

The manifest contains references and audio hashes, with portable placeholder audio
paths. Audio is not redistributed here. See the repository's
[data attribution](../../DATA_LICENSE.md) for provenance and licensing.
Scores can be verified without downloading audio:

```bash
uv run python scripts/verify_scores.py results/openrouter-chirp-2026-09-27 \
  --output /tmp/chirp-score-verification.json
```

OpenRouter's [transcription documentation](https://openrouter.ai/docs/guides/overview/multimodal/stt)
recommends shorter audio segments for processing timeouts. The exact upstream cause
of this clip's timeout is unknown; the observed HTTP status and successful split
recovery support the workaround, not a claim about Google's internal failure.
