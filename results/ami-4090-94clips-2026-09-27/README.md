# Local models on the matched 94-clip benchmark

All five local systems completed one pass on the exact 94 AMI clips used for the OpenRouter comparison: 470 valid predictions, with no repeated trials. Audio hashes, clip IDs, durations, reference transcripts, and speaker activity match the published OpenRouter manifest. Windows are at most 60 seconds, totaling 5,536.704 seconds.

The [single consolidated table](../openrouter-validated-2026-09-27/README.md) combines these five systems with seven additional local pipelines and 20 hosted routes, sorted by word error rate. Its speaker metrics use the same 77 labelled clips for every scored system. The [full summary](summary.json) retains scores over all 94 clips; those speaker scores therefore differ from the common-subset table.

Models ran sequentially on one RTX 4090 using the [published configuration](../../configs/five-systems-openrouter.json). VibeVoice Streaming received audio without artificial real-time waits; its timing measures batch throughput. It does not provide the word timestamps required for the timed speaker metrics.

[Sources and provenance](sources.json) record model fingerprints, input provenance, scoring settings, and original score-file hashes. The `scores/` directory includes predictions and per-clip scores; `environments/` records package versions, and `workers/` records successful worker completion. `checksums.json` covers this bundle. The earlier 25-window local run remains archived and is excluded from the current comparison.

Audio is not redistributed. See the [data license](../../DATA_LICENSE.md) and [reproduction instructions](../../docs/RUNNING.md).
