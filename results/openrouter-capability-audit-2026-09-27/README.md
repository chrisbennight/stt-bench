# Saved-output audit

This bundle audits the original 24 hosted results and corrects Fish Pro's text parsing.
It contains **no new inference requests and no additional API spend**. The original
responses, timings, and costs remain preserved; this is not a new model run.

[Audit counts](audit.json) record the number of successful clips with each metric and
the source score-file hashes. The [corrected Fish scores](fish-reparsed-scores.json) and
[summary](fish-reparsed-summary.json) use the original 94 responses, recognizing the
provider's inline speaker labels and excluding non-speech annotations from word scoring.
WER changes from 69.81% to **31.54%**; full-dataset cpWER is **43.20%**. Four responses
contain an unfinished final annotation, explicitly flagged in the corrected metadata.
No timing is invented, so these saved responses still cannot supply tcpWER or DER.

These corrected scores have not been substituted into the provisional comparison table
while the full hosted configuration is under review. The table's speaker metrics use an
89-clip subset, so the full-dataset cpWER here must not be pasted into that column.

See the [capability audit and validation procedure](../../docs/OPENROUTER_CAPABILITY_AUDIT.md),
[Fish output documentation](https://docs.fish.audio/features/speech-to-text), and
[data license](../../DATA_LICENSE.md). Reproduce with
`uv run python scripts/audit_openrouter_outputs.py --output runs/openrouter-output-audit`.
