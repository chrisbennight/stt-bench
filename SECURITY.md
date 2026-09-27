# Security and publication scope

The public repository contains benchmark code, documentation, public-dataset references,
model predictions, scores, and runtime provenance. It does not contain model weights, source
audio, access tokens, SSH configuration, private host names, or machine-specific home paths.
Archived machine paths use `${BENCHMARK_ROOT}` and `${MODEL_CACHE}`; manifest audio paths are repository-relative.
Publication changed filesystem metadata, not predicted text or measured numbers. See the
[export record](results/ami-4090-2026-09-27/publication.json).

Preparation downloads public evaluation data and pinned model snapshots. Local adapters run
in separate Python environments. Workers set Hugging Face and Transformers offline mode
and disable pyannote metrics telemetry; these settings are not an operating-system network sandbox.

The OpenRouter adapter sends audio to a paid external transcription service. It requires
`OPENROUTER_API_KEY` through the runtime environment and `OPENROUTER_ALLOW_PAID_REQUESTS=1`.
Configuration limits requests and audio duration, not dollars; the provider-enforced credit
limit is the spending boundary. The controller validates the manifest before requests, and
workers do not receive reference transcripts or speaker identities. See
[hosted execution and costs](docs/OPENROUTER.md). Verifying published scores requires no
credentials and makes no inference requests.

Generated text is parsed as JSON or a defined text format, not evaluated as Python or shell
code. Subprocesses use argument arrays. Recording IDs are validated before use in output
paths. Dataset XML containing DTD or entity declarations is rejected. The optional gated
downloader reads a token through a hidden prompt, does not save it, checks destination path
containment, and removes authorization headers on cross-host redirects. Normal Hugging Face
authentication may instead use an existing local credential cache; never commit that cache.

Package installation and model loading execute third-party code. MOSS uses its custom model
implementation; NeMo loads an official checkpoint. Pinned revisions make inputs traceable,
but do not constitute a complete dependency or checkpoint audit. Separate environments resolve
dependency conflicts; they do not isolate host files or privileges.

Raw predictions are intentionally retained, including malformed output, so failures can be
audited. New runs using private recordings will contain private transcript data. Generated
audio, model directories, local runs, environment files, and credential files are ignored by
Git. The publication check detects common token formats and private path/address patterns;
it is an additional check, not a comprehensive security guarantee.

To report a problem, open a repository issue with a minimal reproduction and redacted metadata.
Do not include real credentials, private recordings, or private transcripts.
