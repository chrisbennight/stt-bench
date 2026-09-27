# Evaluation protocol

Local inference uses one RTX 4090 with 24 GB of memory, one model process at a time,
and batch size one where the upstream API provides that control. Loading and downloads
are excluded from inference time. There is no warmup: each system's first window includes
cold inference effects. CUDA allocated memory includes the loaded model; it is not total
device memory. The GPU had roughly 1.5 GB of other GPU memory allocated at inspection.

VibeVoice samples acoustic features even with greedy text decoding. The final run therefore
sets Python, NumPy, and PyTorch random seeds independently before every recording, using
the first four SHA-256 bytes of its recording ID as a big-endian unsigned integer. Each
prediction records the seed. This makes random inputs independent of window ordering and
previous failures; it does not promise bitwise equality across hardware or library versions.
Each local system runs once under this policy on the 94-clip API manifest.

## Shared audio

The evaluation contains AMI test meetings ES2004a, IS1009a, TS3003a, and EN2002a, using
Array1-01 single-channel distant-microphone audio. These are four distinct meeting groups,
selected before comparing scores. Their combined duration is 5,536.704 seconds, divided
into 94 nonoverlapping windows of at most 60 seconds. All systems receive identical
windows. Speaker labels restart independently for each window.

References come from AMI manual annotation release 1.6.2 and the pinned pyannote AMI
only-words RTTM test references. Transcript words belong to the window containing their
midpoint; boundary timings are clipped. The manifest records source hashes and offsets.
The controller holds the references; model workers receive only audio paths, duration,
and language. No reference transcripts, speaker counts, or identities are given to models.

This is a controlled comparison on a small English meeting sample, not the OpenASR
leaderboard or a claim about all speech domains. AMI may occur in model training or
development data. The windows from one meeting are correlated; they are not 94 independent
experiments. Per-meeting results accompany aggregate scores.

## Scoring

The consolidated table sorts by WER, available for every tested transcription route.
For speaker-attributed transcription, concatenated minimum-permutation WER (cpWER)
accounts for both transcript words and their speaker attribution while allowing
anonymous speaker labels to be renamed. Lower is better. Counts are summed before division,
so a short window does not receive the same weight as a long, word-rich window.

WER uses all 94 clips. The consolidated cpWER, tcpWER, and DER columns use
the same labelled subset for every system, with its size in the column headings.
This is the intersection of clips with measurable cpWER across hosted routes that
returned speaker labels: MAI 2, Deepgram, Grok, and Fish Pro. The exact clip IDs and
each system's full-dataset metrics are retained in the JSON result. Selecting this
subset after observing missing labels can favor clips that providers find easier;
these speaker scores are conditional on available output, not full-dataset estimates.
The consolidator rejects differing clip counts, audio hashes, boundaries, or references.
The original 25-window local evaluation is historical and is not included in this table.

Time-constrained cpWER uses a five-second collar. Native segment outputs use MeetEval's
pseudo-word timing; Qwen uses forced alignment. These are different timestamp sources,
so the timing-constrained score is supplementary. DER uses no collar and includes overlap,
over the entire audio duration. Pipeline DER uses the diarizer's regular overlapping
activity, not only the speaker-assigned words.

Text normalization applies Unicode NFKC, lowercasing, and punctuation-to-space conversion,
preserving apostrophes, fillers, repetitions, and number spellings. No language model edits
references or hypotheses. VibeVoice's square-bracket sound annotations are preserved in
raw output but excluded from speech-word scoring. Fish inline speaker markers are parsed
as labels, and its bracketed event annotations are likewise excluded from speech words.
Fish's unlabelled timed segments are not assigned to its
inline speakers by guessing word positions; tcpWER and DER need actual speaker times.
Ordinary unattributed speech is retained
as `unassigned`, so missing speaker labels are not silently discarded.

All hosted routes use the same timestamp boundary policy. Intervals crossing the audio
boundary are intersected with it, with original values and adjustments retained.
Words entirely after the audio remain in lexical scoring as potential insertions;
DER is bounded by the actual audio duration. Equal start/end timestamps are valid point
alignments: their words remain scored, but they add no speech duration. Reversed and
non-finite timestamps remain invalid. When word intervals are reversed, valid native
segments from the same response may be used instead, provided this does not discard
speaker labels. Raw word intervals and the selected alignment source remain recorded.
No timestamp is inferred from the reference.

Both Qwen pipelines use identical 30-second ASR/alignment chunks and whole-window
diarization. Words are assigned by greatest overlap with speaker activity; no intersection
becomes `unassigned`. This is a specific reproducible pipeline, not a claim that its word
assignment is optimal. The historical 25-window runs produced identical raw ASR text.
An exploratory audit found that the upstream aligner can join adjacent word tokens, for
example `pop-up` becoming `popup`. Scoring retains the pipeline's emitted aligned words;
the original ASR text is also saved. This normalization difference is a limitation, not
a silent transcript fix.

## Streaming and validation

VibeVoice Streaming provides anonymous speaker labels and text but no speech timestamps.
Native DER and time-constrained cpWER are therefore unavailable.
Paced runs supply chunks according to their audio availability, including lookahead.
First-text time includes initial silence and buffering; it is not a speech-onset latency.
Chunk emission delay is measured against each chunk's audio end, not individual word ends.
The current comparison processes the supplied audio without artificial real-time waits,
matching the API batch workload. It measures throughput, not live-stream latency; the
historical paced run remains archived separately.

Streaming output can split a word, speaker marker, or sound annotation across chunks.
The initial adapter parsed chunks independently; an observed `[Environmental Sounds]`
annotation split across chunks incorrectly became transcript words and an early speech event.
The corrected parser joins the exact emitted text before interpreting its format. It masks
annotations and speaker markers without shifting character offsets, so each remaining speech
character keeps its measured emission event. First-text time excludes annotation fragments.
This classification uses the completed output; it is not a measurement of a live UI's
buffering policy for incomplete markers.

`scripts/reparse_streaming_run.py` applies that correction to a separate copy of a finished
run and recomputes its scores. The original raw run is preserved. Raw text, random seeds,
model outputs, compute times, wall times, and event timestamps do not change. The derived
directory records source hashes and per-record changes in `reprocessing.json`. The original
inference source is retained in the [result provenance](../results/ami-4090-2026-09-27/provenance/inference-source/).
Current adapter code includes the parser correction for future runs. This is a parsing correction, not new inference
or a trial selected for a better model score.

Initial VibeVoice attempts exposed adapter failures on valid sound annotations.
Corrected runs use new output directories, and the comparison tool
requires explicit selection of a replacement run. It never quietly chooses the best score.
Exploratory inference attempts are not part of the public result bundle. The published
original and corrected results come from the same final seeded trial; no scores from
other trials are mixed into the comparison.

An otherwise completed generation with invalid structured output is a recorded window
failure. Its raw text, token-limit flag, elapsed time, and CUDA memory are retained, and
the worker continues to the next window. Scoring treats that window as an empty hypothesis
and reports the failure count explicitly. Attempted-window runtime includes these failures;
successful-window runtime is also reported separately. CUDA errors and other unexpected
exceptions still stop the worker. Unattempted windows prevent a completed comparison.
