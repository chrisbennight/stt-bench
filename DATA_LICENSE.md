# Dataset attribution and license scope

The software license in `LICENSE` covers the original harness code and documentation.
It does not relicense third-party data, model weights, or upstream software.

## AMI Meeting Corpus

The checked-in reference transcripts and speaker-activity annotations derive from the
**AMI Meeting Corpus**, created by the **AMI consortium**. Credit: Jean Carletta et al.,
*The AMI Meeting Corpus: A Pre-announcement*. See the
[official corpus site](https://groups.inf.ed.ac.uk/ami/corpus/),
[downloads and annotation release 1.6.2](https://groups.inf.ed.ac.uk/ami/download/), and
[official license notice](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).

The corpus and annotations are released under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Retain that attribution and license link when redistributing the AMI-derived material.
This repository does not imply endorsement by the corpus creators or model providers.

Speaker-activity references use the pinned
[pyannote AMI diarization setup](https://github.com/pyannote/AMI-diarization-setup).
Exact source revisions and source-file hashes are recorded in the result manifests and
`upstream-revisions.json`.

Changes made for this evaluation: four test meetings were selected; references were divided
into nonoverlapping windows of at most 240 seconds; boundary words were assigned by midpoint;
activity and word intervals were clipped to the window; punctuation-only and untimed XML
elements were omitted. Text normalization is applied during scoring. Original reference words,
window-relative times, and source hashes are retained in the published manifests.

Model-generated hypotheses reproduce or transform speech from those meetings. They are
included for inspection and score verification, with this same dataset attribution. Treat
AMI-derived transcript material in `results/` as data under the corpus terms, not as original
MIT-licensed prose. Source audio and model weights are not included.

## Models and libraries

Model and library licenses remain with their respective upstream projects. Links are in
[the README](README.md) and [References](docs/REFERENCES.md); exact revisions are pinned.
Downloading a gated model may require accepting its provider's terms. Those access requirements
are separate from the license for this harness.
