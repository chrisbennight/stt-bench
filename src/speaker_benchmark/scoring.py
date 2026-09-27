"""MeetEval transcription scoring and pyannote duration-based diarization scoring."""

import unicodedata

from speaker_benchmark.schema import Prediction


def normalize(text):
    # Keep fillers and repetitions. Do not rewrite numbers or use an LLM to edit references.
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return " ".join(
        "".join(
            ch if not unicodedata.category(ch).startswith("P") or ch == "'" else " " for ch in text
        ).split()
    )


def seglst(segments):
    rows = []
    for s in segments:
        row = {"session_id": "recording", "speaker": s.speaker, "words": normalize(s.text)}
        if s.start is not None:
            row.update(start_time=s.start, end_time=s.end)
        rows.append(row)
    return rows or [
        {"session_id": "recording", "speaker": "empty", "words": "", "start_time": 0, "end_time": 0}
    ]


def error_counts(result):
    return {
        key: int(getattr(result, key))
        for key in ("errors", "length", "insertions", "deletions", "substitutions")
    }


def annotation(segments):
    from pyannote.core import Annotation, Segment

    result = Annotation(uri="recording")
    for index, s in enumerate(segments):
        if s.start is None:
            raise ValueError("DER requires timed speaker activity")
        result[Segment(s.start, s.end), index] = s.speaker
    return result


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def covered_speech(reference, hypothesis):
    """Temporal coverage is diagnostic; hallucinated spans can inflate it."""
    ref = union((s.start, s.end) for s in reference)
    hyp = union((s.start, s.end) for s in hypothesis)
    total = sum(b - a for a, b in ref)
    covered = sum(max(0, min(b, d) - max(a, c)) for a, b in ref for c, d in hyp)
    return {
        "covered_seconds": covered,
        "reference_seconds": total,
        "ratio": covered / total if total else None,
    }


def score_record(record, prediction: Prediction, tcp_collar=5.0, der_collar=0.0):
    import meeteval
    from meeteval.wer.wer.siso import siso_word_error_rate
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref, hyp = seglst(record["reference"]), seglst(prediction.segments)
    speakers_available = prediction.metadata.get("speaker_labels_available", True)
    cp = (
        meeteval.wer.cpwer(ref, hyp, hypothesis_sort=False)["recording"]
        if speakers_available else None
    )
    # Chronological concatenation is intentionally secondary: overlap has no unique word order.
    ordered_ref = sorted(record["reference"], key=lambda s: (s.start, s.end))
    ordered_hyp = prediction.segments
    timed = prediction.timing != "unavailable" and all(s.start is not None for s in ordered_hyp)
    if timed:
        ordered_hyp = sorted(ordered_hyp, key=lambda s: (s.start, s.end))
    wer = siso_word_error_rate(
        " ".join(normalize(s.text) for s in ordered_ref),
        " ".join(normalize(s.text) for s in ordered_hyp),
    )
    output = {
        "wer": error_counts(wer),
        "cpwer": error_counts(cp) if cp is not None else None,
        "tcpwer": None,
        "der": None,
        "coverage": None,
        "timing": prediction.timing,
        "predicted_speakers": (
            len({s.speaker for s in prediction.segments}) if speakers_available else None
        ),
        "reference_speakers": len({s.speaker for s in record["reference"]}),
    }
    activity = prediction.activity if prediction.activity is not None else prediction.segments
    activity_timed = bool(activity) and all(s.start is not None for s in activity)
    if timed or activity_timed:
        output["coverage"] = covered_speech(record["reference_activity"], activity)
    if timed and speakers_available:
        tcp = meeteval.wer.tcpwer(ref, hyp, collar=tcp_collar)["recording"]
        output["tcpwer"] = error_counts(tcp)
        metric = DiarizationErrorRate(collar=der_collar, skip_overlap=False)
        details = metric(
            annotation(record["reference_activity"]),
            annotation(activity),
            uem=Timeline([Segment(0, record["duration"])]),
            detailed=True,
        )
        output["der"] = {str(k): float(v) for k, v in details.items()}
    return output
