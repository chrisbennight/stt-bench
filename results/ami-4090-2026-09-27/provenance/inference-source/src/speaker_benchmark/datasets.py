"""AMI preparation and a portable SegLST importer for other meeting corpora."""

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict
from pathlib import Path

from speaker_benchmark.schema import Segment, digest, read_json, safe_id, write_json

AMI_TEST = """IS1009a IS1009b IS1009c IS1009d ES2004a ES2004b ES2004c ES2004d
TS3003a TS3003b TS3003c TS3003d EN2002a EN2002b EN2002c EN2002d""".split()
ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


def download(url, target):
    """Stream a public dataset asset; incomplete files never masquerade as complete downloads."""
    target = Path(target)
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    temporary.replace(target)


def fetch_ami(destination, meetings):
    destination = Path(destination)
    for meeting in meetings:
        if meeting not in AMI_TEST:
            raise ValueError("This recipe only downloads the documented AMI test meetings")
    destination.mkdir(parents=True, exist_ok=True)
    download(ANNOTATIONS_URL, destination / "annotations.zip")
    sources = [{"url": ANNOTATIONS_URL, "sha256": digest(destination / "annotations.zip")}]
    revision = Path(__file__).parents[2] / "upstream-revisions.json"
    # Packaged installs use the pinned annotation release recorded with the source distribution.
    pinned = read_json(revision)["pyannote/AMI-diarization-setup"] if revision.exists() else None
    if pinned is None:
        raise ValueError("Run dataset preparation from the editable source checkout")
    for meeting in meetings:
        assets = {
            f"{meeting}.wav": (
                "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
                f"{meeting}/audio/{meeting}.Array1-01.wav"
            ),
            f"{meeting}.rttm": (
                "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/"
                f"{pinned}/only_words/rttms/test/{meeting}.rttm"
            ),
        }
        for name, url in assets.items():
            download(url, destination / name)
            sources.append({"url": url, "sha256": digest(destination / name)})
    write_json(destination / "sources.json", sources)


def read_rttm(path):
    segments = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "SPEAKER":
            start, duration = float(fields[3]), float(fields[4])
            if duration > 0:
                segments.append(Segment(fields[7], "", start, start + duration))
    return segments


def ami_words(archive, meeting):
    words = []
    with zipfile.ZipFile(archive) as bundle:
        names = [
            n
            for n in bundle.namelist()
            if re.search(rf"(?:^|/)words/{re.escape(meeting)}\.[A-Z]\.words\.xml$", n)
        ]
        if not names:
            raise ValueError(f"No word annotations for {meeting}")
        for name in sorted(names):
            speaker = Path(name).name.split(".")[1]
            if bundle.getinfo(name).file_size > 10_000_000:
                raise ValueError("Unexpectedly large XML annotation")
            xml = bundle.read(name)
            if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
                raise ValueError("XML declarations are not allowed")
            for word in ET.fromstring(xml).iter():
                if word.tag.rsplit("}", 1)[-1] != "w" or word.get("punc") == "true":
                    continue
                if word.get("starttime") is None or word.get("endtime") is None:
                    continue
                start, end = float(word.get("starttime")), float(word.get("endtime"))
                text = "".join(word.itertext()).strip()
                if text and end > start:
                    words.append(Segment(speaker, text, start, end))
    return sorted(words, key=lambda s: (s.start, s.end, s.speaker))


def write_windows(audio_path, refs, activity, destination, identifier, window_seconds, source):
    """Midpoint assignment preserves each boundary word exactly once; timings are clipped."""
    import soundfile as sf

    destination = Path(destination)
    info = sf.info(audio_path)
    if info.channels != 1:
        raise ValueError("Prepare a single-channel signal before importing")
    window_samples = int(window_seconds * info.samplerate) if window_seconds else info.frames
    if window_samples <= 0:
        raise ValueError("Window length must be positive")
    rows = []
    for first in range(0, info.frames, window_samples):
        last = min(first + window_samples, info.frames)
        start, end = first / info.samplerate, last / info.samplerate
        name = safe_id(f"{identifier}-{first}")
        target = destination / f"{name}.wav"
        if target.exists():
            raise FileExistsError(target)
        data, rate = sf.read(audio_path, start=first, stop=last, dtype="float32")
        sf.write(target, data, rate, subtype="PCM_16")
        selected = [
            Segment(s.speaker, s.text, max(s.start, start) - start, min(s.end, end) - start)
            for s in refs
            if start <= (s.start + s.end) / 2 < end
        ]
        selected_activity = [
            Segment(s.speaker, "", max(s.start, start) - start, min(s.end, end) - start)
            for s in activity
            if s.start < end and s.end > start
        ]
        rows.append(
            {
                "id": name,
                "audio": target.name,
                "language": "English",
                "reference": [asdict(s) for s in selected],
                "reference_activity": [asdict(s) for s in selected_activity],
                "source": {
                    **source,
                    "recording": identifier,
                    "offset_seconds": start,
                    "audio_sha256": digest(audio_path),
                },
                "boundary_rule": "word midpoint; clip word/activity times to window",
            }
        )
    return rows


def prepare_ami(source, destination, meetings, window_seconds=240):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    # A separate destination per track avoids overwriting audio used by earlier runs.
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for meeting in meetings:
        if meeting not in AMI_TEST:
            raise ValueError("Meeting is not in the pinned AMI test list")
        rows.extend(
            write_windows(
                source / f"{meeting}.wav",
                ami_words(source / "annotations.zip", meeting),
                read_rttm(source / f"{meeting}.rttm"),
                destination,
                meeting,
                window_seconds,
                {
                    "corpus": "AMI",
                    "split": "test",
                    "channel": "Array1-01",
                    "annotations_sha256": digest(source / "annotations.zip"),
                    "rttm_sha256": digest(source / f"{meeting}.rttm"),
                },
            )
        )
    save_manifest(destination / "manifest.jsonl", rows)


def save_manifest(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def import_seglst(references, audio_map, destination):
    """Import CHiME/MeetEval SegLST with an explicit recording-to-audio mapping."""
    mapping = read_json(audio_map)
    refs = read_json(references)
    destination = Path(destination).resolve()
    grouped = {}
    for row in refs:
        identifier = safe_id(row["session_id"])
        grouped.setdefault(identifier, []).append(
            Segment(str(row["speaker"]), row["words"], row["start_time"], row["end_time"])
        )
    if set(mapping) != set(grouped):
        raise ValueError("Audio map and reference session IDs must match exactly")
    rows = []
    for identifier, segments in grouped.items():
        entry = mapping[identifier]
        audio = (Path(audio_map).resolve().parent / entry["audio"]).resolve()
        activity = (
            read_rttm(Path(audio_map).resolve().parent / entry["rttm"])
            if entry.get("rttm")
            else segments
        )
        rows.append(
            {
                "id": identifier,
                "audio": str(audio),
                "language": entry.get("language"),
                "reference": [asdict(s) for s in segments],
                "reference_activity": [asdict(s) for s in activity],
                "source": {
                    "reference_sha256": digest(references),
                    "activity_source": "rttm" if entry.get("rttm") else "transcript",
                },
            }
        )
    save_manifest(destination, rows)
