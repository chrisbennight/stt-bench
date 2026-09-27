"""Validated, model-independent records and predictions."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("IDs must contain only letters, digits, dots, underscores and hyphens")
    return value


def number(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Times must be finite and nonnegative")
    return value


@dataclass
class Segment:
    speaker: str
    text: str
    start: float | None = None
    end: float | None = None

    def __post_init__(self):
        if not isinstance(self.speaker, str) or not self.speaker:
            raise ValueError("A speaker label is required")
        if not isinstance(self.text, str):
            raise ValueError("Segment text must be a string")
        if (self.start is None) != (self.end is None):
            raise ValueError("Both timestamps must be supplied or both omitted")
        if self.start is not None:
            self.start, self.end = number(self.start), number(self.end)
            if self.end <= self.start:
                raise ValueError("Segment end must follow start")


@dataclass
class Prediction:
    segments: list[Segment]
    # Separate activity intervals preserve overlap in pipeline DER scoring.
    activity: list[Segment] | None = None
    timing: str = "native"
    events: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            segments=[Segment(**s) for s in data["segments"]],
            activity=None
            if data.get("activity") is None
            else [Segment(**s) for s in data["activity"]],
            timing=data.get("timing", "native"),
            events=data.get("events", []),
            metadata=data.get("metadata", {}),
        )


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest(path):
    """References remain in the controller, never in an adapter's input."""
    import soundfile as sf

    path = Path(path).resolve()
    records, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = safe_id(row["id"])
        if identifier in seen:
            raise ValueError(f"Duplicate recording ID: {identifier}")
        seen.add(identifier)
        audio = (path.parent / row["audio"]).resolve()
        info = sf.info(audio)
        if info.channels != 1:
            raise ValueError(f"{identifier}: prepare a single audio channel explicitly")
        if info.duration <= 0:
            raise ValueError(f"{identifier}: empty audio")
        refs = [Segment(**s) for s in row["reference"]]
        activity = [Segment(**s) for s in row.get("reference_activity", row["reference"])]
        for seg in refs + activity:
            if seg.start is None or seg.end > info.duration + 0.05:
                raise ValueError(f"{identifier}: invalid reference timing")
        row.update(
            audio=str(audio),
            duration=info.duration,
            reference=refs,
            reference_activity=activity,
            audio_sha256=digest(audio),
        )
        records.append(row)
    if not records:
        raise ValueError("Manifest contains no recordings")
    return records
