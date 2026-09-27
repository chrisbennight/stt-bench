"""Record package versions without credentials or environment variable values."""

import importlib.metadata
import json
import sys
from pathlib import Path

packages = {
    d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]
}
(Path(sys.prefix) / "resolved-packages.json").write_text(json.dumps(packages, indent=2) + "\n")
