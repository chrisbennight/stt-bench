"""Create an isolated inference environment using verified upstream packages."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIMES = ("moss", "vibevoice", "qwen_pyannote", "qwen_nemotron")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=RUNTIMES)
    args = parser.parse_args()
    revisions = json.loads((ROOT / "upstream-revisions.json").read_text())
    requirements = {
        "moss": [
            "torch>=2.8",
            "torchaudio>=2.8",
            "accelerate",
            "moss-transcribe-diarize @ git+https://github.com/OpenMOSS/"
            f"MOSS-Transcribe-Diarize@{revisions['OpenMOSS/MOSS-Transcribe-Diarize']}",
        ],
        "vibevoice": [
            "vibevoice @ git+https://github.com/microsoft/"
            f"VibeVoice@{revisions['microsoft/VibeVoice']}"
        ],
        "qwen_pyannote": ["qwen-asr==0.0.6", "pyannote.audio==4.0.7"],
        "qwen_nemotron": [
            "qwen-asr==0.0.6",
            "nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/"
            f"Speech@{revisions['NVIDIA-NeMo/Speech']}",
        ],
    }
    environment = ROOT / f".venv-{args.runtime}"
    if environment.exists():
        raise FileExistsError("Environment already exists; it will not be overwritten")
    subprocess.run(["uv", "venv", "--python", "3.12", str(environment)], check=True)
    python = environment / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            str(ROOT),
            *requirements[args.runtime],
        ],
        check=True,
    )
    # Capture resolved dependencies without exposing source URLs that could contain credentials.
    subprocess.run([str(python), str(ROOT / "scripts" / "record_runtime.py")], check=True)


if __name__ == "__main__":
    main()
