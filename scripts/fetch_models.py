"""Explicit model preparation. Run from the corresponding inference environment."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "moss": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "vibevoice": "microsoft/VibeVoice-ASR",
    "vibevoice_tokenizer": "Qwen/Qwen2.5-7B",
    "vibevoice_streaming": "microsoft/VibeVoice-ASR-Streaming-7B",
    "qwen": "Qwen/Qwen3-ASR-1.7B",
    "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "pyannote": "pyannote/speaker-diarization-community-1",
    "nemotron": "nvidia/Nemotron-3-Diarization",
    "qwen_small": "Qwen/Qwen3-ASR-0.6B",
    "parakeet": "nvidia/parakeet-tdt-0.6b-v3",
    "nemotron_asr": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "whisper": "openai/whisper-large-v3",
    "whisper_turbo": "openai/whisper-large-v3-turbo",
    "voxtral_mini": "mistralai/Voxtral-Mini-3B-2507",
    "voxtral_small": "mistralai/Voxtral-Small-24B-2507",
}


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=ALIASES)
    args = parser.parse_args()
    revisions = json.loads((ROOT / "model-revisions.json").read_text())
    destination = ROOT / "models"
    destination.mkdir(exist_ok=True)
    for alias in args.models:
        repo = ALIASES[alias]
        extra = (
            {"allow_patterns": revisions[repo]["nemo_files"]}
            if "nemo_files" in revisions[repo] else {}
        )
        if alias in {"qwen_small", "whisper", "whisper_turbo", "voxtral_mini", "voxtral_small"}:
            extra = {
                "allow_patterns": [
                    "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja",
                    "model.safetensors", "model-*.safetensors",
                ],
                "ignore_patterns": ["*fp32*"],
            }
        if alias == "vibevoice_tokenizer":
            extra = {"allow_patterns": ["tokenizer*", "vocab.json", "merges.txt", "config.json"]}
        snapshot = Path(snapshot_download(repo, revision=revisions[repo]["revision"], **extra))
        link = destination / alias
        if link.is_symlink() and link.resolve() != snapshot.resolve():
            raise FileExistsError("Existing model link points to a different revision")
        if not link.exists():
            link.symlink_to(snapshot, target_is_directory=True)
        if alias == "pyannote":
            from pyannote.audio import Pipeline

            # Populate any nested segmentation/embedding dependencies before offline benchmarking.
            Pipeline.from_pretrained(str(link))
        print(f"Prepared {alias} at the recorded revision")


if __name__ == "__main__":
    main()
