"""Adapters load dependencies lazily and receive audio, never reference labels."""

from importlib.metadata import entry_points

from speaker_benchmark.adapters.joint import Moss, VibeVoice, VibeVoiceStreaming
from speaker_benchmark.adapters.local_asr import NemoPyannote, VoxtralPyannote, WhisperPyannote
from speaker_benchmark.adapters.openrouter import OpenRouter
from speaker_benchmark.adapters.pipeline import QwenNemotron, QwenPyannote

BUILTINS = {
    "moss": Moss,
    "vibevoice": VibeVoice,
    "vibevoice_streaming": VibeVoiceStreaming,
    "qwen_pyannote": QwenPyannote,
    "qwen_nemotron": QwenNemotron,
    "openrouter": OpenRouter,
    "whisper_pyannote": WhisperPyannote,
    "voxtral_pyannote": VoxtralPyannote,
    "nemo_pyannote": NemoPyannote,
}


def create_adapter(name, options):
    if name in BUILTINS:
        return BUILTINS[name](options)
    matches = list(entry_points(group="speaker_benchmark.adapters", name=name))
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous adapter: {name}")
    return matches[0].load()(options)
