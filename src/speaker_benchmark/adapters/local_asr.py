"""Open-weight transcription followed by Qwen alignment and pyannote diarization."""

import json
import math
import tempfile
import unicodedata
from pathlib import Path

from speaker_benchmark.adapters.common import InvalidModelOutput, assign_words
from speaker_benchmark.adapters.pipeline import QwenPyannote
from speaker_benchmark.audio import load_audio
from speaker_benchmark.schema import Prediction, Segment
from speaker_benchmark.scoring import normalize


def aligned_words(text, items, duration):
    """Reject lost words and invalid times; intersect valid spans with the audio window."""
    items = list(items)
    tokens = normalize(text).split()

    def aligner_token(token):
        return "".join(
            c for c in token if c == "'" or unicodedata.category(c).startswith(("L", "N"))
        )

    # The aligner strips currency signs from numeric words. Restore the original
    # token only after proving a one-to-one match with its alignment token.
    if len(tokens) != len(items) or any(
        not aligner_token(token) or aligner_token(token) != normalize(item.text)
        for token, item in zip(tokens, items)
    ):
        raise InvalidModelOutput(text, {"reason": "alignment_changed_transcript"})
    words, clipped = [], 0
    for token, item in zip(tokens, items, strict=True):
        start, end = float(item.start_time), float(item.end_time)
        if not math.isfinite(start + end) or start < 0 or end < start:
            raise InvalidModelOutput(text, {"reason": "invalid_alignment_timing"})
        clipped += int(start > duration or end > duration)
        words.append(Segment("unassigned", token, min(start, duration), min(end, duration)))
    if normalize(" ".join(w.text for w in words)) != normalize(text):
        raise InvalidModelOutput(text, {"reason": "alignment_changed_transcript"})
    return words, clipped


class AlignedPyannote(QwenPyannote):
    """Use the same diarizer and word assignment contract as the existing Qwen pipeline."""

    max_duration = 300

    def load(self):
        from qwen_asr import Qwen3ForcedAligner

        self.load_asr()
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            self.local_model("aligner"), dtype=self.dtype(), device_map=self.device,
        )
        self.load_diarizer()

    def load_asr(self):
        raise NotImplementedError

    def recognize(self, audio_path, audio, language):
        raise NotImplementedError

    def transcribe(self, audio_path, duration, language):
        audio = load_audio(audio_path, 16000)
        text, metadata = self.recognize(audio_path, audio, language)
        normalized = normalize(text)
        items = self.aligner.align(
            audio=(audio, 16000), text=normalized,
            language="English" if language in ("en", "English") else language,
        )[0] if normalized else []
        words, clipped = aligned_words(text, items, duration)
        activity, assignment_activity = self.diarize(audio_path, audio)
        return Prediction(
            assign_words(words, assignment_activity), activity=activity,
            timing="forced_alignment",
            metadata={
                **metadata, "raw_text": text,
                "diarizer": "pyannote/speaker-diarization-community-1",
                "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
                "alignment_clipped_words": clipped,
                "assignment": "maximum overlap; no overlap becomes unassigned",
            },
        )


class WhisperPyannote(AlignedPyannote):
    def load_asr(self):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.local_model(), torch_dtype=self.dtype(), use_safetensors=True,
        ).to(self.device).eval()
        processor = AutoProcessor.from_pretrained(self.local_model())
        self.asr = pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.dtype(), device=self.device,
        )

    def recognize(self, audio_path, audio, language):
        result = self.asr(
            {"raw": audio, "sampling_rate": 16000}, return_timestamps=True,
            generate_kwargs={"language": language, "task": "transcribe", "do_sample": False},
        )
        return result["text"], {"asr_mode": "whisper_sequential_long_form"}


class VoxtralPyannote(AlignedPyannote):
    def load_asr(self):
        from transformers import AutoProcessor, BitsAndBytesConfig, VoxtralForConditionalGeneration

        quantized = self.options.get("quantization") == "nf4"
        extra = {}
        if quantized:
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype(),
            )
        self.processor = AutoProcessor.from_pretrained(self.local_model())
        self.asr = VoxtralForConditionalGeneration.from_pretrained(
            self.local_model(), torch_dtype=self.dtype(), device_map=self.device,
            **extra,
        ).eval()

    def recognize(self, audio_path, audio, language):
        import torch

        language = "en" if language == "English" else language
        inputs = self.processor.apply_transcription_request(
            audio=str(audio_path), language=language,
            model_id=self.options["upstream_model"],
        ).to(self.device, dtype=self.dtype())
        limit = self.options.get("max_new_tokens", 4096)
        with torch.inference_mode():
            output = self.asr.generate(**inputs, max_new_tokens=limit, do_sample=False)
        tokens = output[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        metadata = {
            "asr_mode": "voxtral_transcription",
            "quantization": self.options.get("quantization", "none"),
            "generation_limit_hit": tokens.shape[1] >= limit,
        }
        if metadata["generation_limit_hit"]:
            raise InvalidModelOutput(text, metadata)
        return text, metadata


class NemoPyannote(AlignedPyannote):
    def load_asr(self):
        from nemo.collections.asr.models import ASRModel

        self.asr = ASRModel.restore_from(self.local_model(), map_location=self.device).eval()

    def recognize(self, audio_path, audio, language):
        import torch

        # NeMo checkpoints select their own decoder; do not substitute the English-only model.
        target_lang = self.options.get("target_lang")
        with torch.inference_mode(), tempfile.TemporaryDirectory() as temporary:
            inputs = [str(audio_path)]
            extra = {}
            if target_lang:
                # The pinned NeMo Lhotse loader reads supervision language from
                # `lang` and otherwise uses a training-time random prompt mode.
                # A manifest carries both required fields through its input path.
                manifest = Path(temporary) / "input.jsonl"
                manifest.write_text(json.dumps({
                    "audio_filepath": str(Path(audio_path).resolve()),
                    "duration": len(audio) / 16000, "text": "",
                    "lang": target_lang, "target_lang": target_lang,
                    "prompt_mode": "auto" if target_lang == "auto" else "langID",
                }) + "\n")
                inputs = [str(manifest)]
                extra["target_lang"] = target_lang
            result = self.asr.transcribe(
                inputs, batch_size=1, return_hypotheses=True, **extra,
            )[0]
        text = result.text.strip()
        if target_lang:
            text = text.removesuffix(f"<{target_lang}>").rstrip()
        return text, {
            "asr_mode": "nemo_offline", **extra,
            "raw_asr_text": result.text, "language_tag_removed": text != result.text.strip(),
        }
