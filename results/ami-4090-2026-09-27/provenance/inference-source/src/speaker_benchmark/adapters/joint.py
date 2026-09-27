"""Official local inference APIs for MOSS and both VibeVoice checkpoints."""

import json
import re
import time
from pathlib import Path

from speaker_benchmark.adapters.common import (
    Adapter,
    InvalidModelOutput,
    parse_moss,
    parse_vibe,
    remove_sound_annotations,
)
from speaker_benchmark.audio import load_audio
from speaker_benchmark.schema import Prediction, Segment


class Moss(Adapter):
    max_duration = 5400

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        model = self.local_model()
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                dtype=self.dtype(),
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )

    def transcribe(self, audio_path, duration, language):
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
        )

        limit = self.options.get("max_new_tokens", 32768)
        output = generate_transcription(
            self.model,
            self.processor,
            build_transcription_messages(audio_path),
            max_new_tokens=limit,
            do_sample=False,
        )
        metadata = {"generation_limit_hit": output["generated_tokens"] >= limit}
        try:
            segments = parse_moss(output["text"])
        except ValueError as exc:
            raise InvalidModelOutput(output["text"], metadata) from exc
        return Prediction(segments, metadata={"raw_output": output["text"], **metadata})


class VibeVoice(Adapter):
    max_duration = 3600

    def load(self):
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        model = self.local_model()
        processor_options = {}
        if "tokenizer" in self.options:
            processor_options["language_model_pretrained_name"] = self.local_model("tokenizer")
        self.processor = VibeVoiceASRProcessor.from_pretrained(model, **processor_options)
        self.model = (
            VibeVoiceASRForConditionalGeneration.from_pretrained(
                model,
                dtype=self.dtype(),
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )

    def transcribe(self, audio_path, duration, language):
        import torch

        audio = load_audio(audio_path, 24000)
        inputs = self.processor(
            audio=[audio],
            sampling_rate=24000,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
        limit = self.options.get("max_new_tokens", 32768)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=False,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = self.processor.decode(generated, skip_special_tokens=True)
        metadata = {"generation_limit_hit": len(generated) >= limit}
        try:
            segments = parse_vibe(text)
        except ValueError as exc:
            raise InvalidModelOutput(text, metadata) from exc
        return Prediction(segments, metadata={"raw_output": text, **metadata})


def parse_stream_chunk(text, current_speaker):
    """The official format permits a speaker's text to continue into the next chunk."""
    text = remove_sound_annotations(text)
    parts = re.split(r"(?:^|\n)\s*Speaker\s+(\d+)\s*:", text)
    segments = []
    if parts[0].strip():
        if current_speaker is None:
            current_speaker = "unassigned"
        segments.append(Segment(current_speaker, parts[0]))
    for index in range(1, len(parts), 2):
        current_speaker = parts[index]
        if parts[index + 1].strip():
            segments.append(Segment(current_speaker, parts[index + 1]))
    return segments, current_speaker


class VibeVoiceStreaming(VibeVoice):
    max_duration = 480

    def load(self):
        super().load()
        config = json.loads((Path(self.local_model()) / "preprocessor_config.json").read_text())
        self.sample_rate = config["target_sample_rate"]
        ratio = config["speech_tok_compress_ratio"]
        self.chunk_samples = config["chunk_frames"] * ratio
        self.lookahead_samples = config["lookahead_frames"] * ratio
        if self.processor.tokenizer.text_chunk_end_id is None:
            raise ValueError("A streaming checkpoint is required")

    def transcribe(self, audio_path, duration, language):
        import numpy as np
        import torch

        audio = load_audio(audio_path, self.sample_rate)
        paced = self.options.get("realtime", True)
        segments, events, raw = [], [], []
        current_speaker = None
        origin = time.perf_counter()
        compute_seconds = 0.0
        with torch.inference_mode():
            state = self.model.init_streaming_state(self.processor.tokenizer)
            for start in range(0, len(audio), self.chunk_samples):
                end = min(len(audio), start + self.chunk_samples + self.lookahead_samples)
                available_at = end / self.sample_rate
                if paced:
                    time.sleep(max(0, origin + available_at - time.perf_counter()))
                window = audio[start:end]
                target = self.chunk_samples + self.lookahead_samples
                window = np.pad(window, (0, target - len(window)))
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize(self.device)
                begin = time.perf_counter()
                features = self.model.encode_speech(
                    torch.from_numpy(window).unsqueeze(0).to(self.device)
                )
                text, state = self.model.streaming_generate_step(
                    audio_features=features,
                    streaming_state=state,
                    tokenizer=self.processor.tokenizer,
                    max_new_tokens=self.options.get("max_new_tokens_per_chunk", 256),
                    temperature=0.0,
                )
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize(self.device)
                emitted = time.perf_counter() - origin
                compute_seconds += time.perf_counter() - begin
                parsed, current_speaker = parse_stream_chunk(text, current_speaker)
                segments.extend(parsed)
                events.append(
                    {
                        "audio_available_seconds": available_at,
                        "chunk_end_seconds": min(
                            duration, (start + self.chunk_samples) / self.sample_rate
                        ),
                        "emitted_seconds": emitted,
                        "has_text": bool(parsed),
                        "text": text,
                    }
                )
                raw.append(text)
        return Prediction(
            segments,
            timing="unavailable",
            events=events,
            metadata={
                "raw_output": "".join(raw),
                "realtime_paced": paced,
                "stream_compute_seconds": compute_seconds,
                "timing_reason": "Official streaming output has no speech timestamps",
                "generation_limit_hit": None,
            },
        )
