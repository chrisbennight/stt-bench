"""Qwen ASR/alignment followed by word-to-speaker reconciliation."""

from speaker_benchmark.adapters.common import Adapter, assign_words
from speaker_benchmark.audio import load_audio
from speaker_benchmark.schema import Prediction, Segment


class QwenPipeline(Adapter):
    def load(self):
        from qwen_asr import Qwen3ASRModel

        self.asr = Qwen3ASRModel.from_pretrained(
            self.local_model(),
            dtype=self.dtype(),
            device_map=self.device,
            max_inference_batch_size=1,
            max_new_tokens=self.options.get("max_new_tokens", 4096),
            forced_aligner=self.local_model("aligner"),
            forced_aligner_kwargs={"dtype": self.dtype(), "device_map": self.device},
        )
        self.load_diarizer()

    def transcribe(self, audio_path, duration, language):
        audio = load_audio(audio_path, 16000)
        # Both Qwen variants use identical ASR chunking; diarization sees the entire recording.
        chunk_seconds = self.options.get("asr_chunk_seconds", 30)
        if not 1 <= chunk_seconds <= 300:
            raise ValueError("Qwen alignment chunks must be between 1 and 300 seconds")
        words, raw_text, zero_duration = [], [], 0
        for start in range(0, len(audio), int(chunk_seconds * 16000)):
            chunk = audio[start : start + int(chunk_seconds * 16000)]
            result = self.asr.transcribe(
                audio=(chunk, 16000), language=language, return_time_stamps=True
            )[0]
            raw_text.append(result.text)
            offset = start / 16000
            for item in result.time_stamps or []:
                begin, end = float(item.start_time), float(item.end_time)
                # Preserve zero-duration aligned words for text scoring with a tiny explicit span.
                if end == begin:
                    end += 0.001
                    zero_duration += 1
                words.append(Segment("unassigned", item.text, offset + begin, offset + end))
            if result.text.strip() and not result.time_stamps:
                raise ValueError("Qwen returned text without alignment")
        activity, assignment_activity = self.diarize(audio_path, audio)
        return Prediction(
            assign_words(words, assignment_activity),
            activity=activity,
            timing="forced_alignment",
            metadata={
                "raw_text": " ".join(raw_text),
                "asr_chunk_seconds": chunk_seconds,
                "zero_duration_alignment_words": zero_duration,
                "assignment": "maximum overlap; no overlap becomes unassigned",
            },
        )


class QwenPyannote(QwenPipeline):
    def load_diarizer(self):
        import torch
        from pyannote.audio import Pipeline

        self.diarizer = Pipeline.from_pretrained(self.local_model("diarizer"))
        self.diarizer.to(torch.device(self.device))

    def diarize(self, audio_path, audio):
        import torch

        result = self.diarizer(
            {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000}
        )

        def convert(annotation):
            return [
                Segment(str(speaker), "", float(turn.start), float(turn.end))
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            ]

        return convert(result.speaker_diarization), convert(result.exclusive_speaker_diarization)


class QwenNemotron(QwenPipeline):
    def load_diarizer(self):
        from nemo.collections.asr.models import SortformerEncLabelModel

        # A local .nemo checkpoint fixes both the weights and model configuration.
        self.diarizer = SortformerEncLabelModel.restore_from(
            self.local_model("diarizer"),
            map_location=self.device,
        ).eval()
        modules = self.diarizer.sortformer_modules
        modules.chunk_len = 340
        modules.chunk_right_context = 40
        modules.fifo_len = 40
        modules.spkcache_update_period = 300
        self.diarizer._check_streaming_parameters()

    def diarize(self, audio_path, audio):
        rows = self.diarizer.diarize(audio=[audio], batch_size=1, sample_rate=16000)[0]
        activity = []
        for row in rows:
            start, end, speaker = row.split()
            activity.append(Segment(speaker, "", float(start), float(end)))
        return activity, activity
