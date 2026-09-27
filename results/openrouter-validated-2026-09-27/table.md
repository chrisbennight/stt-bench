| Model | Run on | WER ↓ | cpWER (77) ↓ | tcpWER (77) ↓ | DER (77) ↓ | RTF ↓ | API $ ↓ | VRAM GiB ↓ | Clips |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [MAI Transcribe 2](https://openrouter.ai/microsoft/mai-transcribe-2) | OpenRouter | 🥇 26.10% | 🥈 35.13% | 🥈 35.97% | 58.76% | 0.145 | 0.1538 | — | 94/94 |
| [MAI Transcribe 1.5](https://openrouter.ai/microsoft/mai-transcribe-1.5) | OpenRouter | 🥈 26.41% | No labels | No labels | No labels | 0.148 | 0.5538 | — | 94/94 |
| [MOSS 0.9B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | Local 4090 | 🥉 30.92% | 🥇 26.29% | 🥇 27.10% | 🥇 23.57% | 0.073 | — | 🥇 1.92 | 94/94 |
| [Fish Transcribe 1 Pro](https://openrouter.ai/fish-audio/transcribe-1-pro) | OpenRouter | 31.97% | 40.28% | No speaker times | No speaker times | 0.157 | 0.5538 | — | 94/94 |
| [NVIDIA Parakeet TDT v3](https://openrouter.ai/nvidia/parakeet-tdt-0.6b-v3) | OpenRouter | 32.76% | No labels | No labels | No labels | 0.140 | 0.1384 | — | 94/94 |
| [Parakeet TDT v3 + pyannote](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Local 4090 | 33.33% | 55.61% | 60.49% | 🥈 26.83% | 🥇 0.011 | — | 5.72 | 94/94 |
| [Qwen3 ASR Flash](https://openrouter.ai/qwen/qwen3-asr-flash-2026-02-10) | OpenRouter | 33.34% | No labels | No labels | No labels | 0.164 | 0.1937 | — | 94/94 |
| [Gemini 3.5 Transcribe](https://openrouter.ai/google/gemini-3.5-transcribe) | OpenRouter | 33.62% | No labels | No labels | No labels | 0.166 | 0.2770 | — | 94/94 |
| [Qwen3 + pyannote](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Local 4090 | 34.17% | 48.13% | 50.03% | 🥈 26.83% | 0.033 | — | 7.11 | 94/94 |
| [Qwen3 + Nemotron](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Local 4090 | 34.17% | 🥉 39.54% | 🥉 40.76% | 29.95% | 0.032 | — | 6.13 | 94/94 |
| [VibeVoice offline](https://huggingface.co/microsoft/VibeVoice-ASR) | Local 4090 | 34.82% | 39.58% | 40.77% | 34.27% | 0.140 | — | 17.09 | 94/94 |
| [Meta Muse Voice 1.0](https://openrouter.ai/meta/muse-voice-transcribe-1.0) | OpenRouter | 35.05% | No labels | No labels | No labels | 0.255 | 0.2768 | — | 94/94 |
| [Voxtral Mini 3B + pyannote](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Local 4090 | 35.82% | 58.48% | 63.02% | 🥈 26.83% | 0.053 | — | 12.02 | 92/94; 2 invalid |
| [Voxtral Mini Transcribe](https://openrouter.ai/mistralai/voxtral-mini-transcribe) | OpenRouter | 36.19% | No labels | No labels | No labels | 0.171 | 0.2738 | — | 94/94 |
| [Voxtral Small 24B NF4 + pyannote](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | Local 4090 | 37.12% | 58.50% | 62.85% | 27.93% | 0.079 | — | 16.84 | 93/94; 1 invalid |
| [Qwen3 ASR 0.6B + pyannote](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Local 4090 | 37.45% | 49.35% | 51.16% | 🥈 26.83% | 0.030 | — | 🥈 4.77 | 94/94 |
| [OpenAI GPT Transcribe](https://openrouter.ai/openai/gpt-transcribe) | OpenRouter | 38.84% | No labels | No labels | No labels | 0.154 | 0.4153 | — | 94/94 |
| [Whisper 1](https://openrouter.ai/openai/whisper-1) | OpenRouter | 39.09% | No labels | No labels | No labels | 0.167 | 0.5538 | — | 94/94 |
| [Whisper Large v3 + pyannote](https://huggingface.co/openai/whisper-large-v3) | Local 4090 | 39.35% | 59.01% | 63.59% | 🥈 26.83% | 0.043 | — | 6.18 | 94/94 |
| [GPT-4o Mini Transcribe](https://openrouter.ai/openai/gpt-4o-mini-transcribe) | OpenRouter | 39.66% | No labels | No labels | No labels | 0.150 | 0.1330 | — | 94/94 |
| [Google Chirp 3](https://openrouter.ai/google/chirp-3) | OpenRouter | 40.16% | No labels | No labels | No labels | 0.151 | 1.4608 | — | 93/94 |
| [Fish Transcribe 1](https://openrouter.ai/fish-audio/transcribe-1) | OpenRouter | 40.24% | No labels | No labels | No labels | 0.152 | 0.5538 | — | 94/94 |
| [Deepgram Nova 3](https://openrouter.ai/deepgram/nova-3) | OpenRouter | 40.58% | 65.96% | 67.24% | 54.87% | 0.138 | 0.3968 | — | 94/94 |
| [AssemblyAI Universal 3.5 Pro](https://openrouter.ai/assemblyai/universal-3-5-pro) | OpenRouter | 42.97% | No labels | No labels | No labels | 0.142 | 0.3460 | — | 94/94 |
| [VibeVoice Streaming](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) | Local 4090 | 43.71% | 49.38% | No speaker times | No speaker times | 0.115 | — | 16.36 | 94/94 |
| [Nemotron 3.5 ASR + pyannote](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Local 4090 | 43.80% | 59.57% | 64.38% | 🥈 26.83% | 🥈 0.015 | — | 5.76 | 94/94 |
| [NVIDIA Nemotron 3.5 ASR 0.6B](https://openrouter.ai/nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b) | OpenRouter | 45.62% | No labels | No labels | No labels | 0.156 | 🥇 0.0184 | — | 94/94 |
| [Whisper Turbo + pyannote](https://huggingface.co/openai/whisper-large-v3-turbo) | Local 4090 | 46.95% | 64.49% | 69.80% | 🥈 26.83% | 🥉 0.017 | — | 🥉 5.05 | 94/94 |
| [Whisper Large v3 Turbo](https://openrouter.ai/openai/whisper-large-v3-turbo) | OpenRouter | 47.68% | No labels | No labels | No labels | 0.164 | 🥈 0.0226 | — | 94/94 |
| [Whisper Large v3](https://openrouter.ai/openai/whisper-large-v3) | OpenRouter | 48.57% | No labels | No labels | No labels | 0.183 | 🥉 0.0538 | — | 94/94 |
| [GPT-4o Transcribe](https://openrouter.ai/openai/gpt-4o-transcribe) | OpenRouter | 57.16% | No labels | No labels | No labels | 0.151 | 0.2254 | — | 94/94 |
| [Grok STT 1.0](https://openrouter.ai/x-ai/grok-stt-1.0) | OpenRouter | 57.55% | 62.83% | 65.88% | 65.96% | 0.158 | 0.1538 | — | 94/94 |
