# OpenRouter cost estimates, 27 September 2026

These are estimates before inference, not benchmark results. See the [method and assumptions](../../docs/OPENROUTER.md).

| Model | Billing unit | Four minutes | 92.2784 minutes |
| --- | --- | --- | --- |
| [google/chirp-3](https://openrouter.ai/google/chirp-3) | second | $0.0640 | $1.4765 |
| [fish-audio/transcribe-1-pro](https://openrouter.ai/fish-audio/transcribe-1-pro) | second | $0.0240 | $0.5537 |
| [fish-audio/transcribe-1](https://openrouter.ai/fish-audio/transcribe-1) | second | $0.0240 | $0.5537 |
| [microsoft/mai-transcribe-1.5](https://openrouter.ai/microsoft/mai-transcribe-1.5) | hour | $0.0240 | $0.5537 |
| [openai/whisper-1](https://openrouter.ai/openai/whisper-1) | second | $0.0240 | $0.5537 |
| [openai/gpt-4o-transcribe](https://openrouter.ai/openai/gpt-4o-transcribe) | token | $0.0240 | $0.5537 |
| [google/gemini-3.5-transcribe](https://openrouter.ai/google/gemini-3.5-transcribe) | token | $0.0204 | $0.4706 |
| [openai/gpt-transcribe](https://openrouter.ai/openai/gpt-transcribe) | second | $0.0180 | $0.4153 |
| [deepgram/nova-3](https://openrouter.ai/deepgram/nova-3) | second | $0.0172 | $0.3968 |
| [assemblyai/universal-3-5-pro](https://openrouter.ai/assemblyai/universal-3-5-pro) | second | $0.0150 | $0.3460 |
| [meta/muse-voice-transcribe-1.0](https://openrouter.ai/meta/muse-voice-transcribe-1.0) | second | $0.0120 | $0.2768 |
| [mistralai/voxtral-small-24b-2507-stt](https://openrouter.ai/mistralai/voxtral-small-24b-2507-stt) | second | $0.0120 | $0.2768 |
| [mistralai/voxtral-mini-transcribe](https://openrouter.ai/mistralai/voxtral-mini-transcribe) | second | $0.0120 | $0.2768 |
| [openai/gpt-4o-mini-transcribe](https://openrouter.ai/openai/gpt-4o-mini-transcribe) | token | $0.0120 | $0.2768 |
| [qwen/qwen3-asr-flash-2026-02-10](https://openrouter.ai/qwen/qwen3-asr-flash-2026-02-10) | second | $0.0084 | $0.1938 |
| [x-ai/grok-stt-1.0](https://openrouter.ai/x-ai/grok-stt-1.0) | second | $0.0067 | $0.1538 |
| [microsoft/mai-transcribe-2](https://openrouter.ai/microsoft/mai-transcribe-2) | hour | $0.0067 | $0.1538 |
| [nvidia/parakeet-tdt-0.6b-v3](https://openrouter.ai/nvidia/parakeet-tdt-0.6b-v3) | second | $0.0060 | $0.1384 |
| [mistralai/voxtral-mini-3b-2507](https://openrouter.ai/mistralai/voxtral-mini-3b-2507) | second | $0.0040 | $0.0923 |
| [qwen/qwen3-asr-1.7b](https://openrouter.ai/qwen/qwen3-asr-1.7b) | second | $0.0018 | $0.0415 |
| [openai/whisper-large-v3](https://openrouter.ai/openai/whisper-large-v3) | second | $0.0018 | $0.0415 |
| [nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b](https://openrouter.ai/nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b) | second | $0.0008 | $0.0184 |
| [qwen/qwen3-asr-0.6b](https://openrouter.ai/qwen/qwen3-asr-0.6b) | second | $0.0008 | $0.0184 |
| [openai/whisper-large-v3-turbo](https://openrouter.ai/openai/whisper-large-v3-turbo) | second | $0.0008 | $0.0184 |

Total: **$0.34** for the short pass; **$7.85** for the full audio duration.

Token-priced rows are estimates based on the assumptions in each JSON file. No model requests have been made.
