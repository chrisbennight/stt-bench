"""Identical audio decoding and resampling for all native adapters."""

import math


def load_audio(path, sample_rate):
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("Choose a single channel during dataset preparation")
    audio = audio[:, 0]
    if original_rate != sample_rate:
        divisor = math.gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, original_rate // divisor)
    return audio
