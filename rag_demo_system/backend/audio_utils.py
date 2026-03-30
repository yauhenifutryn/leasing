from __future__ import annotations

from array import array


def resample_pcm16_mono(audio_bytes: bytes, src_rate_hz: int, dst_rate_hz: int) -> bytes:
    if src_rate_hz == dst_rate_hz or not audio_bytes:
        return audio_bytes
    samples = array("h")
    samples.frombytes(audio_bytes)
    if not samples:
        return audio_bytes
    out_len = max(1, round(len(samples) * dst_rate_hz / src_rate_hz))
    out = array("h", [0]) * out_len
    for idx in range(out_len):
        src_index = min(len(samples) - 1, int(idx * src_rate_hz / dst_rate_hz))
        out[idx] = samples[src_index]
    return out.tobytes()
