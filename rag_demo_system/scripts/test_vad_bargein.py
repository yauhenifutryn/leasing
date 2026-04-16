#!/usr/bin/env python3
"""Test VAD barge-in detection with synthetic or recorded audio.

Usage:
  # Generate synthetic speech-like audio and test VAD response:
  python scripts/test_vad_bargein.py --synthetic

  # Test with a recorded WAV file (16kHz mono PCM):
  python scripts/test_vad_bargein.py --file path/to/audio.wav

  # Test with different thresholds:
  python scripts/test_vad_bargein.py --synthetic --threshold 0.40 --count 4
  python scripts/test_vad_bargein.py --synthetic --threshold 0.45 --count 5

Shows per-frame VAD probability, detection timing, and when barge-in would trigger.
"""

import argparse
import math
import struct
import sys
import time
import wave
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def generate_synthetic_audio(duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate synthetic audio: 0.5s silence + speech-like signal + 0.5s silence."""
    samples = []
    total = int(duration_sec * sample_rate)
    silence_pre = int(0.5 * sample_rate)
    silence_post = int(0.5 * sample_rate)
    speech_len = total - silence_pre - silence_post

    for i in range(total):
        if i < silence_pre or i >= (silence_pre + speech_len):
            samples.append(0)
        else:
            # Speech-like: mix of frequencies with some noise
            t = i / sample_rate
            val = (
                8000 * math.sin(2 * math.pi * 150 * t)
                + 4000 * math.sin(2 * math.pi * 300 * t)
                + 2000 * math.sin(2 * math.pi * 600 * t)
                + int(1000 * math.sin(2 * math.pi * 1200 * t * (1 + 0.1 * math.sin(2 * math.pi * 5 * t))))
            )
            samples.append(max(-32768, min(32767, int(val))))

    return struct.pack(f"<{len(samples)}h", *samples)


def load_wav(path: str) -> tuple[bytes, int]:
    """Load a WAV file and return (pcm_bytes, sample_rate)."""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "Must be mono"
        assert wf.getsampwidth() == 2, "Must be 16-bit"
        return wf.readframes(wf.getnframes()), wf.getframerate()


def run_vad_test(
    pcm_data: bytes,
    sample_rate: int = 16000,
    threshold: float = 0.40,
    consecutive_count: int = 4,
    frame_ms: int = 32,
) -> None:
    """Feed audio through Silero VAD and report barge-in timing."""
    from backend.vad import SileroVAD

    vad = SileroVAD()
    frame_size = int(sample_rate * frame_ms / 1000) * 2  # bytes per frame

    barge_count = 0
    barge_triggered = False
    barge_time_ms = 0
    total_frames = 0

    print(f"VAD Test: threshold={threshold}, count={consecutive_count}, frame={frame_ms}ms")
    print(f"Audio: {len(pcm_data)} bytes, {len(pcm_data) / sample_rate / 2:.2f}s at {sample_rate}Hz")
    print("-" * 70)
    print(f"{'Frame':>6} {'Time':>8} {'Prob':>6} {'Count':>6} {'Status'}")
    print("-" * 70)

    offset = 0
    while offset + frame_size <= len(pcm_data):
        chunk = pcm_data[offset:offset + frame_size]
        offset += frame_size
        total_frames += 1
        time_ms = total_frames * frame_ms

        vad.feed(chunk)
        prob = vad.last_probability

        if prob >= threshold:
            barge_count += 1
        else:
            barge_count = max(0, barge_count - 1)

        status = ""
        if barge_count >= consecutive_count and not barge_triggered:
            barge_triggered = True
            barge_time_ms = time_ms
            status = "<< BARGE-IN TRIGGERED >>"
        elif prob >= threshold:
            status = f"speech ({barge_count}/{consecutive_count})"

        # Print every frame where something happens, or every 10th frame
        if status or total_frames % 10 == 0:
            print(f"{total_frames:>6} {time_ms:>7}ms {prob:>5.2f} {barge_count:>5}/{consecutive_count} {status}")

    print("-" * 70)
    if barge_triggered:
        print(f"BARGE-IN at {barge_time_ms}ms (frame {barge_time_ms // frame_ms})")
    else:
        print("No barge-in detected")
    print(f"Total: {total_frames} frames, {total_frames * frame_ms}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test VAD barge-in detection")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic speech audio")
    parser.add_argument("--file", type=str, help="Path to WAV file (16kHz mono)")
    parser.add_argument("--threshold", type=float, default=0.40, help="VAD probability threshold")
    parser.add_argument("--count", type=int, default=4, help="Consecutive frames needed")
    args = parser.parse_args()

    if args.file:
        pcm_data, sr = load_wav(args.file)
        run_vad_test(pcm_data, sr, args.threshold, args.count)
    elif args.synthetic:
        pcm_data = generate_synthetic_audio(3.0, 16000)
        run_vad_test(pcm_data, 16000, args.threshold, args.count)
    else:
        print("Usage: --synthetic or --file path/to/audio.wav")
        print("Compare thresholds:")
        print("  python scripts/test_vad_bargein.py --synthetic --threshold 0.40 --count 4")
        print("  python scripts/test_vad_bargein.py --synthetic --threshold 0.45 --count 5")


if __name__ == "__main__":
    main()
