"""Tests for WebRTC audio bridge: TTSAudioTrack and resample_frame."""
from pathlib import Path
import sys
import asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_tts_track_chunks_pcm_into_frames():
    from backend.rtc_audio import TTSAudioTrack
    track = TTSAudioTrack(sample_rate=24000)
    # 24kHz * 20ms = 480 samples/frame * 2 bytes = 960 bytes/frame
    pcm = b"\x01\x00" * 1440  # 1440 samples = 3 frames
    track.push_audio(pcm)
    assert track._queue.qsize() == 3
    chunk = track._queue.get_nowait()
    assert len(chunk) == 960


def test_tts_track_handles_partial_frame():
    from backend.rtc_audio import TTSAudioTrack
    track = TTSAudioTrack(sample_rate=24000)
    pcm = b"\x01\x00" * 720  # 720 samples = 1.5 frames
    track.push_audio(pcm)
    assert track._queue.qsize() == 1
    track.push_audio(b"\x01\x00" * 240)
    assert track._queue.qsize() == 2


def test_tts_track_recv_returns_silence_when_empty():
    from backend.rtc_audio import TTSAudioTrack
    track = TTSAudioTrack(sample_rate=24000)
    frame = asyncio.run(track.recv())
    assert frame.format.name == "s16"
    assert frame.sample_rate == 24000
    assert frame.samples == 480
    assert bytes(frame.planes[0]) == b"\x00" * 960


def test_resample_48k_stereo_to_24k_mono():
    from backend.rtc_audio import resample_frame
    from av import AudioFrame
    frame = AudioFrame(format="s16", layout="stereo", samples=960)
    frame.sample_rate = 48000
    frame.planes[0].update(b"\x10\x00" * 960 * 2)
    pcm = resample_frame(frame, target_rate=24000)
    assert isinstance(pcm, bytes)
    # Theoretical: 480 samples * 2 bytes = 960 bytes.
    # ffmpeg resampler may produce slightly more due to filter response;
    # accept within 10% tolerance.
    assert 860 <= len(pcm) <= 1100, f"unexpected resample size: {len(pcm)}"
    # Output must be even (whole 16-bit samples)
    assert len(pcm) % 2 == 0
