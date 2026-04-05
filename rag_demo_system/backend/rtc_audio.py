"""WebRTC audio bridge for streaming mode.

Provides:
- TTSAudioTrack: outbound RTP track that serves TTS audio as 20ms Opus frames
- RTCAudioHandler: manages aiortc peer connection, inbound/outbound audio
"""
from __future__ import annotations

import asyncio
import fractions
import time
from typing import Awaitable, Callable

from aiortc import AudioStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, AudioResampler

AUDIO_PTIME = 0.020  # 20ms per RTP frame


class TTSAudioTrack(AudioStreamTrack):
    """Outbound audio track that serves TTS PCM16 frames via RTP."""

    kind = "audio"

    def __init__(self, sample_rate: int = 24000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = int(AUDIO_PTIME * sample_rate)
        self._bytes_per_frame = self.samples_per_frame * 2
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._remainder = b""
        self._start: float | None = None
        self._timestamp: int = 0

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()
        else:
            self._timestamp += self.samples_per_frame
            target = self._start + (self._timestamp / self.sample_rate)
            wait = target - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        try:
            pcm_data = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pcm_data = b"\x00" * self._bytes_per_frame

        frame = AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.planes[0].update(pcm_data)
        frame.pts = self._timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        return frame

    def push_audio(self, pcm16_bytes: bytes) -> None:
        """Push raw PCM16 bytes into the outbound queue, chunked into frames."""
        data = self._remainder + pcm16_bytes
        offset = 0
        while offset + self._bytes_per_frame <= len(data):
            self._queue.put_nowait(data[offset : offset + self._bytes_per_frame])
            offset += self._bytes_per_frame
        self._remainder = data[offset:]

    def flush(self) -> None:
        """Flush any remaining partial frame (zero-padded) into the queue."""
        if self._remainder:
            padded = self._remainder + b"\x00" * (
                self._bytes_per_frame - len(self._remainder)
            )
            self._queue.put_nowait(padded)
            self._remainder = b""


_resampler_cache: dict[tuple[int, str, int], AudioResampler] = {}


def resample_frame(frame: AudioFrame, target_rate: int = 24000) -> bytes:
    """Resample an AudioFrame to mono PCM16 at the target sample rate."""
    key = (frame.sample_rate, frame.layout.name, target_rate)
    if key not in _resampler_cache:
        _resampler_cache[key] = AudioResampler(
            format="s16",
            layout="mono",
            rate=target_rate,
        )
    resampler = _resampler_cache[key]
    out_frames = resampler.resample(frame)
    return b"".join(bytes(f.planes[0]) for f in out_frames)


class RTCAudioHandler:
    """Manages an aiortc peer connection for one voice session."""

    def __init__(
        self,
        on_audio: Callable[[bytes], Awaitable[None]],
        sample_rate: int = 24000,
    ) -> None:
        self._on_audio = on_audio
        self._sample_rate = sample_rate
        self.pc = RTCPeerConnection()
        self.tts_track = TTSAudioTrack(sample_rate=sample_rate)
        self.pc.addTrack(self.tts_track)
        self._audio_task: asyncio.Task | None = None

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                self._audio_task = asyncio.create_task(
                    self._consume_inbound(track)
                )

            @track.on("ended")
            def on_ended():
                pass

    async def _consume_inbound(self, track) -> None:
        """Read inbound audio frames, resample, and forward to the callback."""
        try:
            while True:
                frame = await track.recv()
                pcm16 = resample_frame(frame, self._sample_rate)
                if pcm16:
                    await self._on_audio(pcm16)
        except Exception:
            pass

    async def handle_offer(self, sdp: str) -> str:
        """Accept a WebRTC offer SDP, return the answer SDP."""
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return self.pc.localDescription.sdp

    async def close(self) -> None:
        """Tear down the peer connection and all associated tasks."""
        if self._audio_task and not self._audio_task.done():
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
        self.tts_track.stop()
        await self.pc.close()
