"""AudioSocket protocol handler and SIP audio adapter.

Implements the Asterisk AudioSocket protocol (TCP-based bidirectional
audio streaming) and provides a transport adapter for the voice pipeline.

Protocol: Each frame is [1B type][2B length big-endian][payload].
Types: 0x01=UUID, 0x10=Audio, 0x00=Hangup, 0x03=DTMF, 0xFF=Error.
Audio is signed 16-bit PCM at 8kHz (160 samples = 20ms per frame).
"""
from __future__ import annotations

import asyncio
import struct
from typing import Any

import numpy as np
from scipy.signal import resample_poly

# AudioSocket frame type constants
FRAME_UUID = 0x01
FRAME_AUDIO = 0x10
FRAME_HANGUP = 0x00
FRAME_DTMF = 0x03
FRAME_ERROR = 0xFF

HEADER_SIZE = 3  # 1 byte type + 2 bytes length


def parse_frame(raw: bytes) -> tuple[int, bytes]:
    """Parse a complete AudioSocket frame into (type, payload)."""
    frame_type = raw[0]
    length = struct.unpack("!H", raw[1:3])[0]
    payload = raw[3 : 3 + length]
    return frame_type, payload


MAX_FRAME_PAYLOAD = 65000  # AudioSocket header uses uint16 length (max 65535)


def build_audio_frame(pcm16_bytes: bytes) -> bytes:
    """Build AudioSocket audio frame(s) from raw PCM16 bytes.

    Splits into multiple frames if payload exceeds 65000 bytes.
    """
    if len(pcm16_bytes) <= MAX_FRAME_PAYLOAD:
        header = struct.pack("!BH", FRAME_AUDIO, len(pcm16_bytes))
        return header + pcm16_bytes
    # Split into chunks
    result = b""
    for i in range(0, len(pcm16_bytes), MAX_FRAME_PAYLOAD):
        chunk = pcm16_bytes[i : i + MAX_FRAME_PAYLOAD]
        header = struct.pack("!BH", FRAME_AUDIO, len(chunk))
        result += header + chunk
    return result


def resample_8k_to_16k(pcm16_8k: bytes) -> bytes:
    """Resample PCM16 audio from 8kHz to 16kHz for Whisper/VAD."""
    samples = np.frombuffer(pcm16_8k, dtype=np.int16)
    resampled = resample_poly(samples, up=2, down=1).astype(np.int16)
    return resampled.tobytes()


def resample_24k_to_8k(pcm16_24k: bytes) -> bytes:
    """Resample PCM16 audio from 24kHz (Silero TTS) to 8kHz for AudioSocket."""
    samples = np.frombuffer(pcm16_24k, dtype=np.int16)
    resampled = resample_poly(samples, up=1, down=3).astype(np.int16)
    return resampled.tobytes()


def resample_24k_to_16k(pcm16_24k: bytes) -> bytes:
    """Resample PCM16 audio from 24kHz (Silero TTS) to 16kHz for AudioSocket slin16."""
    samples = np.frombuffer(pcm16_24k, dtype=np.int16)
    resampled = resample_poly(samples, up=2, down=3).astype(np.int16)
    return resampled.tobytes()


def resample_16k_to_8k(pcm16_16k: bytes) -> bytes:
    """Resample PCM16 audio from 16kHz to 8kHz for AudioSocket."""
    samples = np.frombuffer(pcm16_16k, dtype=np.int16)
    resampled = resample_poly(samples, up=1, down=2).astype(np.int16)
    return resampled.tobytes()


class SIPAudioAdapter:
    """AudioSocket transport adapter for the voice pipeline.

    Reads AudioSocket frames from an asyncio StreamReader, resamples
    audio to 16kHz for the pipeline, and writes TTS audio back
    resampled to 8kHz.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.session_id: str = ""
        self.caller_phone: str | None = None
        self.dtmf_buffer: list[str] = []
        self._closed = False

    async def read_next(self) -> dict[str, Any] | None:
        """Read and parse the next AudioSocket frame.

        Returns a dict with 'type' key and type-specific data:
          {"type": "uuid", "uuid": "..."}
          {"type": "audio", "pcm16": b"..."} (resampled to 16kHz)
          {"type": "dtmf", "digit": "5"}
          {"type": "hangup"}
          {"type": "error", "message": "..."}
        Returns None on EOF.
        """
        try:
            header = await self.reader.readexactly(HEADER_SIZE)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return None

        frame_type = header[0]
        length = struct.unpack("!H", header[1:3])[0]

        if length > 0:
            try:
                payload = await self.reader.readexactly(length)
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return None
        else:
            payload = b""

        if frame_type == FRAME_UUID:
            # Asterisk 18 sends 16 raw bytes (binary UUID), not 36-char ASCII.
            # Handle both formats.
            if len(payload) == 16:
                import uuid as _uuid_mod
                self.session_id = str(_uuid_mod.UUID(bytes=payload))
            else:
                self.session_id = payload.decode("ascii", errors="replace").strip()
            return {"type": "uuid", "uuid": self.session_id}

        if frame_type == FRAME_AUDIO:
            pcm_16k = resample_8k_to_16k(payload)
            return {"type": "audio", "pcm16": pcm_16k, "pcm_raw_8k": payload}

        if frame_type == FRAME_DTMF:
            digit = payload.decode("ascii", errors="replace")
            self.dtmf_buffer.append(digit)
            return {"type": "dtmf", "digit": digit}

        if frame_type == FRAME_HANGUP:
            return {"type": "hangup"}

        if frame_type == FRAME_ERROR:
            msg = payload.decode("utf-8", errors="replace")
            return {"type": "error", "message": msg}

        return {"type": "unknown", "frame_type": frame_type}

    async def write_audio(self, pcm16_24k: bytes) -> None:
        """Write TTS audio (24kHz PCM16) back to AudioSocket as 8kHz 20ms frames.

        AudioSocket uses slin (8kHz signed linear 16-bit).
        20ms at 8kHz = 160 samples = 320 bytes per frame.
        Asterisk reads from the TCP buffer at its own 20ms tick.
        We write all frames at once; Asterisk handles playback timing.
        """
        if self._closed:
            return
        pcm_8k = resample_24k_to_8k(pcm16_24k)
        # Build all frames and write at once
        frame_size = 320  # 160 samples * 2 bytes at 8kHz = 20ms
        buf = bytearray()
        for i in range(0, len(pcm_8k), frame_size):
            chunk = pcm_8k[i : i + frame_size]
            if not chunk:
                break
            buf.extend(struct.pack("!BH", FRAME_AUDIO, len(chunk)))
            buf.extend(chunk)
        if buf and not self._closed:
            self.writer.write(buf)
            await self.writer.drain()


    async def close(self) -> None:
        """Shut down the TCP connection."""
        if self._closed:
            return
        self._closed = True
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


_SENTINEL = object()


async def query_caller_id_ami(
    *,
    channel_id: str,
    ami_host: str,
    ami_port: int,
    ami_username: str,
    ami_secret: str,
    _mock_response: dict[str, str] | None = _SENTINEL,
) -> str | None:
    """Query Asterisk AMI for the caller ID of a channel.

    Uses a raw TCP connection to AMI (no external library dependency).
    Returns the CallerIDNum string, or None on any failure.

    Pass _mock_response (not the sentinel) for testing without AMI.
    """
    if _mock_response is not _SENTINEL:
        if _mock_response is None:
            return None
        return _mock_response.get("CallerIDNum")

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ami_host, ami_port),
            timeout=3.0,
        )
        # Read AMI greeting
        await asyncio.wait_for(reader.readline(), timeout=2.0)

        # Login
        writer.write(
            f"Action: Login\r\n"
            f"Username: {ami_username}\r\n"
            f"Secret: {ami_secret}\r\n"
            f"\r\n".encode()
        )
        await writer.drain()
        login_ok = False
        for _ in range(20):
            line = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if b"Success" in line:
                login_ok = True
            if line.strip() == b"":
                break
        if not login_ok:
            writer.close()
            return None

        # Query channels
        writer.write(
            f"Action: Status\r\n"
            f"ActionID: caller-lookup\r\n"
            f"\r\n".encode()
        )
        await writer.drain()

        caller_id = None
        for _ in range(200):
            line = await asyncio.wait_for(reader.readline(), timeout=3.0)
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded.startswith("CallerIDNum:"):
                candidate = decoded.split(":", 1)[1].strip()
                if candidate and candidate != "<unknown>":
                    caller_id = candidate
            if decoded == "" and caller_id:
                break
            if "StatusComplete" in decoded:
                break

        writer.write(b"Action: Logoff\r\n\r\n")
        await writer.drain()
        writer.close()
        return caller_id

    except Exception:  # noqa: BLE001
        return None
