"""Tests for AudioSocket frame parser and SIP audio adapter."""
import asyncio
import struct
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# AudioSocket frame types
UUID_TYPE = 0x01
AUDIO_TYPE = 0x10
HANGUP_TYPE = 0x00
DTMF_TYPE = 0x03


def _make_frame(frame_type: int, payload: bytes) -> bytes:
    """Build an AudioSocket wire frame: [1B type][2B length BE][payload]."""
    return struct.pack("!BH", frame_type, len(payload)) + payload


def _make_uuid_frame(uuid_str: str = "550e8400-e29b-41d4-a716-446655440000") -> bytes:
    return _make_frame(UUID_TYPE, uuid_str.encode("ascii"))


def _make_audio_frame(n_samples: int = 160, amplitude: int = 1000) -> bytes:
    """160 samples = 20ms at 8kHz. Payload is PCM16 LE."""
    pcm = struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))
    return _make_frame(AUDIO_TYPE, pcm)


def _make_dtmf_frame(digit: str = "5") -> bytes:
    return _make_frame(DTMF_TYPE, digit.encode("ascii"))


def _make_hangup_frame() -> bytes:
    return _make_frame(HANGUP_TYPE, b"")


class TestParseFrame:
    def test_parse_uuid_frame(self):
        from backend.sip_audio import parse_frame
        frame = _make_uuid_frame("test-uuid-1234")
        ftype, payload = parse_frame(frame)
        assert ftype == UUID_TYPE
        assert payload == b"test-uuid-1234"

    def test_parse_audio_frame(self):
        from backend.sip_audio import parse_frame
        frame = _make_audio_frame(160, 500)
        ftype, payload = parse_frame(frame)
        assert ftype == AUDIO_TYPE
        assert len(payload) == 320  # 160 samples * 2 bytes

    def test_parse_dtmf_frame(self):
        from backend.sip_audio import parse_frame
        frame = _make_dtmf_frame("7")
        ftype, payload = parse_frame(frame)
        assert ftype == DTMF_TYPE
        assert payload == b"7"

    def test_parse_hangup_frame(self):
        from backend.sip_audio import parse_frame
        frame = _make_hangup_frame()
        ftype, payload = parse_frame(frame)
        assert ftype == HANGUP_TYPE
        assert payload == b""


class TestResample:
    def test_resample_8k_to_16k_doubles_length(self):
        from backend.sip_audio import resample_8k_to_16k
        pcm_8k = struct.pack("<160h", *([1000] * 160))
        pcm_16k = resample_8k_to_16k(pcm_8k)
        assert len(pcm_16k) == 640  # 320 samples * 2 bytes

    def test_resample_24k_to_8k_reduces_length(self):
        from backend.sip_audio import resample_24k_to_8k
        pcm_24k = struct.pack("<480h", *([1000] * 480))
        pcm_8k = resample_24k_to_8k(pcm_24k)
        assert len(pcm_8k) == 320  # 160 samples * 2 bytes

    def test_resample_roundtrip_preserves_duration(self):
        from backend.sip_audio import resample_8k_to_16k
        pcm_8k = struct.pack("<800h", *([500] * 800))
        pcm_16k = resample_8k_to_16k(pcm_8k)
        assert len(pcm_16k) // 2 == 1600


class TestBuildAudioSocketFrame:
    def test_builds_valid_audio_frame(self):
        from backend.sip_audio import build_audio_frame
        pcm = struct.pack("<160h", *([100] * 160))
        frame = build_audio_frame(pcm)
        assert frame[0] == AUDIO_TYPE
        length = struct.unpack("!H", frame[1:3])[0]
        assert length == len(pcm)
        assert frame[3:] == pcm


class TestSIPAudioAdapter:
    def test_read_audio_frame_resamples_to_16k(self):
        from backend.sip_audio import SIPAudioAdapter

        uuid_frame = _make_uuid_frame()
        audio_frame = _make_audio_frame(160, 1000)
        hangup_frame = _make_hangup_frame()
        data = uuid_frame + audio_frame + hangup_frame

        async def _run():
            reader = asyncio.StreamReader()
            reader.feed_data(data)
            reader.feed_eof()
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            adapter = SIPAudioAdapter(reader, writer)
            result = await adapter.read_next()
            assert result is not None
            assert result["type"] == "uuid"

            result = await adapter.read_next()
            assert result is not None
            assert result["type"] == "audio"
            pcm_16k = result["pcm16"]
            assert len(pcm_16k) == 640  # 160 at 8k -> 320 at 16k = 640 bytes

            result = await adapter.read_next()
            assert result is not None
            assert result["type"] == "hangup"

        asyncio.run(_run())

    def test_dtmf_buffering(self):
        from backend.sip_audio import SIPAudioAdapter

        dtmf_1 = _make_dtmf_frame("1")
        dtmf_2 = _make_dtmf_frame("2")
        dtmf_3 = _make_dtmf_frame("3")
        hangup = _make_hangup_frame()
        data = dtmf_1 + dtmf_2 + dtmf_3 + hangup

        async def _run():
            reader = asyncio.StreamReader()
            reader.feed_data(data)
            reader.feed_eof()
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            adapter = SIPAudioAdapter(reader, writer)
            for _ in range(4):
                await adapter.read_next()
            assert adapter.dtmf_buffer == ["1", "2", "3"]

        asyncio.run(_run())

    def test_write_audio_resamples_from_24k(self):
        from backend.sip_audio import SIPAudioAdapter

        written_data = bytearray()

        async def _run():
            reader = asyncio.StreamReader()
            reader.feed_eof()
            writer = MagicMock()
            writer.write = lambda data: written_data.extend(data)
            writer.drain = AsyncMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            adapter = SIPAudioAdapter(reader, writer)
            pcm_24k = struct.pack("<480h", *([500] * 480))
            await adapter.write_audio(pcm_24k)
            assert len(written_data) == 3 + 320  # header + 160 samples at 8kHz
            assert written_data[0] == AUDIO_TYPE

        asyncio.run(_run())

    def test_close_is_idempotent(self):
        from backend.sip_audio import SIPAudioAdapter

        async def _run():
            reader = asyncio.StreamReader()
            writer = MagicMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            adapter = SIPAudioAdapter(reader, writer)
            await adapter.close()
            await adapter.close()  # second call should not raise
            assert adapter._closed is True

        asyncio.run(_run())


class TestAMICallerLookup:
    def test_query_caller_id_returns_phone(self):
        from backend.sip_audio import query_caller_id_ami

        async def _run():
            phone = await query_caller_id_ami(
                channel_id="test-uuid",
                ami_host="127.0.0.1",
                ami_port=5038,
                ami_username="voicebot",
                ami_secret="test",
                _mock_response={"CallerIDNum": "375291234567"},
            )
            assert phone == "375291234567"

        asyncio.run(_run())

    def test_query_caller_id_returns_none_on_failure(self):
        from backend.sip_audio import query_caller_id_ami

        async def _run():
            phone = await query_caller_id_ami(
                channel_id="test-uuid",
                ami_host="127.0.0.1",
                ami_port=5038,
                ami_username="voicebot",
                ami_secret="test",
                _mock_response=None,
            )
            assert phone is None

        asyncio.run(_run())
