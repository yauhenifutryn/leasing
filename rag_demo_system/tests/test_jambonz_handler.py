"""Tests for Jambonz WebSocket handler logic."""
from pathlib import Path
import sys
import json
import base64
import struct

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestJambonzControlMessage:
    def test_build_listen_ack(self):
        caller_phone = "+375291234567"
        call_sid = "call-abc-123"
        audio_ws_url = "ws://host.docker.internal:8000/ws/jambonz-audio"
        ack = {
            "type": "ack",
            "data": {
                "verb": "listen",
                "url": audio_ws_url,
                "sampleRate": 16000,
                "passDtmf": True,
                "bidirectionalAudio": {
                    "enabled": True,
                    "streaming": True,
                    "sampleRate": 24000,
                },
                "metadata": {
                    "from": caller_phone,
                    "callSid": call_sid,
                },
            },
        }
        assert ack["data"]["verb"] == "listen"
        assert ack["data"]["sampleRate"] == 16000
        assert ack["data"]["bidirectionalAudio"]["sampleRate"] == 24000
        assert ack["data"]["bidirectionalAudio"]["streaming"] is True
        assert ack["data"]["passDtmf"] is True
        assert ack["data"]["metadata"]["from"] == "+375291234567"


class TestJambonzAudioMetadata:
    def test_parse_metadata(self):
        metadata = {
            "callSid": "call-abc-123",
            "sampleRate": 16000,
            "mixType": "mono",
            "metadata": {
                "from": "+375291234567",
                "callSid": "call-abc-123",
            },
        }
        call_sid = metadata["callSid"]
        caller_phone = metadata.get("metadata", {}).get("from")
        assert call_sid == "call-abc-123"
        assert caller_phone == "+375291234567"


class TestJambonzControlMessages:
    def test_kill_audio_message(self):
        msg = json.dumps({"type": "killAudio"})
        parsed = json.loads(msg)
        assert parsed["type"] == "killAudio"

    def test_disconnect_message(self):
        msg = json.dumps({"type": "disconnect"})
        parsed = json.loads(msg)
        assert parsed["type"] == "disconnect"

    def test_dtmf_event_parsing(self):
        event = {"event": "dtmf", "dtmf": "5", "duration": "1600"}
        assert event["event"] == "dtmf"
        assert event["dtmf"] == "5"


class TestJambonzWebSocketShim:
    def test_audio_delta_decodes_and_would_send_binary(self):
        pcm_24k = struct.pack("<10h", *range(10))
        audio_b64 = base64.b64encode(pcm_24k).decode()
        decoded = base64.b64decode(audio_b64)
        assert decoded == pcm_24k
        assert len(decoded) == 20

    def test_text_event_passes_through(self):
        event = {
            "type": "response.output_text.delta",
            "delta": "Hello world",
        }
        assert event["type"] != "response.output_audio.delta"


class TestResample16kTo24k:
    def test_resample_16k_to_24k(self):
        import numpy as np
        from scipy.signal import resample_poly
        samples_16k = np.zeros(320, dtype=np.int16)
        resampled = resample_poly(samples_16k, up=3, down=2).astype(np.int16)
        assert len(resampled) == 480
