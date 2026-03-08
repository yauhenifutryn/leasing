from pathlib import Path
import base64
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.whisper_server import create_app


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls = []

    def transcribe_pcm16(self, audio_bytes: bytes, sample_rate_hz: int, language: str) -> str:
        self.calls.append(
            {
                "audio_bytes": audio_bytes,
                "sample_rate_hz": sample_rate_hz,
                "language": language,
            }
        )
        return "Тестовая расшифровка"


def test_whisper_server_transcribes_pcm16_payload() -> None:
    transcriber = FakeTranscriber()
    client = TestClient(create_app(transcriber))

    audio_b64 = base64.b64encode(b"\x01\x02\x03\x04").decode("ascii")
    resp = client.post(
        "/transcribe",
        json={
            "audio_b64": audio_b64,
            "session_id": "s1",
            "language": "ru",
            "sample_rate_hz": 24000,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["text"] == "Тестовая расшифровка"
    assert resp.json()["provider"] == "whisper"
    assert transcriber.calls == [
        {
            "audio_bytes": b"\x01\x02\x03\x04",
            "sample_rate_hz": 24000,
            "language": "ru",
        }
    ]
