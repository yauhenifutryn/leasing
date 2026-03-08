from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import voice_adapters


class FakeResponse:
    def __init__(self, *, json_data=None, content=b""):
        self._json = json_data or {}
        self.content = content

    @property
    def ok(self) -> bool:
        return True

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json


def test_transcribe_audio_supports_official_sensevoice_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("SENSEVOICE_BASE_URL", "http://sensevoice.local")
    monkeypatch.setenv("SENSEVOICE_API_STYLE", "official")
    monkeypatch.delenv("WHISPER_BASE_URL", raising=False)

    def fake_post(url, files=None, data=None, timeout=None, json=None):
        calls.append({"url": url, "files": files, "data": data, "json": json})
        return FakeResponse(json_data={"result": [{"text": "Привет"}]})

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.transcribe_audio("AQIDBA==", session_id="s1")

    assert data["text"] == "Привет"
    assert data["provider"] == "sensevoice"
    assert calls[0]["url"] == "http://sensevoice.local/api/v1/asr"
    assert calls[0]["data"]["lang"] == "auto"
    assert calls[0]["json"] is None
    assert calls[0]["files"][0][0] == "files"


def test_synthesize_audio_supports_official_cosyvoice_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("COSYVOICE_BASE_URL", "http://cosyvoice.local")
    monkeypatch.setenv("COSYVOICE_API_STYLE", "official")
    monkeypatch.setenv("COSYVOICE_SPK_ID", "russian_female")

    def fake_post(url, data=None, timeout=None, json=None):
        calls.append({"url": url, "data": data, "json": json})
        return FakeResponse(content=b"\x01\x02\x03\x04")

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.synthesize_audio("Здравствуйте", session_id="s2")

    assert data["provider"] == "cosyvoice"
    assert data["audio_b64"] == "AQIDBA=="
    assert data["sample_rate_hz"] == 22050
    assert calls[0]["url"] == "http://cosyvoice.local/inference_sft"
    assert calls[0]["data"] == {
        "tts_text": "Здравствуйте",
        "spk_id": "russian_female",
    }
    assert calls[0]["json"] is None
