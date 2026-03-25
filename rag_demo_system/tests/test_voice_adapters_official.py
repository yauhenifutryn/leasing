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


def test_transcribe_audio_supports_yandex_speechkit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("YC_FOLDER_ID", "folder-1")
    monkeypatch.setenv("YC_API_KEY", "key-1")

    def fake_post(url, headers=None, data=None, timeout=None, files=None, json=None):
        calls.append({"url": url, "headers": headers, "data": data, "files": files, "json": json})
        return FakeResponse(json_data={"result": "Добрый день"})

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.transcribe_audio("AQIDBA==", session_id="s3", preferred="yandex_speechkit")

    assert data["text"] == "Добрый день"
    assert data["provider"] == "yandex_speechkit"
    assert calls[0]["url"].startswith("https://stt.api.cloud.yandex.net/speech/v1/stt:recognize?")
    assert calls[0]["headers"]["Authorization"] == "Api-Key key-1"
    assert calls[0]["headers"]["x-folder-id"] == "folder-1"
    assert calls[0]["json"] is None


def test_synthesize_audio_supports_yandex_speechkit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("YC_FOLDER_ID", "folder-1")
    monkeypatch.setenv("YC_API_KEY", "key-1")
    monkeypatch.setenv("YC_VOICE", "ermil")

    def fake_post(url, headers=None, data=None, timeout=None, json=None, files=None):
        calls.append({"url": url, "headers": headers, "data": data, "json": json, "files": files})
        return FakeResponse(content=b"\x01\x02\x03\x04")

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.synthesize_audio("Здравствуйте", session_id="s4", preferred="yandex_speechkit")

    assert data["provider"] == "yandex_speechkit"
    assert data["audio_b64"] == "AQIDBA=="
    assert data["sample_rate_hz"] == 48000
    assert calls[0]["url"] == "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    assert calls[0]["headers"]["Authorization"] == "Api-Key key-1"
    assert calls[0]["data"]["voice"] == "ermil"
    assert calls[0]["json"] is None


def test_transcribe_audio_supports_vosk_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("VOSK_BASE_URL", "http://vosk.local")

    def fake_post(url, headers=None, data=None, timeout=None, files=None, json=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(json_data={"text": "Лизинг одобрен", "provider": "vosk"})

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.transcribe_audio("AQIDBA==", session_id="s5", preferred="vosk")

    assert data["text"] == "Лизинг одобрен"
    assert data["provider"] == "vosk"
    assert calls[0]["url"] == "http://vosk.local/transcribe"
    assert calls[0]["json"]["sample_rate_hz"] == 24000


def test_synthesize_audio_supports_vosk_tts_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("VOSK_TTS_BASE_URL", "http://vosk-tts.local")

    def fake_post(url, headers=None, data=None, timeout=None, files=None, json=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(
            json_data={
                "audio_b64": "AQIDBA==",
                "sample_rate_hz": 22050,
                "provider": "vosk_tts",
                "session_id": "s6",
            }
        )

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.synthesize_audio("Здравствуйте", session_id="s6", preferred="vosk_tts")

    assert data["provider"] == "vosk_tts"
    assert data["audio_b64"] == "AQIDBA=="
    assert calls[0]["url"] == "http://vosk-tts.local/speak"
    assert calls[0]["json"] == {
        "text": "Здравствуйте",
        "session_id": "s6",
        "language": "ru",
    }


def test_transcribe_audio_supports_qwen3_asr_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("QWEN3_ASR_BASE_URL", "http://qwen3-asr.local")
    monkeypatch.delenv("SENSEVOICE_BASE_URL", raising=False)
    monkeypatch.delenv("WHISPER_BASE_URL", raising=False)

    def fake_post(url, headers=None, data=None, timeout=None, files=None, json=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(json_data={"text": "Лизинг одобрен", "provider": "qwen3_asr"})

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.transcribe_audio("AQIDBA==", session_id="s1", preferred="qwen3_asr")

    assert data["text"] == "Лизинг одобрен"
    assert data["provider"] == "qwen3_asr"
    assert calls[0]["url"] == "http://qwen3-asr.local/transcribe"
    assert calls[0]["json"]["sample_rate_hz"] == 24000


def test_synthesize_audio_supports_qwen3_tts_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("QWEN3_TTS_BASE_URL", "http://qwen3-tts.local")

    def fake_post(url, headers=None, data=None, timeout=None, files=None, json=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(
            json_data={
                "audio_b64": "AQIDBA==",
                "sample_rate_hz": 24000,
                "provider": "qwen3_tts",
                "session_id": "s1",
            }
        )

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.synthesize_audio("Здравствуйте", session_id="s1", preferred="qwen3_tts")

    assert data["provider"] == "qwen3_tts"
    assert data["audio_b64"] == "AQIDBA=="
    assert calls[0]["url"] == "http://qwen3-tts.local/speak"
    assert calls[0]["json"] == {"text": "Здравствуйте", "session_id": "s1", "language": "ru"}


def test_transcribe_audio_supports_voxtral_service(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setenv("VOXTRAL_BASE_URL", "http://voxtral.local")
    monkeypatch.delenv("SENSEVOICE_BASE_URL", raising=False)
    monkeypatch.delenv("WHISPER_BASE_URL", raising=False)

    def fake_post(url, headers=None, data=None, timeout=None, files=None, json=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(json_data={"text": "Лизинг одобрен", "provider": "voxtral"})

    monkeypatch.setattr(voice_adapters.requests, "post", fake_post)

    data = voice_adapters.transcribe_audio("AQIDBA==", session_id="s1", preferred="voxtral")

    assert data["text"] == "Лизинг одобрен"
    assert data["provider"] == "voxtral"
    assert calls[0]["url"] == "http://voxtral.local/transcribe"
    assert calls[0]["json"]["sample_rate_hz"] == 24000


def test_qwen3_asr_hard_fail_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN3_ASR_BASE_URL", raising=False)
    monkeypatch.delenv("SENSEVOICE_BASE_URL", raising=False)
    monkeypatch.delenv("WHISPER_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="QWEN3_ASR_BASE_URL"):
        voice_adapters.transcribe_audio("AQIDBA==", session_id="s1", preferred="qwen3_asr")


def test_qwen3_tts_hard_fail_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN3_TTS_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="QWEN3_TTS_BASE_URL"):
        voice_adapters.synthesize_audio_with_provider("Здравствуйте", session_id="s1", preferred="qwen3_tts")


def test_voxtral_hard_fail_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXTRAL_BASE_URL", raising=False)
    monkeypatch.delenv("SENSEVOICE_BASE_URL", raising=False)
    monkeypatch.delenv("WHISPER_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="VOXTRAL_BASE_URL"):
        voice_adapters.transcribe_audio("AQIDBA==", session_id="s1", preferred="voxtral")
