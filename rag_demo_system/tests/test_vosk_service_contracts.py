from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_vosk_server_contract_exists() -> None:
    content = (ROOT / "services" / "vosk_server.py").read_text(encoding="utf-8")

    assert 'title="Vosk STT Service"' in content
    assert '@app.get("/health")' in content
    assert '@app.post("/transcribe")' in content
    assert '"provider": "vosk"' in content


def test_vosk_tts_server_contract_exists() -> None:
    content = (ROOT / "services" / "vosk_tts_server.py").read_text(encoding="utf-8")

    assert 'title="Vosk TTS Service"' in content
    assert '@app.get("/health")' in content
    assert '@app.post("/speak")' in content
    assert '"provider": "vosk_tts"' in content
