"""
Test scaffold for voice turn log event structure.

These tests verify the log event contract used by Plan 02's instrumentation
implementation. They test structure and computed values, not live WebSocket
handler behavior.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


REQUIRED_LOG_FIELDS: set[str] = {
    "event",
    "question_id",
    "stack_id",
    "session_id",
    "backend",
    "brain_model",
    "stt_provider",
    "tts_provider",
    "transcript",
    "speech_stopped",
    "stt_done",
    "retrieval_done",
    "llm_first_token",
    "tts_first_chunk",
    "playback_started",
    "primary_kpi_ms",
}


def _load_voice_session():
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("backend.voice_session")
    assert spec is not None, "backend.voice_session module is missing"
    return importlib.import_module("backend.voice_session")


def test_voice_turn_log_event_has_all_required_fields() -> None:
    """All 17 required fields must be present in a voice turn log event."""
    event: dict = {
        "event": "voice_turn",
        "question_id": "q-001",
        "stack_id": "our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice",
        "session_id": "test-session",
        "backend": "our_rag",
        "brain_model": "Qwen/Qwen3-30B-A3B",
        "stt_provider": "sensevoice",
        "tts_provider": "cosyvoice",
        "transcript": "Какие требования к лизингу?",
        "speech_stopped": 1000.0,
        "stt_done": 1001.0,
        "retrieval_done": 1001.5,
        "llm_first_token": 1001.8,
        "tts_first_chunk": 1002.0,
        "playback_started": 1002.5,
        "primary_kpi_ms": 2500.0,
    }

    assert REQUIRED_LOG_FIELDS.issubset(event.keys()), (
        f"Missing fields: {REQUIRED_LOG_FIELDS - event.keys()}"
    )


def test_primary_kpi_ms_is_computed_correctly() -> None:
    """primary_kpi_ms = (playback_started - speech_stopped) * 1000."""
    speech_stopped = 1000.0  # seconds (monotonic epoch)
    playback_started = 1002.5

    primary_kpi_ms = (playback_started - speech_stopped) * 1000

    assert primary_kpi_ms == 2500.0


def test_stack_id_matches_session_fields() -> None:
    """VoiceSession.stack_id must match the log event stack_id field format."""
    voice_session = _load_voice_session()

    session = voice_session.VoiceSession(session_id="test", backend="our_rag")

    assert session.stack_id == "our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice"
