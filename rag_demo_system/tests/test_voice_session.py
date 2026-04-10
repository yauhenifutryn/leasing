from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module():
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("backend.voice_session")
    assert spec is not None, "backend.voice_session module is missing"
    return importlib.import_module("backend.voice_session")


def test_audio_chunk_interrupts_assistant_playback() -> None:
    voice_session = _load_module()
    session = voice_session.VoiceSession(session_id="s1", backend="dify_rag")
    session.assistant_speaking = True
    session.active_task_id = "task-7"

    events = session.on_audio_chunk("ZmFrZQ==")

    assert events == [
        {
            "type": "interrupt",
            "session_id": "s1",
            "task_id": "task-7",
            "backend": "dify_rag",
        }
    ]
    assert session.assistant_speaking is False
    assert session.interrupted is True


def test_final_transcript_dispatches_selected_backend() -> None:
    voice_session = _load_module()
    session = voice_session.VoiceSession(session_id="s2", backend="our_rag")

    events = session.on_transcript_final("Какие требования к лизингу?")

    assert events == [
        {
            "type": "dispatch_message",
            "session_id": "s2",
            "backend": "our_rag",
            "message": "Какие требования к лизингу?",
            "voice_fast": True,
        }
    ]
    assert session.last_user_message == "Какие требования к лизингу?"


def test_provider_response_marks_assistant_speaking_and_captures_task() -> None:
    voice_session = _load_module()
    session = voice_session.VoiceSession(session_id="s3", backend="dify_rag")

    events = session.on_provider_response(
        {
            "backend": "dify_rag",
            "answer": "Здравствуйте",
            "conversation_ref": {"task_id": "task-11"},
            "can_barge_in": True,
        }
    )

    assert events == [
        {
            "type": "assistant_response",
            "session_id": "s3",
            "backend": "dify_rag",
            "answer": "Здравствуйте",
            "can_barge_in": True,
        }
    ]
    assert session.assistant_speaking is True
    assert session.active_task_id == "task-11"


def test_default_brain_model() -> None:
    voice_session = _load_module()

    session = voice_session.VoiceSession(session_id="s1")

    assert session.brain_model == "Qwen/Qwen3.5-35B-A3B-FP8"


def test_default_stt_provider() -> None:
    voice_session = _load_module()

    session = voice_session.VoiceSession(session_id="s1")

    assert session.stt_provider == "whisper"


def test_default_tts_provider() -> None:
    voice_session = _load_module()

    session = voice_session.VoiceSession(session_id="s1")

    assert session.tts_provider == "silero_tts"


def test_stack_id_composition() -> None:
    voice_session = _load_module()

    session = voice_session.VoiceSession(
        session_id="s1",
        backend="our_rag",
        brain_model="Qwen/Qwen3-30B-A3B",
        stt_provider="sensevoice",
        tts_provider="cosyvoice",
    )

    assert session.stack_id == "our_rag__Qwen3-30B-A3B__sensevoice__cosyvoice"


def test_stack_id_updates_on_field_change() -> None:
    voice_session = _load_module()

    session = voice_session.VoiceSession(session_id="s1")
    session.backend = "dify_rag"

    assert session.stack_id.startswith("dify_rag__")


class TestVoiceSessionSIPFields:
    def test_default_transport_is_websocket(self):
        from backend.voice_session import VoiceSession
        s = VoiceSession(session_id="test-1")
        assert s.transport == "websocket"
        assert s.client_phone is None
        assert s.call_id is None

    def test_sip_transport_fields(self):
        from backend.voice_session import VoiceSession
        s = VoiceSession(
            session_id="test-2",
            transport="sip",
            client_phone="375291234567",
            call_id="abc-123",
        )
        assert s.transport == "sip"
        assert s.client_phone == "375291234567"
        assert s.call_id == "abc-123"
