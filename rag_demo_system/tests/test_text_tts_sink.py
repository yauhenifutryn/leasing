import asyncio
import pytest

from backend.execute_adapters import TextTtsSink


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_text_tts_sink_collects_text(loop):
    captured: list[dict] = []

    async def fake_broadcast(event):
        captured.append(event)

    sink = TextTtsSink(session_id="chat-abc12345", broadcast_fn=fake_broadcast)
    loop.run_until_complete(sink.say("Привет!"))
    loop.run_until_complete(sink.say("Чем помочь?"))

    assert sink.collected == ["Привет!", "Чем помочь?"]
    assert len(captured) == 2
    assert captured[0]["type"] == "sip.llm.sentence"
    assert captured[0]["call_id"] == "chat-abc12345"
    assert captured[0]["text"] == "Привет!"


def test_text_tts_sink_disconnect_returns_true(loop):
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    assert loop.run_until_complete(sink.disconnect()) is True


def test_text_tts_sink_drain_is_noop(loop):
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    loop.run_until_complete(sink.await_playback_drain())
    assert sink.total_audio_seconds == 0.0


def test_text_tts_sink_handles_none_broadcast(loop):
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    loop.run_until_complete(sink.say("test"))
    assert sink.collected == ["test"]


def test_text_tts_sink_skips_empty_text(loop):
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    loop.run_until_complete(sink.say(""))
    assert sink.collected == []
