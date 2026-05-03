import pytest

from backend.execute_adapters import TextTtsSink


@pytest.mark.asyncio
async def test_text_tts_sink_collects_text():
    captured: list[dict] = []

    async def fake_broadcast(event):
        captured.append(event)

    sink = TextTtsSink(session_id="chat-abc12345", broadcast_fn=fake_broadcast)
    await sink.say("Привет!")
    await sink.say("Чем помочь?")

    assert sink.collected == ["Привет!", "Чем помочь?"]
    assert len(captured) == 2
    assert captured[0]["type"] == "sip.llm.sentence"
    assert captured[0]["call_id"] == "chat-abc12345"
    assert captured[0]["text"] == "Привет!"


@pytest.mark.asyncio
async def test_text_tts_sink_disconnect_returns_true():
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    assert await sink.disconnect() is True


@pytest.mark.asyncio
async def test_text_tts_sink_drain_is_noop():
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    await sink.await_playback_drain()
    assert sink.total_audio_seconds == 0.0
    assert sink.collected == []  # added per review: stronger noop assertion


@pytest.mark.asyncio
async def test_text_tts_sink_handles_none_broadcast():
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    await sink.say("test")
    assert sink.collected == ["test"]


@pytest.mark.asyncio
async def test_text_tts_sink_skips_empty_text():
    sink = TextTtsSink(session_id="chat-x", broadcast_fn=None)
    await sink.say("")
    assert sink.collected == []
