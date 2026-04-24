"""Unit tests for the production adapters that bridge apply_turn +
execute_action to the real orchestrator plumbing.

Adapter behaviour is lightweight glue; these tests verify the shape
contract only (what execute_action consumes) plus a few edge cases the
real runtime hits (RAG task failure, empty audio, WebSocket send fail).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================ LLMStreamBackend


@pytest.mark.asyncio
async def test_llm_stream_backend_yields_content_tokens(monkeypatch) -> None:
    from backend import execute_adapters as mod

    captured_calls: list[dict] = []

    def _fake_iter(**kwargs):
        captured_calls.append(kwargs)
        yield {"choices": [{"delta": {"content": "hello"}}]}
        yield {"choices": [{"delta": {"content": " world"}}]}
        yield {"choices": [{"delta": {"role": "assistant"}}]}

    monkeypatch.setattr(mod, "iter_openai_compatible_stream_events", _fake_iter)

    backend = mod.LLMStreamBackend(
        base_url="http://x", model="qwen",
        temperature=0.2, max_tokens=200, timeout_sec=10.0,
        system_prompt="system say",
    )
    tokens: list[str] = []
    async for tok in backend.stream(messages=[{"role": "user", "content": "hi"}]):
        tokens.append(tok)

    assert tokens == ["hello", " world"]
    assert captured_calls and captured_calls[0]["model"] == "qwen"
    msgs = captured_calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "system say"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert captured_calls[0]["tools"] is None


@pytest.mark.asyncio
async def test_llm_stream_backend_does_not_duplicate_system_prompt(monkeypatch) -> None:
    from backend import execute_adapters as mod

    captured: list[dict] = []

    def _fake_iter(**kwargs):
        captured.append(kwargs)
        if False:
            yield {}  # pragma: no cover

    monkeypatch.setattr(mod, "iter_openai_compatible_stream_events", _fake_iter)

    backend = mod.LLMStreamBackend(
        base_url="http://x", model="qwen",
        temperature=0.2, max_tokens=200, timeout_sec=10.0,
        system_prompt="would-be-duplicate",
    )
    async for _ in backend.stream(messages=[
        {"role": "system", "content": "caller-provided"},
        {"role": "user", "content": "hi"},
    ]):
        pass

    msgs = captured[0]["messages"]
    system_msgs = [m for m in msgs if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "caller-provided"


# ============================================================ TtsSink


class _RecordingWebSocket:
    def __init__(self, fail_on: str | None = None) -> None:
        self.sent: list[dict] = []
        self._fail_on = fail_on

    async def send_json(self, payload: dict) -> None:
        if self._fail_on and payload.get("type") == self._fail_on:
            raise RuntimeError("ws dead")
        self.sent.append(payload)


class _BareSession:
    def __init__(self) -> None:
        self.interrupted = False


@pytest.mark.asyncio
async def test_tts_sink_emits_text_and_audio(monkeypatch) -> None:
    from backend import execute_adapters as mod

    def _fake_synth(text, session_id):
        return {"audio_b64": "AAAA", "sample_rate_hz": 24000}

    monkeypatch.setattr(mod, "synthesize_audio", _fake_synth)

    ws = _RecordingWebSocket()
    sink = mod.TtsSink(
        websocket=ws, session_id="s1",
        session=_BareSession(), rtc_handler=None,
    )
    await sink.say("Hi there.")

    types_sent = [p["type"] for p in ws.sent]
    assert types_sent == [
        "response.output_text.delta",
        "response.output_audio.delta",
    ]
    assert ws.sent[0]["delta"] == "Hi there. "
    assert ws.sent[1]["delta"] == "AAAA"
    assert ws.sent[1]["sample_rate_hz"] == 24000


@pytest.mark.asyncio
async def test_tts_sink_pushes_rtc_audio_when_rtc_handler_present(monkeypatch) -> None:
    from backend import execute_adapters as mod

    monkeypatch.setattr(mod, "synthesize_audio", lambda t, s: {
        "audio_b64": "AAAA", "sample_rate_hz": 24000,
    })

    class _Track:
        def __init__(self) -> None: self.pushed: list[bytes] = []
        def push_audio(self, pcm: bytes) -> None: self.pushed.append(pcm)

    class _Handler:
        def __init__(self) -> None: self.tts_track = _Track()

    rtc = _Handler()
    ws = _RecordingWebSocket()
    sink = mod.TtsSink(websocket=ws, session_id="s1", session=_BareSession(), rtc_handler=rtc)
    await sink.say("hi")

    assert rtc.tts_track.pushed  # raw PCM reached the track
    assert [p["type"] for p in ws.sent] == ["response.output_text.delta"]


@pytest.mark.asyncio
async def test_tts_sink_flags_session_interrupted_on_ws_failure(monkeypatch) -> None:
    from backend import execute_adapters as mod

    monkeypatch.setattr(mod, "synthesize_audio", lambda t, s: {"audio_b64": "AAAA"})

    ws = _RecordingWebSocket(fail_on="response.output_text.delta")
    sess = _BareSession()
    sink = mod.TtsSink(websocket=ws, session_id="s1", session=sess, rtc_handler=None)
    await sink.say("hi")

    assert sess.interrupted is True


@pytest.mark.asyncio
async def test_tts_sink_tolerates_empty_audio_payload(monkeypatch) -> None:
    from backend import execute_adapters as mod

    monkeypatch.setattr(mod, "synthesize_audio", lambda t, s: {"audio_b64": ""})

    ws = _RecordingWebSocket()
    sink = mod.TtsSink(websocket=ws, session_id="s1", session=_BareSession())
    await sink.say("hi")

    assert [p["type"] for p in ws.sent] == ["response.output_text.delta"]


# ============================================================ CalcAdapter


@pytest.mark.asyncio
async def test_calc_adapter_fills_defaults_and_executes(monkeypatch) -> None:
    from backend import execute_adapters as mod

    recorded: dict = {}

    class _FakeTool:
        def fill_defaults(self, params):
            filled = dict(params)
            filled.setdefault("currency", "BYN")
            return filled, {"currency": "BYN"}

        def execute(self, params, ctx):
            recorded["params"] = params
            recorded["ctx"] = ctx
            return {"ok": True, "params": params, "advance_sum": 100}

    monkeypatch.setattr(mod, "get_all_tools", lambda: {"calculator": _FakeTool()})

    adapter = mod.CalcAdapter(session_id="sid-x", client_phone="+375291111111")
    result = await adapter.calculate({"cost": 100})

    assert result["ok"] is True
    assert result["advance_sum"] == 100
    assert result.get("defaulted") == {"currency": "BYN"}
    assert recorded["params"] == {"cost": 100, "currency": "BYN"}
    assert recorded["ctx"] == {
        "session_id": "sid-x", "client_phone": "+375291111111",
    }


@pytest.mark.asyncio
async def test_calc_adapter_raises_when_tool_registry_misses(monkeypatch) -> None:
    from backend import execute_adapters as mod
    monkeypatch.setattr(mod, "get_all_tools", lambda: {})
    with pytest.raises(RuntimeError, match="calculator tool unavailable"):
        await mod.CalcAdapter(session_id="s1").calculate({})


@pytest.mark.asyncio
async def test_calc_adapter_broadcasts_sip_tool_events_on_success(monkeypatch) -> None:
    """Monitor UI expects sip.tool.start before calc + sip.tool.result
    after so the operator sees 'Tool: calculator' / 'Tool done' lines.
    Legacy DirectTool emits these at app.py:2415 / 2474. Adapter must
    mirror or UI is silent even when calc fires."""
    from backend import execute_adapters as mod

    class _FakeTool:
        def fill_defaults(self, params):
            return dict(params), {}

        def execute(self, params, ctx):
            return {"ok": True, "params": params, "advance_sum": 72000}

    monkeypatch.setattr(mod, "get_all_tools", lambda: {"calculator": _FakeTool()})

    broadcasts: list[dict] = []

    async def _capture(event: dict) -> None:
        broadcasts.append(event)

    monkeypatch.setattr(mod.CalcAdapter, "_broadcast", staticmethod(_capture))

    adapter = mod.CalcAdapter(session_id="sid-live", client_phone=None)
    result = await adapter.calculate({"cost": 240000})

    assert result["ok"] is True
    types_sent = [b["type"] for b in broadcasts]
    assert types_sent == ["sip.tool.start", "sip.tool.result"]
    assert broadcasts[0]["call_id"] == "sid-live"
    assert broadcasts[0]["tool"] == "calculator"
    assert broadcasts[0]["params"] == {"cost": 240000}
    assert broadcasts[1]["ok"] is True


@pytest.mark.asyncio
async def test_calc_adapter_broadcasts_tool_failure_on_exception(monkeypatch) -> None:
    """Upstream calc exception → sip.tool.result with ok=False, then
    the exception propagates so the outer FireCalc handler can bump the
    circuit breaker."""
    from backend import execute_adapters as mod

    class _FakeTool:
        def fill_defaults(self, params):
            return dict(params), {}

        def execute(self, params, ctx):
            raise RuntimeError("calc upstream 500")

    monkeypatch.setattr(mod, "get_all_tools", lambda: {"calculator": _FakeTool()})

    broadcasts: list[dict] = []
    async def _capture(event: dict) -> None:
        broadcasts.append(event)
    monkeypatch.setattr(mod.CalcAdapter, "_broadcast", staticmethod(_capture))

    with pytest.raises(RuntimeError, match="calc upstream 500"):
        await mod.CalcAdapter(session_id="s").calculate({"cost": 100})

    types_sent = [b["type"] for b in broadcasts]
    assert types_sent == ["sip.tool.start", "sip.tool.result"]
    assert broadcasts[1]["ok"] is False


# ============================================================ RagFuture


@pytest.mark.asyncio
async def test_rag_future_resolves_chunks_into_fragment_block() -> None:
    from backend import execute_adapters as mod

    async def _produce() -> dict:
        return {"final": [
            {"text": "address Minsk"},
            {"text": "annuity schedule"},
        ]}

    task = asyncio.create_task(_produce())
    fut = mod.RagFuture(task)
    ctx = await fut.result()

    assert "[Fragment 1]" in ctx
    assert "address Minsk" in ctx
    assert "[Fragment 2]" in ctx
    assert "annuity schedule" in ctx


@pytest.mark.asyncio
async def test_rag_future_caches_result() -> None:
    from backend import execute_adapters as mod

    call_count = 0

    async def _produce() -> dict:
        nonlocal call_count
        call_count += 1
        return {"final": [{"text": "x"}]}

    task = asyncio.create_task(_produce())
    fut = mod.RagFuture(task)
    first = await fut.result()
    second = await fut.result()

    assert first == second
    assert call_count == 1


@pytest.mark.asyncio
async def test_rag_future_handles_empty_retrieval() -> None:
    from backend import execute_adapters as mod

    async def _empty() -> dict:
        return {}

    task = asyncio.create_task(_empty())
    fut = mod.RagFuture(task)
    assert await fut.result() == ""
