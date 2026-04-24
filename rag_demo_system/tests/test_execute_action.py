"""Integration tests for execute_action — the pure-IO dispatcher
that consumes a TurnAction and emits TTS chunks.

Phase 3.D of the apply_turn refactor. Tests use fake TTS / LLM /
calc collaborators so they run in-process without the real FastAPI
WebSocket plumbing. Every emit path is verified to bypass the LLM
unless the TurnAction variant is FireLLMFallback.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.turn_action import (
    EmitReadback,
    EmitClarify,
    EmitChangeConfirm,
    FireCalc,
    FireLLMFallback,
    FireOORMessage,
    Noop,
    ProfileSnapshot,
)


# ---------------------------------------------------------------- fakes


class FakeTts:
    """Captures text chunks passed to say(). Async-compatible."""
    def __init__(self) -> None:
        self.chunks: list[str] = []

    async def say(self, text: str) -> None:
        self.chunks.append(text)

    def collected_text(self) -> str:
        return " ".join(self.chunks)


class FakeLLMBackend:
    """Counts stream() invocations so tests can assert LLM-bypass."""
    def __init__(self) -> None:
        self.call_count = 0

    async def stream(self, *args, **kwargs):
        self.call_count += 1
        if False:  # pragma: no cover — async generator shape
            yield ""


class FakeCalc:
    """Returns a preset calculator result on calculate(). Supports
    error injection via the `raises` kwarg."""
    def __init__(self, result: dict | None = None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.call_count = 0
        self.last_params: dict | None = None

    async def calculate(self, params: dict) -> dict:
        self.call_count += 1
        self.last_params = params
        if self.raises is not None:
            raise self.raises
        return self.result or {}


def _complete_snapshot_usd_to_byn() -> ProfileSnapshot:
    """Mirrors the call cc7fc318 scenario: Физлицо, legковой, 80k USD
    converted to 240k BYN for the calc."""
    return ProfileSnapshot(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=240000.0,
        currency="BYN",
        original_cost=80000.0,
        original_currency="USD",
        condition_new=1,
        age_years=None,
        prepaid_pct=30.0,
        prepaid_amount=None,
        term_months=36,
        type_schedule="0",
    )


# ---------------------------------------------------------------- E8b
# E8b — FireCalc handler runs the calculator, then passes the result
# to profile_prompts.render_calc_result, then ships the string to TTS.
# LLM is NEVER invoked. This is the structural E8 guarantee.


@pytest.mark.asyncio
async def test_e8b_fire_calc_ships_usd_disclosure_to_tts_without_llm() -> None:
    from backend.turn_dispatcher import execute_action

    calc_result = {
        "params": {"cost": 240000, "currency": "BYN", "prepaid": 30},
        "advance_sum": 72000,
        "payment_min": 8109,
        "buyout_sum": 10000,
        "total": 342317,
        "num_payments": 36,
        "increase_percent": 12.5,
        "currency_conversion": {
            "from": "USD",
            "amount_from": 80000,
            "rate": 3.0,
        },
    }
    tts = FakeTts()
    llm = FakeLLMBackend()
    calc = FakeCalc(result=calc_result)
    fire = FireCalc(
        snapshot=_complete_snapshot_usd_to_byn(),
        calc_params={"cost": 240000, "currency": "BYN", "prepaid": 30},
    )

    async for _ in execute_action(
        fire,
        ws=None,
        session=None,
        backend=llm,
        tts=tts,
        calc=calc,
        rag_future=None,
    ):
        pass

    spoken = tts.collected_text()
    # USD disclosure prefix must survive the path (E8 invariant).
    assert "Стоимость 80000 долларов" in spoken
    assert "по курсу 3 к 1" in spoken
    # LLM NEVER called during FireCalc handling.
    assert llm.call_count == 0
    # Calc was invoked with the params apply_turn computed.
    assert calc.call_count == 1
    assert calc.last_params == {"cost": 240000, "currency": "BYN", "prepaid": 30}


@pytest.mark.asyncio
async def test_e8b_fire_calc_propagates_calc_exception() -> None:
    # Phase 3.D Task 20 adds circuit-breaker semantics; for now the
    # scaffold just propagates exceptions so the orchestrator can log
    # and route to an OOR / fallback message.
    from backend.turn_dispatcher import execute_action

    tts = FakeTts()
    llm = FakeLLMBackend()
    calc = FakeCalc(raises=RuntimeError("calc upstream 500"))
    fire = FireCalc(
        snapshot=_complete_snapshot_usd_to_byn(),
        calc_params={"cost": 240000, "currency": "BYN"},
    )

    with pytest.raises(RuntimeError, match="calc upstream 500"):
        async for _ in execute_action(
            fire,
            ws=None,
            session=None,
            backend=llm,
            tts=tts,
            calc=calc,
            rag_future=None,
        ):
            pass

    # LLM still not called even on calc failure.
    assert llm.call_count == 0


# ---------------------------------------------------------------- EmitReadback

@pytest.mark.asyncio
async def test_emit_readback_speaks_deterministic_readback_without_llm() -> None:
    from backend.turn_dispatcher import execute_action

    snap = _complete_snapshot_usd_to_byn()
    tts = FakeTts()
    llm = FakeLLMBackend()

    async for _ in execute_action(
        EmitReadback(snapshot=snap),
        ws=None, session=None, backend=llm,
        tts=tts, calc=None, rag_future=None,
    ):
        pass

    spoken = tts.collected_text()
    # Readback mentions the captured subject verbatim (anti-hallucination
    # anchor per E7).
    assert "Легковой автомобиль" in spoken
    # LLM not involved.
    assert llm.call_count == 0


# ---------------------------------------------------------------- EmitClarify

@pytest.mark.asyncio
async def test_emit_clarify_asks_for_missing_fields_without_llm() -> None:
    from backend.turn_dispatcher import execute_action

    snap = ProfileSnapshot(
        client_type=None, subject=None,
        cost=80000.0, currency="USD",
        original_cost=None, original_currency=None,
        condition_new=None, age_years=None,
        prepaid_pct=None, prepaid_amount=None,
        term_months=None, type_schedule=None,
    )
    tts = FakeTts()
    llm = FakeLLMBackend()

    async for _ in execute_action(
        EmitClarify(missing=["client_type", "subject"], snapshot=snap),
        ws=None, session=None, backend=llm,
        tts=tts, calc=None, rag_future=None,
    ):
        pass

    spoken = tts.collected_text()
    # build_clarification_prompt handles {client_type, subject} explicitly.
    assert spoken  # non-empty
    assert llm.call_count == 0


# ---------------------------------------------------------------- EmitChangeConfirm

@pytest.mark.asyncio
async def test_emit_change_confirm_asks_to_confirm_change_without_llm() -> None:
    from backend.turn_dispatcher import execute_action

    snap = _complete_snapshot_usd_to_byn()
    tts = FakeTts()
    llm = FakeLLMBackend()

    async for _ in execute_action(
        EmitChangeConfirm(
            changes={"term_months": {"old": 36, "new": 60}},
            snapshot=snap,
        ),
        ws=None, session=None, backend=llm,
        tts=tts, calc=None, rag_future=None,
    ):
        pass

    spoken = tts.collected_text()
    assert spoken  # non-empty (build_change_confirm_text produces a phrase)
    assert llm.call_count == 0


# ---------------------------------------------------------------- Noop / OOR

@pytest.mark.asyncio
async def test_noop_emits_nothing() -> None:
    from backend.turn_dispatcher import execute_action

    tts = FakeTts()
    llm = FakeLLMBackend()

    async for _ in execute_action(
        Noop(reason="test"),
        ws=None, session=None, backend=llm,
        tts=tts, calc=None, rag_future=None,
    ):
        pass

    assert tts.chunks == []
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_fire_oor_message_speaks_deterministic_text_without_llm() -> None:
    from backend.turn_dispatcher import execute_action

    tts = FakeTts()
    llm = FakeLLMBackend()

    async for _ in execute_action(
        FireOORMessage(message="Стоимость вне допустимого диапазона."),
        ws=None, session=None, backend=llm,
        tts=tts, calc=None, rag_future=None,
    ):
        pass

    assert "вне допустимого диапазона" in tts.collected_text()
    assert llm.call_count == 0


# ---------------------------------------------------------------- FireLLMFallback
# Task 18 — FireLLMFallback handler. Mirrors app.py:2513-2778 streaming
# semantics: tokens flow through SentenceDetector, phrase-boundary
# sentences ship to TTS one at a time, LLM is the ONLY path that awaits
# rag_future (spec §7.2 invariants #1, #2, #5).


class StreamingLLMBackend:
    """Fake streaming backend: async gen of content tokens.

    Matches the FireLLMFallback handler's contract — `backend.stream(...)`
    yields raw text tokens (no OpenAI event envelope). Production adapter
    wraps `iter_openai_compatible_stream_events` to extract deltas.
    """
    def __init__(self, tokens, per_token_sleep: float = 0.0) -> None:
        self.tokens = list(tokens)
        self.per_token_sleep = per_token_sleep
        self.call_count = 0
        self.last_kwargs: dict | None = None

    async def stream(self, *args, **kwargs):
        import asyncio as _asyncio
        self.call_count += 1
        self.last_kwargs = kwargs
        for token in self.tokens:
            if self.per_token_sleep:
                await _asyncio.sleep(self.per_token_sleep)
            yield token


class ReadyRAG:
    """Pre-resolved RAG future — `.result()` is awaitable and returns
    the context string synchronously. Mirrors the adapter the
    orchestrator wraps around `asyncio.create_task(engine.retrieve, ...)`.
    """
    def __init__(self, context: str = "") -> None:
        self.context = context
        self.result_calls = 0

    async def result(self) -> str:
        self.result_calls += 1
        return self.context


class InterruptingSession:
    """Session stand-in that flips `interrupted=True` after the Nth
    sentence boundary. Simulates barge-in mid-readback (invariant #5).
    """
    def __init__(self, flip_after_n_sentences: int) -> None:
        self.interrupted = False
        self._seen = 0
        self._limit = flip_after_n_sentences

    def observe_sentence(self) -> None:
        self._seen += 1
        if self._seen >= self._limit:
            self.interrupted = True


@pytest.mark.asyncio
async def test_fire_llm_fallback_first_token_latency_under_budget() -> None:
    """Invariant #1 + #2: the handler adds negligible overhead to the
    LLM's own token cadence. With a fake streaming at 20 ms/token, the
    first sentence must reach TTS within a 100 ms synthetic budget.
    """
    import time
    from backend.turn_dispatcher import execute_action

    # First boundary hits after 2 tokens (period + trailing space).
    tokens = ["Здрав", "ствуйте. ", "Как ", "дела?"]
    backend = StreamingLLMBackend(tokens, per_token_sleep=0.02)
    tts = FakeTts()

    start = time.monotonic()
    first_chunk_at: float | None = None
    async for _chunk in execute_action(
        FireLLMFallback(user_utterance="привет"),
        ws=None, session=None, backend=backend, tts=tts, calc=None,
        rag_future=ReadyRAG(context="кб-контекст"),
    ):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic() - start
            assert first_chunk_at < 0.1, f"first chunk too late: {first_chunk_at:.3f}s"

    assert first_chunk_at is not None
    # Handler invoked the backend exactly once.
    assert backend.call_count == 1
    # TTS saw at least two phrase-level sentences (no buffering regression).
    assert len(tts.chunks) >= 2


@pytest.mark.asyncio
async def test_fire_llm_fallback_awaits_rag_future_result() -> None:
    """Invariant #1: RAG overlap preservation. Handler MUST await
    `rag_future.result()` before streaming so the LLM prompt carries
    the KB context."""
    from backend.turn_dispatcher import execute_action

    backend = StreamingLLMBackend(["Ответ."], per_token_sleep=0.0)
    tts = FakeTts()
    rag = ReadyRAG(context="адрес: Минск, Немига 5")

    async for _ in execute_action(
        FireLLMFallback(user_utterance="где вы находитесь?"),
        ws=None, session=None, backend=backend, tts=tts, calc=None,
        rag_future=rag,
    ):
        pass

    # RAG future was awaited exactly once (no re-entry).
    assert rag.result_calls == 1
    # RAG context reached the backend prompt. We inspect last_kwargs to
    # assert the handler actually threaded rag_context into messages.
    msgs = (backend.last_kwargs or {}).get("messages", [])
    joined = " ".join(m.get("content", "") for m in msgs if isinstance(m, dict))
    assert "Немига 5" in joined


@pytest.mark.asyncio
async def test_fire_llm_fallback_tolerates_rag_future_none() -> None:
    """Handler must not crash when orchestrator passes `rag_future=None`
    (e.g. speculative RAG failed to launch)."""
    from backend.turn_dispatcher import execute_action

    backend = StreamingLLMBackend(["Здравствуйте. "], per_token_sleep=0.0)
    tts = FakeTts()

    async for _ in execute_action(
        FireLLMFallback(user_utterance="привет"),
        ws=None, session=None, backend=backend, tts=tts, calc=None,
        rag_future=None,
    ):
        pass

    assert backend.call_count == 1
    assert tts.chunks  # non-empty


@pytest.mark.asyncio
async def test_fire_llm_fallback_swallows_rag_future_exception() -> None:
    """Invariant #1 fail-open: if `rag_future.result()` raises, the
    handler proceeds with empty RAG context instead of bubbling the
    exception. Speculative RAG is best-effort."""
    from backend.turn_dispatcher import execute_action

    class ExplodingRAG:
        async def result(self) -> str:
            raise RuntimeError("RAG upstream 500")

    backend = StreamingLLMBackend(["Здравствуйте. "], per_token_sleep=0.0)
    tts = FakeTts()

    async for _ in execute_action(
        FireLLMFallback(user_utterance="привет"),
        ws=None, session=None, backend=backend, tts=tts, calc=None,
        rag_future=ExplodingRAG(),
    ):
        pass

    assert backend.call_count == 1  # LLM still fired on empty context


@pytest.mark.asyncio
async def test_fire_llm_fallback_injects_snapshot_as_anchor() -> None:
    """E7 anti-hallucination anchor: when snapshot has captured fields,
    the user content must carry them so the LLM does not re-ask."""
    from backend.turn_dispatcher import execute_action

    snap = _complete_snapshot_usd_to_byn()
    backend = StreamingLLMBackend(["ok"], per_token_sleep=0.0)
    tts = FakeTts()

    async for _ in execute_action(
        FireLLMFallback(user_utterance="а что по документам?", snapshot=snap),
        ws=None, session=None, backend=backend, tts=tts, calc=None,
        rag_future=None,
    ):
        pass

    msgs = (backend.last_kwargs or {}).get("messages", [])
    joined = " ".join(m.get("content", "") for m in msgs if isinstance(m, dict))
    # Snapshot values (captured subject, cost, currency) appear in prompt.
    assert "Легковой автомобиль" in joined
    assert "240000" in joined or "240000.0" in joined


@pytest.mark.asyncio
async def test_fire_llm_fallback_respects_session_interrupted() -> None:
    """Invariant #5: barge-in aborts mid-stream at the next phrase
    boundary. After the session flips `interrupted=True`, no further
    sentences should reach TTS."""
    from backend.turn_dispatcher import execute_action

    # 3 full sentences in the stream.
    tokens = [
        "Один. ",
        "Два. ",
        "Три. ",
    ]
    backend = StreamingLLMBackend(tokens, per_token_sleep=0.0)
    tts = FakeTts()
    session = InterruptingSession(flip_after_n_sentences=1)

    # Wrap tts.say to signal the session after each sentence lands.
    orig_say = tts.say

    async def _observing_say(text: str) -> None:
        await orig_say(text)
        session.observe_sentence()

    tts.say = _observing_say  # type: ignore[method-assign]

    async for _ in execute_action(
        FireLLMFallback(user_utterance="расскажи"),
        ws=None, session=session, backend=backend, tts=tts, calc=None,
        rag_future=None,
    ):
        pass

    # First sentence emitted; subsequent sentences suppressed.
    assert tts.chunks == ["Один."]


# ---------------------------------------------------------------- Task 20
# Calc-failure circuit breaker + tool-call history append.
# Matches legacy shape: session.consecutive_calc_failures counts
# same-signature failures (different signature resets to 1). On third
# consecutive failure the FireCalc handler routes to the OOR message
# instead of re-raising.


class FakeVoiceSession:
    """Minimal voice_session stand-in exposing the fields the FireCalc
    handler reads / writes for circuit breaker + tool-call history."""
    def __init__(self) -> None:
        self.interrupted = False
        self.tool_calls_this_turn: list[dict] = []
        self.last_calc_signature: str = ""
        self.consecutive_calc_failures: int = 0


_CALC_PARAMS = {"cost": 240000, "currency": "BYN", "prepaid": 30}
_CALC_RESULT_OK = {
    "params": _CALC_PARAMS,
    "advance_sum": 72000,
    "payment_min": 8109,
    "buyout_sum": 10000,
    "total": 342317,
    "num_payments": 36,
    "increase_percent": 12.5,
}


@pytest.mark.asyncio
async def test_fire_calc_appends_tool_call_on_success() -> None:
    """Spec §7.2 #3: FireCalc appends the invocation to
    `session.tool_calls_this_turn` so SMS replay + prior-result reuse
    still work after the refactor."""
    from backend.turn_dispatcher import execute_action

    session = FakeVoiceSession()
    tts = FakeTts()
    calc = FakeCalc(result=dict(_CALC_RESULT_OK))
    fire = FireCalc(
        snapshot=_complete_snapshot_usd_to_byn(),
        calc_params=dict(_CALC_PARAMS),
    )

    async for _ in execute_action(
        fire, ws=None, session=session, backend=FakeLLMBackend(),
        tts=tts, calc=calc, rag_future=None,
    ):
        pass

    assert len(session.tool_calls_this_turn) == 1
    entry = session.tool_calls_this_turn[0]
    assert entry["tool"] == "calculator"
    assert entry["params"] == _CALC_PARAMS
    assert entry["result"]["advance_sum"] == 72000


@pytest.mark.asyncio
async def test_fire_calc_resets_failure_counter_on_success() -> None:
    """Spec §7.2 #4: a successful calc zeroes the consecutive-failure
    counter, matching legacy shape at app.py:2642-2648."""
    from backend.turn_dispatcher import execute_action

    session = FakeVoiceSession()
    session.consecutive_calc_failures = 2  # prior failures
    session.last_calc_signature = "stale"
    calc = FakeCalc(result=dict(_CALC_RESULT_OK))
    fire = FireCalc(
        snapshot=_complete_snapshot_usd_to_byn(),
        calc_params=dict(_CALC_PARAMS),
    )

    async for _ in execute_action(
        fire, ws=None, session=session, backend=FakeLLMBackend(),
        tts=FakeTts(), calc=calc, rag_future=None,
    ):
        pass

    assert session.consecutive_calc_failures == 0
    # New signature is recorded on success.
    assert session.last_calc_signature == str(sorted(_CALC_PARAMS.items()))


@pytest.mark.asyncio
async def test_fire_calc_increments_failure_counter_on_same_sig_exception() -> None:
    """Same-signature failure → counter += 1. First failure propagates
    so the orchestrator can log / route, matching current behaviour."""
    from backend.turn_dispatcher import execute_action

    session = FakeVoiceSession()
    session.last_calc_signature = str(sorted(_CALC_PARAMS.items()))
    session.consecutive_calc_failures = 1  # one prior same-sig fail
    calc = FakeCalc(raises=RuntimeError("calc 500"))
    fire = FireCalc(
        snapshot=_complete_snapshot_usd_to_byn(),
        calc_params=dict(_CALC_PARAMS),
    )

    with pytest.raises(RuntimeError, match="calc 500"):
        async for _ in execute_action(
            fire, ws=None, session=session, backend=FakeLLMBackend(),
            tts=FakeTts(), calc=calc, rag_future=None,
        ):
            pass

    assert session.consecutive_calc_failures == 2
    assert session.last_calc_signature == str(sorted(_CALC_PARAMS.items()))


@pytest.mark.asyncio
async def test_fire_calc_resets_counter_to_one_on_different_sig_exception() -> None:
    """Different-signature failure → counter resets to 1 (legacy shape
    at app.py:2646-2648). Prior signature is replaced."""
    from backend.turn_dispatcher import execute_action

    session = FakeVoiceSession()
    session.last_calc_signature = "other"
    session.consecutive_calc_failures = 2
    calc = FakeCalc(raises=RuntimeError("calc 500"))
    fire = FireCalc(
        snapshot=_complete_snapshot_usd_to_byn(),
        calc_params=dict(_CALC_PARAMS),
    )

    with pytest.raises(RuntimeError):
        async for _ in execute_action(
            fire, ws=None, session=session, backend=FakeLLMBackend(),
            tts=FakeTts(), calc=calc, rag_future=None,
        ):
            pass

    assert session.consecutive_calc_failures == 1
    assert session.last_calc_signature == str(sorted(_CALC_PARAMS.items()))


@pytest.mark.asyncio
async def test_fire_calc_routes_to_oor_on_third_consecutive_failure() -> None:
    """Spec §7.2 #4: on the third consecutive same-signature failure
    the handler yields the deterministic OOR text instead of re-raising
    so the user hears something sane. LLM is NOT invoked."""
    from backend.turn_dispatcher import execute_action

    session = FakeVoiceSession()
    session.last_calc_signature = str(sorted(_CALC_PARAMS.items()))
    session.consecutive_calc_failures = 2  # two prior same-sig fails
    calc = FakeCalc(raises=RuntimeError("calc 500"))
    llm = FakeLLMBackend()
    tts = FakeTts()
    fire = FireCalc(
        snapshot=_complete_snapshot_usd_to_byn(),
        calc_params=dict(_CALC_PARAMS),
    )

    # Third failure MUST NOT raise — OOR routing instead.
    chunks: list[str] = []
    async for chunk in execute_action(
        fire, ws=None, session=session, backend=llm,
        tts=tts, calc=calc, rag_future=None,
    ):
        chunks.append(chunk)

    assert session.consecutive_calc_failures == 3
    # OOR text shipped to TTS.
    spoken = tts.collected_text().lower()
    assert "перепровер" in spoken or "не могу посчитать" in spoken
    assert chunks  # yielded at least the OOR sentence
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_fire_calc_tolerates_session_none() -> None:
    """When no session is provided (unit-test shortcut) the handler
    still runs calc + renders + propagates exceptions, but skips the
    circuit-breaker bookkeeping."""
    from backend.turn_dispatcher import execute_action

    calc = FakeCalc(raises=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in execute_action(
            FireCalc(
                snapshot=_complete_snapshot_usd_to_byn(),
                calc_params=dict(_CALC_PARAMS),
            ),
            ws=None, session=None, backend=FakeLLMBackend(),
            tts=FakeTts(), calc=calc, rag_future=None,
        ):
            pass


# ---------------------------------------------------------------- Bug 1: ok=False + USD
# 2026-04-24 live regression ac0e35d6: FireCalc handler rendered "? USD"
# placeholders on calc API failure. Also the USD disclosure prefix wasn't
# attached on success when the snapshot carried original_* stash.


@pytest.mark.asyncio
async def test_fire_calc_on_ok_false_speaks_error_not_question_marks() -> None:
    """Calc returned ok=False (API rejected params). Handler MUST speak
    the API's error text instead of running `render_calc_result` — the
    renderer prints '?' for every missing numeric field when the result
    dict is empty."""
    from backend.turn_dispatcher import execute_action

    calc = FakeCalc(result={
        "ok": False,
        "error": "Стоимость должна быть от 10 000 до 10 000 000 рублей.",
        "params": dict(_CALC_PARAMS),
    })
    tts = FakeTts()
    async for _ in execute_action(
        FireCalc(
            snapshot=_complete_snapshot_usd_to_byn(),
            calc_params=dict(_CALC_PARAMS),
        ),
        ws=None, session=FakeVoiceSession(), backend=FakeLLMBackend(),
        tts=tts, calc=calc, rag_future=None,
    ):
        pass

    spoken = tts.collected_text()
    # NOT the placeholder pattern "? USD" / "? BYN" from render_calc_result.
    assert "? USD" not in spoken
    assert "? BYN" not in spoken
    # The API error text reached the user verbatim.
    assert "от 10 000 до 10 000 000" in spoken


@pytest.mark.asyncio
async def test_fire_calc_on_ok_false_empty_dict_speaks_generic_fallback() -> None:
    """Calc returned ok=False with no 'error' key — handler falls back
    to a generic Russian retry phrase, not '?' placeholders."""
    from backend.turn_dispatcher import execute_action

    calc = FakeCalc(result={"ok": False})
    tts = FakeTts()
    async for _ in execute_action(
        FireCalc(
            snapshot=_complete_snapshot_usd_to_byn(),
            calc_params=dict(_CALC_PARAMS),
        ),
        ws=None, session=FakeVoiceSession(), backend=FakeLLMBackend(),
        tts=tts, calc=calc, rag_future=None,
    ):
        pass

    spoken = tts.collected_text()
    assert "? USD" not in spoken
    assert "? BYN" not in spoken
    assert "рассч" in spoken.lower() or "уточн" in spoken.lower()


@pytest.mark.asyncio
async def test_fire_calc_attaches_currency_conversion_from_snapshot() -> None:
    """When snapshot.original_currency == 'USD', the handler must
    populate result['currency_conversion'] so render_calc_result
    prefixes the spoken narration with 'Стоимость N долларов (это M
    белорусских рублей по курсу X к 1)'. Live ac0e35d6 lost this
    prefix because the handler never attached the disclosure."""
    from backend.turn_dispatcher import execute_action

    calc_result = {
        "ok": True,
        "params": {"cost": 240000, "currency": "BYN", "prepaid": 30},
        "advance_sum": 72000, "payment_min": 8109,
        "buyout_sum": 10000, "total": 342317,
        "num_payments": 36, "increase_percent": 12.5,
    }
    calc = FakeCalc(result=dict(calc_result))
    tts = FakeTts()
    async for _ in execute_action(
        FireCalc(
            snapshot=_complete_snapshot_usd_to_byn(),
            calc_params={"cost": 240000, "currency": "BYN"},
        ),
        ws=None, session=FakeVoiceSession(), backend=FakeLLMBackend(),
        tts=tts, calc=calc, rag_future=None,
    ):
        pass

    spoken = tts.collected_text()
    assert "Стоимость 80000 долларов" in spoken
    assert "240000 белорусских рублей" in spoken
    assert "по курсу 3 к 1" in spoken
