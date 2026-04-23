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
    FireCalc,
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
