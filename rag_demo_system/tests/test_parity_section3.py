"""Parity harness for APPLY_TURN_ENABLED=1 path.

Drives scripted turn sequences through the apply_turn / execute_action
pipeline with every external I/O stubbed (LLM tokens fixed, TTS
captured, calc result fixed, RAG fragment fixed). Asserts the
externally-observable sequence (TTS text + tool events) per turn.

The legacy path side of the parity check is added in Task 10 once a
callable wrapper around app.py's 5-gate block is extracted. For now
this harness ensures the apply_turn path produces consistent,
deterministic behavior for the MVP happy-path scenario.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class ScriptedTurn:
    utterance: str
    classifier_output: dict[str, Any]
    expect_tts_contains: list[str] = field(default_factory=list)
    expect_tts_absent: list[str] = field(default_factory=list)
    expect_tool_events: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    turns: list[ScriptedTurn]


class _FixedLLMBackend:
    """Yields a fixed two-sentence reply — used when apply_turn fires
    FireLLMFallback. For non-fallback turns this backend is unused.
    """

    def __init__(self, reply_tokens=None):
        self._tokens = reply_tokens or ["Понятно. ", "Расскажу подробнее."]

    async def stream(self, messages):
        for tok in self._tokens:
            yield tok


class _CapturingTts:
    """Captures every cleaned sentence sent to TTS for the current turn."""

    def __init__(self):
        self.calls: list[str] = []

    async def say(self, text: str) -> None:
        self.calls.append(text)


class _FixedCalc:
    """Returns a deterministic calc result. Records each invocation."""

    def __init__(self):
        self.invocations: list[dict[str, Any]] = []
        # Shape mirrors `tools/calculator.py`'s execute() return: keys
        # consumed by `render_calc_result` (advance_sum, payment_min,
        # buyout_sum, total, increase_percent, num_payments, params).
        self.result = {
            "ok": True,
            "params": {
                "cost": 30000,
                "currency": "BYN",
                "prepaid": 20,
                "term": 24,
                "type_schedule": "0",
                "client_type": "Физическое лицо",
                "subject": "Легковой автомобиль",
                "condition_new": 1,
            },
            "advance_sum": 6000,
            "payment_min": 1345,
            "payment_max": 1345,
            "buyout_sum": 300,
            "total": 38583,
            "increase_percent": 14,
            "num_payments": 24,
        }

    async def calculate(self, params: dict[str, Any]) -> dict[str, Any]:
        self.invocations.append(dict(params))
        return self.result


class _FixedRag:
    """Resolves to a fixed fragment string."""

    def __init__(self, fragment: str = "[Fragment 1]\nИнформация о лизинге."):
        self._fragment = fragment

    async def result(self) -> str:
        return self._fragment


async def _run_scenario(
    scenario: Scenario, *, apply_turn_enabled: bool = True
) -> dict:
    """Drive the scenario through the apply_turn path with all I/O mocked.

    Returns a dict {"tts": [[...turn0_tts...], ...], "tool_events":
    [[...turn0_events...], ...]}.
    """
    if not apply_turn_enabled:
        pytest.skip("Legacy-path harness wiring is deferred to Task 10.")

    from backend.classifier_schema import parse_classifier_output
    from backend.session import ClientProfile
    from backend.turn_dispatcher import apply_turn, execute_action

    backend = _FixedLLMBackend()
    calc = _FixedCalc()
    rag_future = _FixedRag()

    profile = ClientProfile()
    session = MagicMock()
    session.interrupted = False
    session.assistant_speaking = False
    # Realistic per-turn collectors so execute_action's history /
    # circuit-breaker side-paths don't trip on MagicMock auto-attrs.
    session.tool_calls_this_turn = []
    session.consecutive_calc_failures = 0
    session.last_calc_signature = ""

    captured_tts: list[list[str]] = []
    captured_tool_events: list[list[str]] = []

    for turn_idx, turn in enumerate(scenario.turns):
        tts = _CapturingTts()
        prior_calc_count = len(calc.invocations)

        co = parse_classifier_output(
            json.dumps(turn.classifier_output),
            utterance=turn.utterance,
        )
        action = apply_turn(profile, co, turn.utterance, turn_id=turn_idx + 1)

        async for _chunk in execute_action(
            action,
            ws=MagicMock(),
            session=session,
            backend=backend,
            tts=tts,
            calc=calc,
            rag_future=rag_future,
        ):
            pass

        events: list[str] = []
        if len(calc.invocations) > prior_calc_count:
            events.append("calculator")

        captured_tts.append(list(tts.calls))
        captured_tool_events.append(events)

    return {"tts": captured_tts, "tool_events": captured_tool_events}


MVP_HAPPY_PATH = Scenario(
    name="mvp_happy_path",
    turns=[
        ScriptedTurn(
            utterance=(
                "Меня зовут Борис, физическое лицо, новый автомобиль за 10000 "
                "долларов, 24 месяца, аванс 20 процентов, аннуитет"
            ),
            classifier_output={
                "intent": "TOOL",
                "name": "Борис",
                "client_type": "Физическое лицо",
                "subject": "Легковой автомобиль",
                "cost": 10000.0,
                "currency": "USD",
                "condition_new": 1,
                "term_months": 24,
                "prepaid_pct": 20.0,
                "type_schedule": "0",
                "action": "calculate",
            },
            expect_tts_contains=["параметр"],  # readback intro
        ),
        ScriptedTurn(
            utterance="Да",
            classifier_output={"intent": "CONVERSATION", "is_confirmation": True},
            expect_tool_events=["calculator"],
            expect_tts_contains=["BYN"],  # USD→BYN converted body uses BYN currency code
        ),
    ],
)


@pytest.mark.asyncio
async def test_parity_mvp_happy_path():
    out = await _run_scenario(MVP_HAPPY_PATH, apply_turn_enabled=True)
    for i, turn in enumerate(MVP_HAPPY_PATH.turns):
        tts_joined = " ".join(out["tts"][i])
        for expected in turn.expect_tts_contains:
            assert expected.lower() in tts_joined.lower(), (
                f"turn {i}: missing '{expected}' in TTS; got: {tts_joined!r}"
            )
        for forbidden in turn.expect_tts_absent:
            assert forbidden.lower() not in tts_joined.lower(), (
                f"turn {i}: unexpected '{forbidden}' in TTS; got: {tts_joined!r}"
            )
        if turn.expect_tool_events:
            assert out["tool_events"][i] == turn.expect_tool_events, (
                f"turn {i}: tool events mismatch; got {out['tool_events'][i]!r}"
            )
