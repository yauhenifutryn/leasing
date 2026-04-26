"""Scripted-turn harness for the apply_turn / execute_action dispatch.

Drives scripted turn sequences through the apply_turn / execute_action
pipeline with every external I/O stubbed (LLM tokens fixed, TTS
captured, calc result fixed, RAG fragment fixed). Asserts the
externally-observable sequence (TTS text + tool events) per turn.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from backend.session import ClientProfile


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
    """Returns a deterministic calc result derived from `params`. Records
    each invocation. The result fields scale linearly off `params["cost"]`
    so multi-cost scenarios (e.g. cost-change replays) produce the right
    numbers in the rendered TTS without needing per-test fixture values.
    """

    def __init__(self):
        self.invocations: list[dict[str, Any]] = []

    async def calculate(self, params: dict[str, Any]) -> dict[str, Any]:
        self.invocations.append(dict(params))
        cost = float(params.get("cost") or 30000)
        # Shape mirrors `tools/calculator.py`'s execute() return: keys
        # consumed by `render_calc_result` (advance_sum, payment_min,
        # buyout_sum, total, increase_percent, num_payments, params).
        # Synthetic deterministic numbers based on cost so the renderer
        # speaks figures matching the staged params.
        return {
            "ok": True,
            "params": dict(params),
            "advance_sum": round(cost * 0.2),
            "payment_min": round(cost * 0.045),
            "payment_max": round(cost * 0.045),
            "buyout_sum": 300,
            "total": round(cost * 1.3),
            "increase_percent": 14,
            "num_payments": params.get("term", 24),
        }


class _FixedRag:
    """Resolves to a fixed fragment string."""

    def __init__(self, fragment: str = "[Fragment 1]\nИнформация о лизинге."):
        self._fragment = fragment

    async def result(self) -> str:
        return self._fragment


async def _run_scenario(
    scenario: Scenario,
    *,
    apply_turn_enabled: bool = True,
    initial_profile: Optional["ClientProfile"] = None,
) -> dict:
    """Drive the scenario through the apply_turn path with all I/O mocked.

    Returns a dict {"tts": [[...turn0_tts...], ...], "tool_events":
    [[...turn0_events...], ...]}.

    `initial_profile` lets a test seed the profile in a non-COLLECTING
    state (e.g. CONFIRMED) so scenarios that exercise post-confirmation
    behavior don't have to replay the full clarify funnel each turn.
    """
    if not apply_turn_enabled:
        pytest.skip("Legacy-path harness wiring is deferred to Task 10.")

    from backend.classifier_schema import parse_classifier_output
    from backend.session import ClientProfile
    from backend.turn_dispatcher import apply_turn, execute_action

    backend = _FixedLLMBackend()
    calc = _FixedCalc()
    rag_future = _FixedRag()

    profile = initial_profile if initial_profile is not None else ClientProfile()
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


# ====================================================================
# Live-call regression scenarios (Task 10).
#
# Each scenario replays the relevant turns from a live SIP call and
# asserts the Section-3 fix (Tasks 5/7/8) prevents the original
# regression from recurring.
# ====================================================================


def _confirmed_phys_profile():
    """Profile in CONFIRMED state with all calc-required fields set.

    Used by scenarios that exercise post-confirmation behavior so they
    don't have to replay the full clarify funnel each turn. Mirrors the
    state the live caller would be in after the readback "Да".
    """
    from backend.session import ClientProfile, ProfileState

    return ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=30000.0,
        currency="BYN",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        type_schedule="0",
        state=ProfileState.CONFIRMED,
    )


def _almost_complete_rub_phys_profile():
    """Phys profile with currency=RUB and only `type_schedule` missing.

    Replays the live state at f7e5aa1d turn ~9: caller already supplied
    cost, prepaid, term, condition, client_type, subject, AND switched
    currency to RUB on a prior turn (mid-funnel). Profile is COLLECTING.
    The next turn that fills `type_schedule` should hit step 5a's
    preflight (Task 5) and emit FireOORMessage rejecting RUB — NOT a
    readback that quotes "10000 RUB" as confirmed parameters.
    """
    from backend.session import ClientProfile, ProfileState

    return ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=10000.0,
        currency="RUB",
        condition_new=1,
        term_months=24,
        prepaid_pct=20.0,
        state=ProfileState.COLLECTING,
    )


# ---- Scenario 1: RUB rejected before readback (Task 5 fix) ----

RUB_REJECT_FOR_PHYS = Scenario(
    name="rub_reject_for_phys",
    turns=[
        ScriptedTurn(
            utterance="Аннуитетный",
            classifier_output={
                "intent": "TOOL",
                "type_schedule": "0",
                "action": "calculate",
            },
            # Task 5 fix: preflight runs BEFORE EmitReadback, so RUB
            # never reaches the caller as "confirmed" parameters.
            expect_tts_contains=["RUB", "не поддерживается"],
            expect_tts_absent=["10000 RUB", "стоимость 10000"],
        ),
    ],
)


@pytest.mark.asyncio
async def test_parity_rub_reject_for_phys():
    profile = _almost_complete_rub_phys_profile()
    out = await _run_scenario(
        RUB_REJECT_FOR_PHYS,
        apply_turn_enabled=True,
        initial_profile=profile,
    )
    last_tts = " ".join(out["tts"][-1])
    # Task 5 fix: OOR message fires (preflight catches RUB) instead of
    # speaking "стоимость 10000 RUB" in a readback.
    assert "RUB" in last_tts, f"expected RUB in OOR message; got: {last_tts!r}"
    assert "не поддерживается" in last_tts, (
        f"expected unsupported-currency phrasing; got: {last_tts!r}"
    )
    # The bug: a readback that confirms RUB as a valid param.
    assert "10000 RUB" not in last_tts
    assert "Параметры расчёта" not in last_tts, (
        f"readback fired for an unsupported currency: {last_tts!r}"
    )


# ---- Scenario 2: Numeric-word cost change grounds (Task 7 fix) ----

NUMERIC_WORDS_COST_CHANGE = Scenario(
    name="numeric_words_cost_change",
    turns=[
        ScriptedTurn(
            utterance="Оставим двадцать тысяч долларов",
            classifier_output={
                "intent": "TOOL",
                "cost": 20000.0,
                "currency": "USD",
                "change_field": "cost",
                "change_value": 20000,
                "action": "calculate",
                "is_confirmation": False,
            },
            # Task 7 fix: parse_ru_number grounds cost=20000 from the
            # word form. Without it, has_field_signal would reject
            # cost=20000 (no "20000" digit substring in utterance) and
            # the change-confirm would not name the new cost.
            expect_tts_contains=["Меняю", "20000"],
        ),
        ScriptedTurn(
            utterance="Да",
            classifier_output={
                "intent": "CONVERSATION",
                "is_confirmation": True,
            },
            expect_tool_events=["calculator"],
            # 20000 USD * rate 3 = 60000 BYN (preflight conversion).
            expect_tts_contains=["60000"],
        ),
    ],
)


@pytest.mark.asyncio
async def test_parity_numeric_words_cost_change():
    profile = _confirmed_phys_profile()
    out = await _run_scenario(
        NUMERIC_WORDS_COST_CHANGE,
        apply_turn_enabled=True,
        initial_profile=profile,
    )
    # Turn 0: change-confirm names the new cost.
    turn0_tts = " ".join(out["tts"][0])
    assert "20000" in turn0_tts, (
        f"expected '20000' in change-confirm TTS; got: {turn0_tts!r}"
    )
    # Turn 1: calc fires with the new cost; renderer speaks BYN figures.
    assert out["tool_events"][1] == ["calculator"], (
        f"expected calculator event on turn 1; got {out['tool_events'][1]!r}"
    )
    turn1_tts = " ".join(out["tts"][1])
    # Conv prefix: "Стоимость 20000 долларов (это 60000 белорусских
    # рублей по курсу 3 к 1)." plus the calculator-derived BYN totals.
    assert "60000" in turn1_tts, (
        f"expected BYN-converted cost in calc TTS; got: {turn1_tts!r}"
    )


# ---- Scenario 3: Mixed client_type+subject clarify (Task 8 fix) ----

MIXED_CLIENT_TYPE_SUBJECT_CLARIFY = Scenario(
    name="mixed_client_type_subject_clarify",
    turns=[
        ScriptedTurn(
            utterance=(
                "Хорошо, ладно. А можно взять для юрлица "
                "коммерческие автомобили?"
            ),
            classifier_output={
                "intent": "CONVERSATION",
                "client_type": "Юридическое лицо",
                "action": "clarify",
            },
            # Task 8 fix: mixed delta (client_type) + commercial gesture
            # ("коммерч") + classifier subject is None → ask for subject
            # instead of silently staging only the client_type half.
            expect_tts_contains=["легков", "грузов"],
        ),
    ],
)


@pytest.mark.asyncio
async def test_parity_mixed_client_type_subject_clarify():
    profile = _confirmed_phys_profile()
    out = await _run_scenario(
        MIXED_CLIENT_TYPE_SUBJECT_CLARIFY,
        apply_turn_enabled=True,
        initial_profile=profile,
    )
    turn0 = " ".join(out["tts"][0]).lower()
    # Subject-clarify renders the subject-category vocabulary.
    assert "легков" in turn0 and "грузов" in turn0, (
        f"expected subject-clarify (легков/грузов); got: {turn0!r}"
    )
    # The bug: a single-field client_type change-confirm leaks through
    # ("Меняю тип клиента на юридическое лицо ...") even though the
    # subject is unresolved. Assert that didn't happen.
    assert "меняю" not in turn0, (
        f"client_type change-confirm leaked into TTS: {turn0!r}"
    )
