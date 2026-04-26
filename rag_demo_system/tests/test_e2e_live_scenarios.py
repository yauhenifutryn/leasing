"""End-to-end replay tests for live SIP-call regressions.

Each test replays the relevant turns from a single live call through
the parity harness and asserts the Section-3 fix (Tasks 5/7/8) prevents
the original regression. Stricter assertions than the parity tests so a
silent renderer change won't slip past.

Live call ID -> fixed by:
    f7e5aa1d turn ~9  -> Task 5 (preflight before EmitReadback)
    f7e5aa1d turn ~7  -> Task 7 (numeric-word cost grounding)
    f7e5aa1d turn 11  -> Task 8 (mixed client_type+subject clarify)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_parity_section3 import (  # noqa: E402  (post-sys.path)
    MIXED_CLIENT_TYPE_SUBJECT_CLARIFY,
    NUMERIC_WORDS_COST_CHANGE,
    RUB_REJECT_FOR_PHYS,
    _almost_complete_rub_phys_profile,
    _confirmed_phys_profile,
    _run_scenario,
)


@pytest.mark.asyncio
async def test_e2e_f7e5aa1d_rub_rejected_before_readback():
    """Live regression f7e5aa1d turn ~9: profile complete + currency=RUB
    + Физ лицо produced "Параметры расчёта: ... стоимость 10000 RUB ..."
    in a readback that asked the caller to confirm. Task 5 moved the
    preflight policy check ahead of EmitReadback so unsupported
    currencies route to FireOORMessage instead.
    """
    profile = _almost_complete_rub_phys_profile()
    out = await _run_scenario(
        RUB_REJECT_FOR_PHYS,
        apply_turn_enabled=True,
        initial_profile=profile,
    )
    last_tts = " ".join(out["tts"][-1])
    # OOR message fires.
    assert "RUB" in last_tts
    assert "не поддерживается" in last_tts
    # The bug: a readback that quoted the unsupported currency.
    assert "10000 RUB" not in last_tts
    assert "Параметры расчёта" not in last_tts
    # Calc must NOT have run on an unsupported-currency profile.
    assert out["tool_events"][-1] == [], (
        f"calc fired on RUB+phys; got events: {out['tool_events']!r}"
    )


@pytest.mark.asyncio
async def test_e2e_f7e5aa1d_numeric_words_cost_grounded():
    """Live regression f7e5aa1d turn ~7: caller said "оставим двадцать
    тысяч долларов" — classifier emitted cost=20000 + change_field=cost,
    but `value_grounded("cost", 20000, utterance)` returned False because
    the digit "20000" wasn't in the utterance. Task 7 added the
    `parse_ru_number` fallback so word-form costs ground correctly.
    """
    profile = _confirmed_phys_profile()
    out = await _run_scenario(
        NUMERIC_WORDS_COST_CHANGE,
        apply_turn_enabled=True,
        initial_profile=profile,
    )
    # Turn 0: change-confirm names the new cost (20000) in the spoken
    # text. Without Task 7, the change is silently dropped — TTS would
    # not contain "20000".
    turn0 = " ".join(out["tts"][0])
    assert "Меняю" in turn0
    assert "20000" in turn0
    # Turn 1: confirmation -> calc fires with the new cost.
    assert out["tool_events"][1] == ["calculator"]
    turn1 = " ".join(out["tts"][1])
    # 20000 USD * 3 = 60000 BYN; conv-prefix discloses both figures.
    assert "20000" in turn1
    assert "60000" in turn1
    assert "белорусских рублей" in turn1


@pytest.mark.asyncio
async def test_e2e_f7e5aa1d_mixed_clarify_not_silent_stage():
    """Live regression f7e5aa1d turn 11: caller said "для юрлица
    коммерческие автомобили" — classifier emitted client_type=Юр and
    subject=Грузовой, but subject grounding dropped "коммерческие"
    (no overlap with the cue regex). Step 4 silently staged ONLY the
    client_type half and a re-calc later ran with Легковой+Юр (wrong
    subject). Task 8 added a mixed-clarify branch that asks for the
    subject when this pattern appears.
    """
    profile = _confirmed_phys_profile()
    out = await _run_scenario(
        MIXED_CLIENT_TYPE_SUBJECT_CLARIFY,
        apply_turn_enabled=True,
        initial_profile=profile,
    )
    turn0 = " ".join(out["tts"][0]).lower()
    # Subject-clarify renders the subject-category vocabulary.
    assert "легков" in turn0
    assert "грузов" in turn0
    # The bug: a single-field client_type change-confirm leaks through.
    assert "меняю" not in turn0
    # Calc must NOT have fired on this turn — we're asking for clarity.
    assert out["tool_events"][0] == []
