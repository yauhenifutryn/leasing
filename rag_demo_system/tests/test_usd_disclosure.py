"""Fix 1.2 — dual-currency disclosure for Физлицо + USD.

When a Физическое лицо client names the cost in USD, the direct-call path
converts to BYN using the MVP 3:1 rate before calling the calculator. Before
Fix 1.2, every downstream render (readback, SMS, post-calc voice summary)
showed only the converted BYN figure — the client heard "стоимость 60000
рублей" without any acknowledgement of the 20000 USD they originally quoted,
and complained they could not tell what the bot was referencing.

This test suite locks in three behaviours:
  1. `build_readback_text` shows BOTH amounts when the profile carries
     original_currency="USD".
  2. `build_readback_text` renders the legacy single-currency form when
     original_currency is unset (BYN-only path).
  3. `CalculatorTool.format_sms_body` discloses both amounts when the tool
     result carries `currency_conversion` metadata emitted by the DirectTool
     USD->BYN path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import build_readback_text
from backend.session import ClientProfile
from backend.tools.calculator import CalculatorTool


def _profile(**overrides):
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=60000.0,
        currency="BYN",
        condition_new=1,
        term_months=36,
        type_schedule="0",
        prepaid_pct=30.0,
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def test_usd_readback_includes_both_amounts() -> None:
    p = _profile(original_cost=20000.0, original_currency="USD")
    txt = build_readback_text(p)
    # Both amounts must be visible so TTS reads "двадцать тысяч долларов …
    # шестьдесят тысяч белорусских рублей" rather than only the BYN figure.
    assert "20000" in txt, f"original USD amount missing from readback: {txt}"
    assert "долларов" in txt, f"word 'долларов' missing from readback: {txt}"
    assert "60000" in txt, f"converted BYN amount missing: {txt}"
    assert "рубл" in txt, f"BYN word missing: {txt}"
    # The rate must be disclosed so the client can audit the arithmetic.
    assert "3" in txt and ("курс" in txt.lower() or "к 1" in txt or "1:3" in txt), (
        f"conversion rate not disclosed: {txt}"
    )


def test_byn_only_readback_unchanged() -> None:
    # Legacy path: no original_* => single-currency render.
    p = _profile()  # original_* stay None
    txt = build_readback_text(p)
    assert "долларов" not in txt
    # The BYN amount appears exactly once (no dual disclosure).
    assert txt.count("60000") == 1


def test_usd_readback_pre_conversion_discloses_both() -> None:
    """Fix 1.6 — client quoted USD, DirectTool hasn't fired yet. Readback
    must still speak both amounts so the caller knows BYN will be used
    downstream. Observed 2026-04-19 live call: bare "120000 USD" in
    readback confused the caller who did not realise conversion was
    coming."""
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=120000.0,         # still USD here, DirectTool not fired
        currency="USD",
        condition_new=1,
        term_months=36,
        type_schedule="0",
        prepaid_pct=30.0,
        # original_cost / original_currency intentionally None at this stage
    )
    txt = build_readback_text(p)
    assert "120000" in txt
    assert "долларов" in txt
    # 120000 * 3 = 360000 is the MVP-rate BYN equivalent.
    assert "360000" in txt, f"BYN equivalent missing: {txt}"
    assert "рубл" in txt
    assert "3 к 1" in txt or "курсу 3" in txt


def test_fractional_rate_readback_math_matches_narration(monkeypatch) -> None:
    """Fix 1.9 — Codex adversarial review found that a fractional
    USD_BYN_RATE (e.g. 3.25) caused the pre-conversion readback to
    compute BYN with an int-rounded rate (x3) while narrating "по курсу 3
    к 1" — producing 360000 for a 120000 USD quote while the actual
    calculator conversion produced 390000. Readback must either show the
    precise math or say the precise rate; silent drift is a financial
    UX bug.
    """
    # Bypass the settings cache so we can inject a fractional rate.
    import backend.profile_prompts as pp
    monkeypatch.setattr(pp, "_USD_BYN_RATE_CACHE", 3.25, raising=False)

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=120000.0,
        currency="USD",
        condition_new=1,
        term_months=36,
        type_schedule="0",
        prepaid_pct=30.0,
    )
    txt = build_readback_text(p)

    # 120000 * 3.25 = 390000. Int-rounded rate path produces 360000 — bug.
    assert "390000" in txt, f"pre-conversion BYN amount must use the exact rate: {txt}"
    assert "360000" not in txt, f"int-rounded rate silently diverged: {txt}"
    # Narration must expose the real rate, not a rounded "3 к 1".
    assert "3.25 к 1" in txt, f"fractional rate must be narrated: {txt}"


def test_integer_rate_readback_still_clean(monkeypatch) -> None:
    """Whole rate 3.0 must render as '3 к 1', no trailing zeros."""
    import backend.profile_prompts as pp
    monkeypatch.setattr(pp, "_USD_BYN_RATE_CACHE", 3.0, raising=False)

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=20000.0,
        currency="USD",
        condition_new=1,
        term_months=36,
        type_schedule="0",
        prepaid_pct=30.0,
    )
    txt = build_readback_text(p)
    assert "60000" in txt
    assert "3 к 1" in txt and "3.0 к 1" not in txt


def test_usd_readback_legal_entity_stays_usd() -> None:
    """Юрлицо can settle in USD directly — no conversion happens, no
    dual-disclosure should appear."""
    p = ClientProfile(
        client_type="Юридическое лицо",
        subject="Легковой автомобиль",
        cost=120000.0,
        currency="USD",
        condition_new=1,
        term_months=36,
        type_schedule="0",
        prepaid_pct=30.0,
    )
    txt = build_readback_text(p)
    # Should say USD, should NOT invent a BYN conversion the caller isn't
    # going to see on the invoice.
    assert "USD" in txt or "долларов" in txt
    assert "рубл" not in txt, f"юрлицо readback leaked BYN disclosure: {txt}"


def test_sms_body_usd_disclosure() -> None:
    result = {
        "ok": True,
        "params": {
            "subject": "Легковой автомобиль",
            "cost": 60000.0,
            "currency": "BYN",
            "prepaid": 30.0,
        },
        "advance_sum": 18000,
        "num_payments": 36,
        "increase_percent": 12.5,
        "url": "http://example",
        "currency_conversion": {
            "from": "USD",
            "to": "BYN",
            "amount_from": 20000.0,
            "amount_to": 60000.0,
            "rate": 3.0,
        },
    }
    body = CalculatorTool(base_url="", token="").format_sms_body(result)
    assert body is not None
    assert "20000" in body and "долларов" in body, (
        f"SMS did not surface original USD amount: {body!r}"
    )
    assert "60000" in body, f"SMS missing BYN amount: {body!r}"


def test_sms_body_byn_only_unchanged() -> None:
    # Without currency_conversion metadata the SMS must render the legacy form.
    result = {
        "ok": True,
        "params": {
            "subject": "Легковой автомобиль",
            "cost": 60000.0,
            "currency": "BYN",
            "prepaid": 30.0,
        },
        "advance_sum": 18000,
        "num_payments": 36,
        "increase_percent": 12.5,
        "url": "http://example",
    }
    body = CalculatorTool(base_url="", token="").format_sms_body(result)
    assert body is not None
    assert "долларов" not in body
    assert "60000" in body


def test_profile_accepts_original_fields() -> None:
    # Dataclass schema sanity: the new fields must be assignable without
    # blowing existing constructors.
    p = ClientProfile()
    p.original_cost = 20000.0
    p.original_currency = "USD"
    assert p.original_cost == 20000.0
    assert p.original_currency == "USD"
