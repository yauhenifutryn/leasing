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
