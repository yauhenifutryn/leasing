"""Fix 1.1 — deterministic renderer for post-calc voice summary.

The LLM must never synthesise the monetary numbers it speaks to the
client. The post-calc direct-call path builds a result-summary string
from calculator output, then asks the LLM to paraphrase it. Numbers
arrive fully formatted — the LLM contributes tone, not arithmetic.

This test locks in two guarantees:
  1. `render_calc_result` produces the canonical summary shared by the
     orchestrator's direct-call path. All numeric fields come from the
     `result` dict and are formatted deterministically.
  2. No LLM-facing prompt string in backend/ contains `{cost}`, `{prepaid}`,
     `{term}`, `{payment_min}`, or `{advance_sum}` as a template
     placeholder. If that regression ever lands (e.g. someone refactors
     and hands the LLM a format string with profile numbers), this test
     fails at CI time so we don't ship LLM-hallucinated figures.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_prompts import render_calc_result


def _minimal_result(**overrides):
    base = {
        "ok": True,
        "params": {
            "subject": "Легковой автомобиль",
            "cost": 60000.0,
            "currency": "BYN",
            "prepaid": 30.0,
        },
        "advance_sum": 18000,
        "payment_min": 1500,
        "buyout_sum": 1000,
        "total": 61000.0,
        "increase_percent": 12.5,
        "num_payments": 36,
        "defaulted": [],
    }
    base.update(overrides)
    return base


def test_render_calc_result_basic_byn() -> None:
    out = render_calc_result(_minimal_result())
    assert "18000" in out
    assert "1500" in out
    assert "1000" in out  # buyout
    assert "61000" in out
    assert "12.5" in out
    assert "36" in out


def test_render_calc_result_rounds_decimal_money() -> None:
    """Fix 1.8 — TTS cannot pronounce "536.55 USD" cleanly. Monetary
    fields must round to integers; percentages stay decimal when needed
    (12.5%), integer when whole (30%)."""
    out = render_calc_result(_minimal_result(
        advance_sum=6000.0,      # whole USD amount, must render "6000"
        payment_min=536.55,      # decimal that broke live call 674e3957
        buyout_sum=200.0,
        total=25516.039,
        increase_percent=12.5,   # kept decimal (percentage)
    ))
    assert "536.55" not in out, f"decimal leaked into spoken summary: {out}"
    # round(536.55) == 537 via banker's rounding in Python 3 — either 536
    # or 537 is an acceptable integer form; both are TTS-clean.
    assert " 537 " in out or " 536 " in out, f"rounded payment missing: {out}"
    assert "6000" in out
    assert "200" in out
    assert "25516" in out
    assert "12.5" in out          # percentage keeps one decimal


def test_render_calc_result_whole_percentage_no_trailing_zero() -> None:
    """30.0% should render as '30%' not '30.0%' for cleaner TTS."""
    out = render_calc_result(_minimal_result(
        params={"subject": "Легковой автомобиль", "cost": 60000.0,
                "currency": "BYN", "prepaid": 30.0},
    ))
    assert "30.0%" not in out, f"trailing .0 on percentage: {out}"
    assert "30%" in out


def test_render_calc_result_usd_disclosure_prefix() -> None:
    out = render_calc_result(_minimal_result(currency_conversion={
        "from": "USD",
        "to": "BYN",
        "amount_from": 20000.0,
        "amount_to": 60000.0,
        "rate": 3.0,
    }))
    assert "20000" in out and "долларов" in out
    assert "60000" in out
    assert "по курсу" in out.lower() or "курс" in out.lower()


def test_render_calc_result_defaults_note() -> None:
    out = render_calc_result(_minimal_result(defaulted=["prepaid", "term"]))
    assert "умолчанию" in out.lower()


def test_no_llm_prompt_has_numeric_templates() -> None:
    """Guardrail: no .py file under backend/ should hand the LLM a prompt
    string containing {cost}, {prepaid}, {term}, {payment_min}, or
    {advance_sum} as an f-string field. All number formatting must go
    through a deterministic renderer.

    This regex is deliberately tight — it only flags f-string field
    references like {cost} / {prepaid:,.0f}, not bare word occurrences."""
    backend_dir = ROOT / "backend"
    pattern = re.compile(r"\{(cost|prepaid|term|payment_min|advance_sum)(?:[:!][^}]*)?\}")
    offenders: list[str] = []
    for py in backend_dir.rglob("*.py"):
        # Deterministic renderer + calculator.format_sms_body legitimately
        # reference these fields inside f-strings derived from validated
        # dicts — they are the canonical safe renderers.
        if py.name in {"profile_prompts.py", "calculator.py"}:
            continue
        text = py.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            # Only flag references that look like f-string placeholders
            # inside a Russian prompt. A cheap proxy: the match is in a
            # line containing either "system_prompt" or Cyrillic letters
            # (LLM prompt lines are in Russian).
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            ctx = text[start:end]
            if re.search(r"[А-Яа-я]", ctx) or "system_prompt" in ctx:
                offenders.append(f"{py.relative_to(ROOT)}: {m.group(0)} in {ctx.strip()[:120]!r}")
    assert not offenders, (
        "Found LLM-facing prompts with numeric template placeholders:\n"
        + "\n".join(offenders)
        + "\n\nFix 1.1 requires all numeric rendering to go through "
        "profile_prompts.render_calc_result / build_readback_text / "
        "calculator.format_sms_body."
    )
