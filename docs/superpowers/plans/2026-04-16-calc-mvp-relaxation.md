# Calculator MVP Relaxation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip all defaults from the calculator, source every parameter from `ClientProfile`, accept prepaid as percent OR amount, forward `type_schedule`, auto-convert USD→BYN at 3:1 for physical persons, reject EUR/RUB for physical persons with an explicit message, and widen classifier accepted ranges to 0-40% prepaid / 12-84 months.

**Architecture:** `CalculatorTool.defaults()` returns `{}`. `CalculatorTool.execute()` raises `IncompleteProfileError` on missing fields. A pre-execute layer in the DirectTool path converts USD and rejects unsupported currencies for physical persons before the API call. The classifier prompt loses its `invalid_param` 10% floor and gains `type_schedule` extraction.

**Tech Stack:** Python 3.12, httpx, pytest.

**Spec:** `docs/superpowers/specs/2026-04-16-calc-mvp-relaxation-design.md`

**Depends on:** session-agent-profile plan Task 1-2 (`ClientProfile` must exist). Can be implemented in parallel but must land together to avoid mid-deploy breakage.

---

### Task 1: Custom exceptions

**Files:**
- Modify: `rag_demo_system/backend/tools/calculator.py`

- [ ] **Step 1: Write failing test**

Create `rag_demo_system/tests/test_calculator_exceptions.py`:

```python
"""Calculator exception types for precise error handling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_incomplete_profile_error_importable() -> None:
    from backend.tools.calculator import IncompleteProfileError
    err = IncompleteProfileError(missing=["cost", "term_months"])
    assert "cost" in str(err)
    assert "term_months" in str(err)
    assert err.missing == ["cost", "term_months"]


def test_unsupported_currency_error_importable() -> None:
    from backend.tools.calculator import UnsupportedCurrencyError
    err = UnsupportedCurrencyError(currency="EUR", client_type="Физическое лицо")
    assert "EUR" in str(err)
    assert err.currency == "EUR"
    assert err.client_type == "Физическое лицо"
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator_exceptions.py -v`
Expected: FAIL — classes not defined.

- [ ] **Step 3: Add exception classes**

Open `rag_demo_system/backend/tools/calculator.py`. Add near top of file (after imports):

```python
class IncompleteProfileError(Exception):
    """Raised when calculator is invoked without all required profile fields."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        super().__init__(f"Калькулятор требует обязательные поля: {', '.join(self.missing)}")


class UnsupportedCurrencyError(Exception):
    """Raised when a currency is not allowed for a client type (MVP: EUR/RUB for Физ лицо)."""

    def __init__(self, currency: str, client_type: str) -> None:
        self.currency = currency
        self.client_type = client_type
        super().__init__(
            f"Валюта {currency} не поддерживается для клиента '{client_type}'. "
            "Поддерживаются: BYN, USD (физлицо с конвертацией по курсу 3)."
        )
```

- [ ] **Step 4: Run test, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator_exceptions.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/tools/calculator.py rag_demo_system/tests/test_calculator_exceptions.py
git commit -m "feat(calculator): add IncompleteProfileError and UnsupportedCurrencyError"
```

---

### Task 2: Remove defaults from `CalculatorTool`

**Files:**
- Modify: `rag_demo_system/backend/tools/calculator.py`

- [ ] **Step 1: Write failing test**

Create `rag_demo_system/tests/test_calculator_no_defaults.py`:

```python
"""Calculator must never apply hidden defaults; missing fields raise IncompleteProfileError."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.calculator import CalculatorTool, IncompleteProfileError  # noqa: E402


def _make_tool() -> CalculatorTool:
    return CalculatorTool(base_url="http://example.invalid", token="x")


def test_defaults_returns_empty() -> None:
    tool = _make_tool()
    assert tool.defaults() == {}


def test_execute_raises_on_empty_params() -> None:
    tool = _make_tool()
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute({}, {})
    missing = exc.value.missing
    for required in ["subject", "cost", "client_type", "currency", "condition_new",
                     "term_months", "type_schedule"]:
        assert required in missing or "prepaid" in missing


def test_execute_raises_when_prepaid_missing() -> None:
    tool = _make_tool()
    params = dict(
        subject="Легковой автомобиль",
        cost=70000,
        client_type="Физическое лицо",
        currency="BYN",
        condition_new=1,
        term_months=60,
        type_schedule="0",
    )
    with pytest.raises(IncompleteProfileError) as exc:
        tool.execute(params, {})
    assert "prepaid" in exc.value.missing
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator_no_defaults.py -v`
Expected: all three FAIL.

- [ ] **Step 3: Modify `CalculatorTool`**

In `rag_demo_system/backend/tools/calculator.py`:

(a) Replace the `defaults()` method body:

```python
def defaults(self) -> dict[str, Any]:
    """No defaults. All parameters must come from the confirmed ClientProfile."""
    return {}
```

(b) Modify `execute()` to validate and raise. Replace the current body's opening block (from the method def through the `filled, defaulted = self.fill_defaults(params)` line) with:

```python
def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
    REQUIRED = ["subject", "cost", "client_type", "currency",
                "condition_new", "term_months", "type_schedule"]
    missing = [k for k in REQUIRED if params.get(k) in (None, "")]
    if params.get("prepaid_pct") in (None, "") and params.get("prepaid_amount") in (None, ""):
        missing.append("prepaid")
    if params.get("condition_new") == 0 and params.get("age_years") in (None, ""):
        missing.append("age_years")
    if missing:
        raise IncompleteProfileError(missing=missing)

    # Normalize subject & client_type (existing maps)
    _subj = params.get("subject", "")
    params["subject"] = self._SUBJECT_MAP.get(_subj.lower().strip(), _subj)
    _ct = params.get("client_type", "")
    params["client_type"] = self._CLIENT_TYPE_MAP.get(_ct.lower().strip(), _ct)

    # Resolve prepaid: pct OR amount. If amount, derive pct for API call.
    if params.get("prepaid_pct") is None and params.get("prepaid_amount") is not None:
        cost = float(params["cost"])
        amount = float(params["prepaid_amount"])
        if cost <= 0:
            raise IncompleteProfileError(missing=["cost"])
        params["prepaid_pct"] = round((amount / cost) * 100.0, 2)
    prepaid_pct = float(params["prepaid_pct"])
    if prepaid_pct < 0 or prepaid_pct > 40:
        raise IncompleteProfileError(missing=[f"prepaid_pct_out_of_range:{prepaid_pct}"])

    # No fill_defaults call anymore.
    filled = {
        "client_type": params["client_type"],
        "subject": params["subject"],
        "condition_new": params["condition_new"],
        "currency": params["currency"],
        "cost": params["cost"],
        "prepaid": prepaid_pct,                 # API expects percent under key `prepaid`
        "term": params["term_months"],
        "type_schedule": params["type_schedule"],
    }
    if params.get("condition_new") == 0:
        filled["age"] = params["age_years"]
    defaulted: list[str] = []  # no defaults ever

    # ... continue with existing API call block (unchanged below)
```

Leave the rest of `execute()` (API call, response parse) as-is. Also keep `fill_defaults` function body but ensure it's no longer invoked from `execute`. You may delete `fill_defaults` entirely if it's not used elsewhere.

Also: update `_parse_response` to include both `prepaid_pct` and `prepaid_amount` in the returned dict:

```python
return {
    "ok": True,
    "url": advance.get("URL", ""),
    # ... existing fields ...
    "prepaid_pct": params.get("prepaid"),
    "prepaid_amount": advance.get("sum", 0),  # api returns absolute amount in advance.sum
    "params": params,
    "defaulted": defaulted,
}
```

- [ ] **Step 4: Run tests, verify PASS and other calc tests not broken**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator_no_defaults.py tests/test_calculator.py -v`
Expected: new tests PASS. Existing `test_calculator.py` may FAIL because its old assertions used defaults. If any fail:
- Update the old tests to pass all required fields explicitly.
- Do NOT regress the default-removal logic; update the tests to match.

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/tools/calculator.py rag_demo_system/tests/test_calculator_no_defaults.py rag_demo_system/tests/test_calculator.py
git commit -m "refactor(calculator): remove all defaults; raise on incomplete params"
```

---

### Task 3: USD→BYN conversion + EUR/RUB rejection for физических лиц

**Files:**
- Modify: `rag_demo_system/backend/config.py` (add `USD_BYN_RATE`)
- Modify: `rag_demo_system/backend/app.py` (DirectTool pre-step)
- Modify: `rag_demo_system/.env.example` (add `USD_BYN_RATE=3.0`)
- Create: `rag_demo_system/tests/test_calculator_currency.py`

- [ ] **Step 1: Write failing tests**

```python
"""Currency handling: USD->BYN hardcoded conversion for Физ лицо, EUR/RUB rejected."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_usd_to_byn_conversion_for_individual(monkeypatch) -> None:
    monkeypatch.setenv("USD_BYN_RATE", "3.0")
    from backend.app import _apply_currency_policy

    params = {
        "client_type": "Физическое лицо", "currency": "USD", "cost": 24300,
        # other fields omitted for brevity; helper only cares about currency+cost+client_type
    }
    out = _apply_currency_policy(params)
    assert out["currency"] == "BYN"
    assert out["cost"] == 24300 * 3.0
    assert out["currency_conversion"]["rate"] == 3.0
    assert out["currency_conversion"]["from"] == "USD"


def test_usd_untouched_for_legal_entity() -> None:
    from backend.app import _apply_currency_policy
    params = {"client_type": "Юридическое лицо", "currency": "USD", "cost": 30000}
    out = _apply_currency_policy(params)
    assert out["currency"] == "USD"  # legal entities may keep USD
    assert out["cost"] == 30000


def test_eur_rejected_for_individual() -> None:
    from backend.app import _apply_currency_policy
    from backend.tools.calculator import UnsupportedCurrencyError
    params = {"client_type": "Физическое лицо", "currency": "EUR", "cost": 24000}
    with pytest.raises(UnsupportedCurrencyError):
        _apply_currency_policy(params)


def test_rub_rejected_for_individual() -> None:
    from backend.app import _apply_currency_policy
    from backend.tools.calculator import UnsupportedCurrencyError
    params = {"client_type": "Физическое лицо", "currency": "RUB", "cost": 2_500_000}
    with pytest.raises(UnsupportedCurrencyError):
        _apply_currency_policy(params)
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator_currency.py -v`
Expected: FAIL — `_apply_currency_policy` does not exist.

- [ ] **Step 3: Add `USD_BYN_RATE` setting**

In `rag_demo_system/backend/config.py`, add:

```python
usd_byn_rate: float = Field(
    default=3.0,
    description="MVP-only hardcoded USD->BYN conversion rate for physical persons. "
                "Remove when calculator API provides server-side currency conversion."
)
```

- [ ] **Step 4: Implement `_apply_currency_policy` in `app.py`**

Add near `_run_session_agent`:

```python
def _apply_currency_policy(params: dict) -> dict:
    """MVP currency policy: convert USD->BYN for Физ лицо; reject EUR/RUB for Физ лицо.

    Returns a new dict (does not mutate input). Raises UnsupportedCurrencyError.
    """
    from .tools.calculator import UnsupportedCurrencyError
    from .config import get_settings

    out = dict(params)
    ct = out.get("client_type")
    cur = out.get("currency")
    if ct != "Физическое лицо":
        return out
    if cur == "USD":
        rate = float(get_settings().usd_byn_rate)
        old_cost = float(out["cost"])
        out["cost"] = round(old_cost * rate, 2)
        out["currency"] = "BYN"
        out["currency_conversion"] = {
            "from": "USD", "to": "BYN",
            "amount_from": old_cost, "amount_to": out["cost"],
            "rate": rate, "rate_source": "MVP hardcoded",
        }
        return out
    if cur in ("EUR", "RUB", "RUR", "CNY"):
        raise UnsupportedCurrencyError(currency=cur, client_type=ct)
    return out
```

Use whatever is the existing accessor for `Settings` (e.g. `settings` global or `get_settings()` function — match the pattern already used).

- [ ] **Step 5: Run tests, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator_currency.py -v`
Expected: PASS.

- [ ] **Step 6: Update `.env.example`**

Append:

```
# MVP hardcoded USD->BYN conversion rate for physical persons.
# Remove when calculator API adds server-side conversion.
USD_BYN_RATE=3.0
```

- [ ] **Step 7: Commit**

```bash
git add rag_demo_system/backend/config.py rag_demo_system/backend/app.py rag_demo_system/.env.example rag_demo_system/tests/test_calculator_currency.py
git commit -m "feat(calculator): USD->BYN 3:1 conversion for individuals; reject EUR/RUB"
```

---

### Task 4: DirectTool path consumes `ClientProfile`

**Files:**
- Modify: `rag_demo_system/backend/app.py` — the calculator invocation section

- [ ] **Step 1: Write failing test**

Create `rag_demo_system/tests/test_direct_tool_from_profile.py`:

```python
"""DirectTool path: builds params from ClientProfile, applies currency policy, forwards type_schedule."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_build_calc_params_from_profile_forwards_type_schedule() -> None:
    from backend.app import _build_calc_params_from_profile
    from backend.session import ClientProfile

    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=70000,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=84,
        type_schedule="1",
    )
    params = _build_calc_params_from_profile(p)
    assert params["type_schedule"] == "1"
    assert params["prepaid_pct"] == 20.0
    assert params["term_months"] == 84


def test_build_calc_params_from_profile_keeps_prepaid_amount() -> None:
    from backend.app import _build_calc_params_from_profile
    from backend.session import ClientProfile

    p = ClientProfile(
        client_type="Физическое лицо", subject="Легковой автомобиль",
        cost=70000, currency="BYN", condition_new=1,
        prepaid_amount=14000, term_months=60, type_schedule="0",
    )
    params = _build_calc_params_from_profile(p)
    assert params["prepaid_amount"] == 14000
    assert "prepaid_pct" not in params or params.get("prepaid_pct") is None
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_direct_tool_from_profile.py -v`
Expected: FAIL — helper absent.

- [ ] **Step 3: Add `_build_calc_params_from_profile`**

In `rag_demo_system/backend/app.py`:

```python
def _build_calc_params_from_profile(profile) -> dict:
    """Serialize ClientProfile to calculator params dict. No defaults applied."""
    params = {
        "client_type": profile.client_type,
        "subject": profile.subject,
        "cost": profile.cost,
        "currency": profile.currency,
        "condition_new": profile.condition_new,
        "term_months": profile.term_months,
        "type_schedule": profile.type_schedule,
    }
    if profile.age_years is not None:
        params["age_years"] = profile.age_years
    if profile.prepaid_pct is not None:
        params["prepaid_pct"] = profile.prepaid_pct
    elif profile.prepaid_amount is not None:
        params["prepaid_amount"] = profile.prepaid_amount
    return params
```

- [ ] **Step 4: Wire the decision path in `_stream_voice_response`**

Locate the calculator call (the `_direct_tool_result = await asyncio.to_thread(calc_tool.execute, ...)` block, around line 1000). Replace with:

```python
# Already passed readback gate (decision["action"] == "calc")
raw_params = _build_calc_params_from_profile(session.client_profile)
try:
    raw_params = _apply_currency_policy(raw_params)
except UnsupportedCurrencyError as exc:
    text = (
        "Сейчас я могу считать в белорусских рублях или долларах. "
        "В какой из этих валют стоимость?"
    )
    session.client_profile.currency = None   # ask currency again
    session.client_profile.confirmed_at = None  # re-confirm after currency change
    await _speak_text(websocket, session, text)
    return

conv = raw_params.pop("currency_conversion", None)
if conv is not None:
    disclosure = (
        f"По курсу {conv['rate']:.0f} рубля за доллар это "
        f"{conv['amount_to']:.0f} BYN."
    )
else:
    disclosure = ""

try:
    _direct_tool_result = await asyncio.to_thread(
        calc_tool.execute, raw_params, {}
    )
except IncompleteProfileError as exc:
    print(f"[DirectTool] IncompleteProfileError: missing={exc.missing}", flush=True)
    # Should not happen — state machine guards this. Log and fallback.
    await _speak_text(websocket, session, "Секундочку, уточним параметры.")
    return

# Presentation (existing pattern + conversion disclosure prepended)
_p = _direct_tool_result.get("params", {})
_cur = _p.get("currency", "BYN")
summary = (
    (disclosure + " " if disclosure else "") +
    f"Аванс {_p.get('prepaid', '?')} процентов — {_direct_tool_result.get('advance_sum')} {_cur}. "
    f"Ежемесячный платёж — {_direct_tool_result.get('payment_min')} {_cur}. "
    f"Срок — {_direct_tool_result.get('num_payments')} месяцев. "
    f"Общая сумма — {_direct_tool_result.get('total'):.0f} {_cur}. "
    f"Удорожание — {_direct_tool_result.get('increase_percent')} процентов. "
    "Отправить график по СМС или изменить параметры?"
)
await _speak_text(websocket, session, summary)
```

Ensure `IncompleteProfileError` and `UnsupportedCurrencyError` are imported at top of `app.py`:

```python
from .tools.calculator import IncompleteProfileError, UnsupportedCurrencyError
```

Remove any now-dead code (the old hint-based parameter building, the old 30% prepaid assumption, the `defaulted` handling in the summary).

- [ ] **Step 5: Run full suite**

Run: `cd rag_demo_system && python -m pytest tests/ -v -x`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/tests/test_direct_tool_from_profile.py
git commit -m "feat(app): DirectTool path sources params from ClientProfile, forwards type_schedule"
```

---

### Task 5: Relax classifier ranges + extract type_schedule

**Files:**
- Modify: `rag_demo_system/backend/app.py` — the SessionAgent prompt (may overlap with session-agent-profile plan Task 3; coordinate merge order)

- [ ] **Step 1: Verify current state**

Open `rag_demo_system/backend/app.py`. Find the SessionAgent system prompt string (set in `_run_session_agent` after the session-agent-profile plan Task 3 landed).

- [ ] **Step 2: Remove hardcoded "min 10%" and "до 36" language**

Inside the system prompt string:
- Delete the line: `"prepaid: минимум 10%. Если клиент просит 0% или 5%, ставь action='invalid_param'."`
- Delete: `"Если клиент просит аванс ниже 10%, ставь action='invalid_param'."`
- Replace with:
  `"prepaid: допустимый диапазон 0-40 процентов. Термин 12-84 месяца. Не ставь 'invalid_param' — пусть API решает."`

- [ ] **Step 3: Confirm type_schedule extraction rule is present**

Verify the prompt contains the rule added in session-agent-profile Task 3:

```
type_schedule: 'аннуитет/аннуитетный' = '0'; 'линейный/убывающий/дифференцированный' = '1'.
```

If missing, add it.

- [ ] **Step 4: Write integration test**

Append to `rag_demo_system/tests/test_session_agent_schema.py`:

```python
def test_session_agent_extracts_type_schedule_linear() -> None:
    from backend import app
    from unittest.mock import patch
    from types import SimpleNamespace
    import json

    fake = SimpleNamespace(text=json.dumps({
        "intent": "TOOL",
        "profile_patches": {"type_schedule": "1"},
        "is_confirmation": False, "is_stop_request": False,
        "wants_readback": False, "change_field": None,
        "change_value": None, "action": "change_param",
    }, ensure_ascii=False))

    with patch("backend.llm.call_openai_compatible", return_value=fake):
        result = _run(app._run_session_agent(
            "сделай линейный график",
            SimpleNamespace(transcript=[]),
            SimpleNamespace(tool_calls_this_turn=[]),
            tool_schemas=[{}], session_id="test",
        ))

    assert result["profile_patches"]["type_schedule"] == "1"
```

- [ ] **Step 5: Run test, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_session_agent_schema.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/tests/test_session_agent_schema.py
git commit -m "feat(session-agent): widen prepaid/term ranges; extract type_schedule"
```

---

### Task 6: Manual transcript replay validation

- [ ] **Step 1: After full deploy, replay the critical points from transcript 779154b4**

Via SIP or browser mic:

1. Say only: *"Здравствуйте."* Expect bot asks for name (no calculator).
2. Say: *"Меня зовут Сергей, хочу легковой автомобиль."* Expect bot asks next missing field (cost, client_type, or currency).
3. Say: *"Стоимость семьдесят тысяч рублей, я физлицо, новый."* Expect bot continues asking remaining fields (term, prepaid, schedule).
4. Complete the remaining: *"Срок восемьдесят четыре месяца, аванс двадцать процентов, аннуитетный."* Expect bot says **readback**, not calc.
5. Say: *"Да, всё верно."* Expect calculator fires, bot delivers summary.
6. Say: *"Сделай линейный график."* Expect change-confirmation ("меняю тип графика на линейный, всё верно?").
7. Say: *"Да."* Expect recalc with `type_schedule=1` (verify via backend log).
8. Say: *"Стоимость двадцать четыре тысячи триста долларов."* Expect conversion disclosure ("по курсу 3 рубля это 72 900 BYN") then recalc.

No code changes — observational validation.

---

## Self-review

**Spec coverage:**
- Remove defaults → Task 2 ✓
- Prepaid pct XOR amount → Task 2 ✓
- Currency policy (USD conversion, EUR/RUB reject) → Task 3 ✓
- DirectTool consumes ClientProfile → Task 4 ✓
- Classifier range relax + type_schedule extract → Task 5 ✓
- System prompt rewrite → covered by session-agent-profile plan Task 5
- Post-calc summary with full fields → Task 4 ✓
- IncompleteProfileError + UnsupportedCurrencyError → Task 1 ✓

**Placeholders:** none.

**Type consistency:**
- Calculator params: `subject`, `cost`, `client_type`, `currency`, `condition_new`, `age_years`, `prepaid_pct` XOR `prepaid_amount`, `term_months`, `type_schedule`.
- Same keys used in `_build_calc_params_from_profile`, `_apply_currency_policy`, `CalculatorTool.execute`.
- `currency_conversion` dict keys: `from`, `to`, `amount_from`, `amount_to`, `rate`, `rate_source`. Consumed at presentation layer.
