# SessionAgent + ClientProfile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `ClientProfile` dataclass plus a state-machine-driven SessionAgent that collects leasing parameters incrementally, confirms them through a semantic read-back gate, and refuses to call the calculator until the profile is complete and confirmed.

**Architecture:** `backend/session.py` owns `ClientProfile`. `backend/app.py` SessionAgent helper (extracted in the classifier-latency plan) expands its JSON schema to emit profile patches, confirmation flags, stop-request, change-field intents. State machine transitions tracked on `ChatSession`. System prompt rewritten to describe the collection protocol with no defaults.

**Tech Stack:** Python 3.12 dataclasses, asyncio, pytest, existing SessionAgent LLM call.

**Spec:** `docs/superpowers/specs/2026-04-16-session-agent-profile-design.md`

**Depends on:** classifier-latency plan (Task 2 extracts `_run_session_agent`) — recommended but not strictly required; this plan can proceed with the inline classifier and be patched later.

---

### Task 1: `ClientProfile` dataclass with completeness logic

**Files:**
- Create: `rag_demo_system/backend/session.py` (if not present) OR modify existing
- Create: `rag_demo_system/tests/test_client_profile.py`

- [ ] **Step 1: Write failing tests**

Create `rag_demo_system/tests/test_client_profile.py`:

```python
"""Tests for ClientProfile: completeness checks and field merge semantics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from backend.session import ClientProfile  # noqa: E402


def test_empty_profile_is_not_complete() -> None:
    p = ClientProfile()
    assert p.is_complete_for_calc() is False
    missing = p.missing_fields()
    assert "client_type" in missing
    assert "subject" in missing
    assert "cost" in missing


def test_profile_with_all_fields_is_complete() -> None:
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=70000.0,
        currency="BYN",
        condition_new=1,
        prepaid_pct=20.0,
        term_months=84,
        type_schedule="0",
    )
    assert p.is_complete_for_calc() is True
    assert p.missing_fields() == set()


def test_used_subject_requires_age() -> None:
    p = ClientProfile(
        client_type="Физическое лицо",
        subject="Легковой автомобиль",
        cost=50000.0,
        currency="BYN",
        condition_new=0,  # used
        prepaid_pct=20.0,
        term_months=60,
        type_schedule="0",
    )
    assert p.is_complete_for_calc() is False
    assert "age_years" in p.missing_fields()


def test_prepaid_either_pct_or_amount() -> None:
    """prepaid_pct OR prepaid_amount satisfies the requirement."""
    base = dict(
        client_type="Физическое лицо", subject="Легковой автомобиль",
        cost=70000.0, currency="BYN", condition_new=1,
        term_months=60, type_schedule="0",
    )
    assert ClientProfile(**base, prepaid_pct=20.0).is_complete_for_calc() is True
    assert ClientProfile(**base, prepaid_amount=14000.0).is_complete_for_calc() is True
    assert ClientProfile(**base).is_complete_for_calc() is False


def test_apply_patches_updates_fields() -> None:
    p = ClientProfile()
    p.apply_patches({"subject": "Легковой автомобиль", "cost": 70000.0})
    assert p.subject == "Легковой автомобиль"
    assert p.cost == 70000.0


def test_apply_patches_skips_none() -> None:
    """None values in patches must NOT overwrite existing fields."""
    p = ClientProfile(subject="Легковой автомобиль")
    p.apply_patches({"subject": None, "cost": 70000.0})
    assert p.subject == "Легковой автомобиль"
    assert p.cost == 70000.0


def test_apply_patches_respects_locked_fields() -> None:
    p = ClientProfile(term_months=84)
    p.locked_fields.add("term_months")
    p.apply_patches({"term_months": 48})
    assert p.term_months == 84  # unchanged
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_client_profile.py -v`
Expected: All fail — `ClientProfile` not yet defined.

- [ ] **Step 3: Create / modify `backend/session.py`**

Check if `rag_demo_system/backend/session.py` already exists. If yes, append; if no, create with full content:

```python
"""Per-session state: ChatSession, ClientProfile, and related types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

ClientType = Literal["Физическое лицо", "Юридическое лицо"]
ScheduleType = Literal["0", "1"]


@dataclass
class ClientProfile:
    """Leasing client parameters collected during the session.

    All fields Optional to allow incremental population. is_complete_for_calc()
    returns True only when every calculator-required field is set.
    """

    name: Optional[str] = None
    client_type: Optional[ClientType] = None
    subject: Optional[str] = None
    cost: Optional[float] = None
    currency: Optional[str] = None
    condition_new: Optional[int] = None
    age_years: Optional[int] = None
    prepaid_pct: Optional[float] = None
    prepaid_amount: Optional[float] = None
    term_months: Optional[int] = None
    type_schedule: Optional[ScheduleType] = None

    # State-machine bookkeeping
    confirmed_at: Optional[float] = None
    last_change_pending: Optional[str] = None
    locked_fields: set[str] = field(default_factory=set)

    _CORE_FIELDS = (
        "client_type", "subject", "cost", "currency",
        "condition_new", "term_months", "type_schedule",
    )

    def missing_fields(self) -> set[str]:
        missing: set[str] = set()
        for f_name in self._CORE_FIELDS:
            if getattr(self, f_name) is None:
                missing.add(f_name)
        if self.prepaid_pct is None and self.prepaid_amount is None:
            missing.add("prepaid")
        if self.condition_new == 0 and self.age_years is None:
            missing.add("age_years")
        return missing

    def is_complete_for_calc(self) -> bool:
        return not self.missing_fields()

    def apply_patches(self, patches: dict[str, Any]) -> dict[str, Any]:
        """Merge non-None patches into the profile, respecting locked_fields.

        Returns a dict of fields actually changed (for logging / telemetry).
        """
        changed: dict[str, Any] = {}
        for k, v in patches.items():
            if v is None:
                continue
            if k in self.locked_fields:
                continue
            if not hasattr(self, k):
                continue
            old = getattr(self, k)
            if old != v:
                setattr(self, k, v)
                changed[k] = v
        return changed

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict (for logging, snapshots, calculator params)."""
        return {
            "name": self.name, "client_type": self.client_type, "subject": self.subject,
            "cost": self.cost, "currency": self.currency,
            "condition_new": self.condition_new, "age_years": self.age_years,
            "prepaid_pct": self.prepaid_pct, "prepaid_amount": self.prepaid_amount,
            "term_months": self.term_months, "type_schedule": self.type_schedule,
            "confirmed_at": self.confirmed_at,
            "last_change_pending": self.last_change_pending,
            "locked_fields": sorted(self.locked_fields),
        }
```

If `session.py` already holds a `ChatSession`, add the `ClientProfile` import reference there. If `ChatSession` lives in another file, note its path and add `client_profile: ClientProfile = field(default_factory=ClientProfile)` to it in Task 2.

- [ ] **Step 4: Run tests, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_client_profile.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/session.py rag_demo_system/tests/test_client_profile.py
git commit -m "feat(session): add ClientProfile dataclass with completeness and patch logic"
```

---

### Task 2: Attach `ClientProfile` to `ChatSession`

**Files:**
- Modify: wherever `ChatSession` (or the active session dataclass) is defined

- [ ] **Step 1: Locate `ChatSession`**

Run: `cd rag_demo_system && grep -rn "class ChatSession" backend/`

- [ ] **Step 2: Write failing test**

Append to `rag_demo_system/tests/test_client_profile.py`:

```python
def test_chat_session_has_client_profile() -> None:
    # Exact import path based on grep result; adjust if different
    from backend.app import ChatSession  # or wherever it lives

    cs = ChatSession(session_id="test")  # use whatever required args exist
    assert hasattr(cs, "client_profile")
    assert isinstance(cs.client_profile, ClientProfile)
    assert cs.client_profile.is_complete_for_calc() is False
```

- [ ] **Step 3: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_client_profile.py::test_chat_session_has_client_profile -v`
Expected: FAIL — attribute missing.

- [ ] **Step 4: Add field to `ChatSession`**

Add to the `ChatSession` dataclass:

```python
from .session import ClientProfile

@dataclass
class ChatSession:
    # ... existing fields
    client_profile: ClientProfile = field(default_factory=ClientProfile)
```

If `ChatSession` isn't a dataclass but a plain class, add it in `__init__`:

```python
self.client_profile = ClientProfile()
```

- [ ] **Step 5: Run test, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_client_profile.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "feat(session): attach ClientProfile to ChatSession"
```

---

### Task 3: SessionAgent richer JSON schema

**Files:**
- Modify: `rag_demo_system/backend/app.py` — the `_run_session_agent` helper (or classifier block if not yet extracted)
- Modify: `rag_demo_system/config/system_prompt_ru_v2.txt` (indirect: prompt feeds SessionAgent)

- [ ] **Step 1: Write failing test**

Create `rag_demo_system/tests/test_session_agent_schema.py`:

```python
"""SessionAgent JSON schema: new fields emitted and parsed correctly."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_session_agent_emits_profile_patches() -> None:
    """SessionAgent output must include `profile_patches` field."""
    from backend import app

    fake_llm_response = SimpleNamespace(text=json.dumps({
        "intent": "TOOL",
        "profile_patches": {"subject": "Легковой автомобиль", "cost": 70000, "currency": "BYN"},
        "is_confirmation": False,
        "is_stop_request": False,
        "wants_readback": False,
        "change_field": None,
        "change_value": None,
        "action": "calculate",
    }, ensure_ascii=False))

    with patch("backend.llm.call_openai_compatible", return_value=fake_llm_response):
        fake_session = SimpleNamespace(tool_calls_this_turn=[])
        fake_chat_session = SimpleNamespace(transcript=[])
        result = _run(app._run_session_agent(
            "хочу легковой за семьдесят тысяч",
            fake_chat_session, fake_session,
            tool_schemas=[{}], session_id="test",
        ))

    assert "profile_patches" in result
    assert result["profile_patches"]["subject"] == "Легковой автомобиль"
    assert result["profile_patches"]["cost"] == 70000
    assert result["is_stop_request"] is False


def test_session_agent_is_stop_request_true() -> None:
    from backend import app

    fake_llm_response = SimpleNamespace(text=json.dumps({
        "intent": "CONVERSATION",
        "profile_patches": {},
        "is_confirmation": False,
        "is_stop_request": True,
        "wants_readback": False,
        "change_field": None, "change_value": None,
        "action": None,
    }, ensure_ascii=False))

    with patch("backend.llm.call_openai_compatible", return_value=fake_llm_response):
        result = _run(app._run_session_agent(
            "стоп",
            SimpleNamespace(transcript=[]),
            SimpleNamespace(tool_calls_this_turn=[]),
            tool_schemas=[{}], session_id="test",
        ))

    assert result["is_stop_request"] is True


def test_session_agent_is_confirmation_true_on_readback() -> None:
    from backend import app

    fake_llm_response = SimpleNamespace(text=json.dumps({
        "intent": "CONVERSATION",
        "profile_patches": {},
        "is_confirmation": True,
        "is_stop_request": False,
        "wants_readback": False,
        "change_field": None, "change_value": None,
        "action": "confirm",
    }, ensure_ascii=False))

    with patch("backend.llm.call_openai_compatible", return_value=fake_llm_response):
        result = _run(app._run_session_agent(
            "да всё верно",
            SimpleNamespace(transcript=[]),
            SimpleNamespace(tool_calls_this_turn=[]),
            tool_schemas=[{}], session_id="test",
        ))

    assert result["is_confirmation"] is True
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_session_agent_schema.py -v`
Expected: FAIL — `profile_patches` and related keys absent from output.

- [ ] **Step 3: Rewrite SessionAgent prompt and parser**

In `rag_demo_system/backend/app.py`, locate `_run_session_agent` (or the classifier block if not yet extracted). Replace the `system_prompt=...` argument with:

```python
system_prompt=(
    "Ты SessionAgent голосового бота лизинговой компании. "
    "Анализируешь НОВОЕ сообщение клиента в контексте диалога. "
    "Возвращаешь строго JSON:\n"
    '{"intent": "TOOL"|"RAG"|"CONVERSATION",\n'
    ' "profile_patches": {"subject": ...|null, "cost": ...|null, '
    '"currency": "BYN"|"USD"|null, '
    '"client_type": "Физическое лицо"|"Юридическое лицо"|null, '
    '"condition_new": 1|0|null, "age_years": ...|null, '
    '"prepaid_pct": ...|null, "prepaid_amount": ...|null, '
    '"term_months": ...|null, "type_schedule": "0"|"1"|null, '
    '"name": "..."|null},\n'
    ' "is_confirmation": true|false,\n'
    ' "is_stop_request": true|false,\n'
    ' "wants_readback": true|false,\n'
    ' "change_field": "..."|null, "change_value": ...|null,\n'
    ' "action": "calculate"|"recalculate"|"sms"|"clarify"|"confirm"|null}\n\n'
    "Правила:\n"
    "- profile_patches: извлекай ТОЛЬКО явно сказанное в НОВОМ сообщении. "
    "НЕ переноси значения из истории. Если не упомянуто, ставь null.\n"
    "- ИП / индивидуальный предприниматель маппится в 'Юридическое лицо'.\n"
    "- type_schedule: 'аннуитет/аннуитетный' = '0'; 'линейный/убывающий/дифференцированный' = '1'.\n"
    "- prepaid_pct: клиент сказал процент (напр. '20 процентов' -> 20). "
    "prepaid_amount: клиент назвал сумму в валюте (напр. '14 тысяч' при BYN -> 14000). Одно из двух, не оба.\n"
    "- is_confirmation: клиент подтверждает предыдущее предложение бота ('да', 'всё верно', 'давай', 'согласен').\n"
    "- is_stop_request: клиент явно просит молчать/подождать ('стоп', 'подожди', 'помолчи', 'хватит', 'не продолжай').\n"
    "- wants_readback: клиент просит повторить параметры ('повтори', 'какие параметры').\n"
    "- change_field+change_value: клиент явно меняет конкретный параметр ('поменяй срок на 48' -> change_field='term_months', change_value=48).\n"
    "- intent=TOOL если клиент хочет рассчитать, посчитать, изменить параметры, отправить СМС. RAG если задаёт информационный вопрос. CONVERSATION если короткая реакция (да/нет/стоп/спасибо).\n"
    "Только JSON, никаких пояснений."
),
```

Bump `max_tokens` from 80 to 220 (richer output).

Rewrite the parse block to produce the new return dict:

```python
_raw = classify_resp.text.strip()
_js_start = _raw.find("{")
_js_end = _raw.rfind("}") + 1
parsed = {}
if _js_start >= 0 and _js_end > _js_start:
    import json as _json_classify
    try:
        parsed = _json_classify.loads(_raw[_js_start:_js_end])
    except Exception:
        parsed = {}

return {
    "intent": parsed.get("intent", "RAG"),
    "needs_tool": parsed.get("intent") == "TOOL",
    "profile_patches": parsed.get("profile_patches") or {},
    "is_confirmation": bool(parsed.get("is_confirmation")),
    "is_stop_request": bool(parsed.get("is_stop_request")),
    "wants_readback": bool(parsed.get("wants_readback")),
    "change_field": parsed.get("change_field"),
    "change_value": parsed.get("change_value"),
    "action": parsed.get("action"),
    "latency_ms": _t_classify_ms,
    "raw": _raw,
}
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_session_agent_schema.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/tests/test_session_agent_schema.py
git commit -m "feat(session-agent): richer JSON schema with profile_patches and semantic flags"
```

---

### Task 4: Profile merge and readback state machine

**Files:**
- Modify: `rag_demo_system/backend/app.py` — `_stream_voice_response` around line 950-1020 (DirectTool path)

- [ ] **Step 1: Write failing integration test**

Create `rag_demo_system/tests/test_readback_state_machine.py`:

```python
"""State-machine integration: profile patches applied + readback gate blocks calc."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_calculator_not_called_before_profile_complete() -> None:
    """With an incomplete profile, the DirectTool path must not invoke calculator."""
    from backend.app import _direct_tool_from_profile  # helper to be added

    profile = SimpleNamespace(
        is_complete_for_calc=lambda: False,
        missing_fields=lambda: {"term_months"},
        confirmed_at=None,
    )
    result = _direct_tool_from_profile(profile)
    assert result["action"] == "ask_next_missing"
    assert "term_months" in result["missing"]
    assert result.get("calc_called", False) is False


def test_calculator_called_only_after_confirmation() -> None:
    """Profile complete but not confirmed -> readback, not calc."""
    from backend.app import _direct_tool_from_profile

    profile = SimpleNamespace(
        is_complete_for_calc=lambda: True,
        missing_fields=lambda: set(),
        confirmed_at=None,
    )
    result = _direct_tool_from_profile(profile)
    assert result["action"] == "readback"
    assert result.get("calc_called", False) is False


def test_calculator_called_after_confirmation() -> None:
    """Profile complete AND confirmed -> calc."""
    from backend.app import _direct_tool_from_profile

    profile = SimpleNamespace(
        is_complete_for_calc=lambda: True,
        missing_fields=lambda: set(),
        confirmed_at=time.time(),
    )
    result = _direct_tool_from_profile(profile)
    assert result["action"] == "calc"
```

- [ ] **Step 2: Run tests, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_readback_state_machine.py -v`
Expected: FAIL — `_direct_tool_from_profile` does not exist.

- [ ] **Step 3: Add `_direct_tool_from_profile` helper**

In `rag_demo_system/backend/app.py`, add near `_run_session_agent`:

```python
def _direct_tool_from_profile(profile) -> dict:
    """Decide next action based on ClientProfile state.

    Returns one of:
      - {"action": "ask_next_missing", "missing": set, "next_field": str}
      - {"action": "readback"}
      - {"action": "calc"}
    """
    if not profile.is_complete_for_calc():
        missing = profile.missing_fields()
        priority = ["subject", "client_type", "cost", "currency",
                    "condition_new", "age_years", "term_months",
                    "prepaid", "type_schedule"]
        next_field = next((f for f in priority if f in missing), None)
        return {"action": "ask_next_missing", "missing": missing, "next_field": next_field}
    if profile.confirmed_at is None:
        return {"action": "readback"}
    return {"action": "calc"}
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_readback_state_machine.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Integrate into `_stream_voice_response`**

Replace the DirectTool decision block (currently around `app.py:943-1020`) with a new flow:

```python
# Apply SessionAgent profile_patches to the session's client_profile.
sa = _session_agent_out  # from _run_session_agent
changes = session.client_profile.apply_patches(sa.get("profile_patches") or {})
if changes:
    print(f"[Profile] patched: {changes}", flush=True)

# Semantic flags from SessionAgent.
if sa.get("is_stop_request"):
    # Handled by turn-taking plan; here just short-circuit.
    print("[SessionAgent] is_stop_request -> no response", flush=True)
    return  # Spec 4 extends this with listen_mode entry.

if sa.get("change_field") and session.client_profile.confirmed_at:
    # Post-confirm change: ask single-field re-confirmation, don't calc yet.
    session.client_profile.last_change_pending = sa["change_field"]
    # Apply the change now (merge) so readback speaks the new value.
    session.client_profile.apply_patches({sa["change_field"]: sa["change_value"]})
    # TTS single-field readback handled below.

if sa.get("is_confirmation") and session.client_profile.last_change_pending:
    session.client_profile.confirmed_at = time.time()
    session.client_profile.last_change_pending = None
elif sa.get("is_confirmation") and session.client_profile.is_complete_for_calc() and not session.client_profile.confirmed_at:
    # First-time confirmation on full readback.
    session.client_profile.confirmed_at = time.time()

decision = _direct_tool_from_profile(session.client_profile)

if decision["action"] == "ask_next_missing":
    # Emit prompt asking for next missing field.
    prompt_map = {
        "subject": "Что планируете в лизинг?",
        "client_type": "Вы физлицо, ИП или юрлицо?",
        "cost": "Какая стоимость?",
        "currency": "В какой валюте стоимость, в рублях или долларах?",
        "condition_new": "Новый предмет или с пробегом?",
        "age_years": "Какого года?",
        "term_months": "На какой срок хотите рассчитать?",
        "prepaid": "Какой аванс, в процентах или в сумме?",
        "type_schedule": "График аннуитетный или линейный?",
    }
    text = prompt_map.get(decision["next_field"], "Уточните, пожалуйста, параметры.")
    # Stream text via TTS pipeline (existing helper, whatever it's called):
    await _speak_text(websocket, session, text)
    return

if decision["action"] == "readback":
    p = session.client_profile
    subj = p.subject
    cost_str = f"{p.cost:.0f} {p.currency}"
    cond_str = "новый" if p.condition_new == 1 else f"б/у {p.age_years or ''} лет"
    prepaid_str = (f"аванс {p.prepaid_pct}%" if p.prepaid_pct is not None
                   else f"аванс {p.prepaid_amount} {p.currency}")
    sched_str = "аннуитетный" if p.type_schedule == "0" else "линейный"
    text = (
        f"Проверим параметры: {subj}, {cond_str}, {cost_str}, {p.client_type}, "
        f"{prepaid_str}, срок {p.term_months} месяцев, {sched_str} график. Всё верно?"
    )
    await _speak_text(websocket, session, text)
    return

# decision["action"] == "calc": fall through to calculator invocation.
# (Calculator plan consumes session.client_profile directly via the DirectTool path.)
```

If `_speak_text` doesn't exist, replace with the existing pattern used elsewhere in `_stream_voice_response` to stream a short response via the TTS pipeline (look at where the bot speaks generic fallback text).

- [ ] **Step 6: Run full test suite, verify no regressions**

Run: `cd rag_demo_system && python -m pytest tests/ -v -x`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/tests/test_readback_state_machine.py
git commit -m "feat(session): profile merge + readback state machine gates calculator"
```

---

### Task 5: System prompt rewrite (remove all defaults)

**Files:**
- Modify: `rag_demo_system/config/system_prompt_ru_v2.txt`

- [ ] **Step 1: Replace the `# Инструменты` section**

Open `rag_demo_system/config/system_prompt_ru_v2.txt`. Replace lines 133-174 (the `# Инструменты` section) with:

```
# Инструменты

ВАЖНО: Инструменты вызываются автоматически когда SessionAgent собрал все
необходимые параметры И клиент подтвердил расчёт. Никаких умолчаний. Если
клиент не назвал параметр, вы должны его спросить — не подставлять.

## Калькулятор лизинга

Для расчёта нужны все параметры:
- предмет лизинга (легковой автомобиль / грузовой автомобиль / спецтехника /
  оборудование / недвижимость / прочий транспорт)
- тип клиента (физическое лицо / ИП / юридическое лицо)
- состояние (новый / б/у, если б/у — возраст)
- стоимость и валюта (BYN или USD; EUR/RUB для физлиц не поддерживаются)
- срок лизинга в месяцах (от 12 до 84)
- размер аванса (в процентах или в рублях; от 0 до 40%)
- тип графика (аннуитетный или линейный)

Порядок сбора: если клиент назвал несколько параметров сразу, принимайте
всё и переходите к следующему недостающему. Если клиент говорит
сокращённо, задавайте один вопрос за раз.

ПЕРЕД первым расчётом: перечислите все собранные параметры одной фразой и
спросите "Всё верно?". Только после явного подтверждения вызывается
калькулятор.

ПОСЛЕ изменения параметра: подтвердите изменение ("меняю срок на 48,
верно?") и только после "да" — пересчёт.

Физические лица могут оформить лизинг ТОЛЬКО в BYN. Если клиент назвал
стоимость в долларах, озвучьте конвертацию по курсу 3 рубля за доллар и
продолжайте в рублях. Если в евро или российских рублях — скажите, что
сейчас поддерживаются только BYN и USD.

Грузовой транспорт, спецтехника, оборудование, недвижимость: только для
ИП и юрлиц. Если клиент физлицо — предложите легковой автомобиль или
прочий транспорт.

## Отправка СМС (send_sms)
- Только после успешного расчёта.
- СМС отправляется на номер звонящего. Не называйте номер вслух.
- Содержит только результат последнего расчёта.

## Передача специалисту (escalate_to_human)
- Когда клиент просит специалиста или вы не можете помочь.
- Убедитесь что есть номер и имя клиента.
```

- [ ] **Step 2: Verify no stray "30%" / "36 месяцев" / "по умолчанию" references**

Run: `grep -n "30%\|36 мес\|по умолчанию\|умолчанию" rag_demo_system/config/system_prompt_ru_v2.txt`
Expected: no matches (or only occurrences in the new text that describe the rule, not the default).

If any remain, delete or rephrase.

- [ ] **Step 3: Commit**

```bash
git add rag_demo_system/config/system_prompt_ru_v2.txt
git commit -m "refactor(prompt): remove all calculator defaults; describe collection protocol"
```

---

## Self-review

**Spec coverage:**
- `ClientProfile` dataclass → Task 1 ✓
- `ChatSession.client_profile` attachment → Task 2 ✓
- SessionAgent JSON schema → Task 3 ✓
- Profile merge logic → Task 1 (`apply_patches`) + Task 4 (integration) ✓
- Read-back gate → Task 4 ✓
- Change gate → Task 4 ✓
- System prompt rewrite → Task 5 ✓
- Integration / regression tests → Tasks 1, 2, 3, 4 ✓
- Transcript replay — deferred to end-of-round manual validation (test playbook)

**Placeholders:** none.

**Type consistency:**
- `apply_patches` returns a dict of changed fields (used in log line `[Profile] patched`).
- `_direct_tool_from_profile` returns `{"action": "ask_next_missing"|"readback"|"calc", ...}` — consumed at call site only.
- `profile_patches` keys map 1:1 to `ClientProfile` fields (`subject`, `cost`, `currency`, `client_type`, `condition_new`, `age_years`, `prepaid_pct`, `prepaid_amount`, `term_months`, `type_schedule`, `name`).
