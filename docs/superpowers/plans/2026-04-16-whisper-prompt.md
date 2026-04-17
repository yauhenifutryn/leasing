# Whisper Prompt Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Whisper's `initial_prompt` with a tightened 680-char version that fits in the 224-token cap and biases transcription toward "Ксения", graph types, and leasing-domain Russian vocabulary.

**Architecture:** Single string replacement in `services/whisper_server.py` + a pytest regression test that validates the token budget and key-term inclusion.

**Tech Stack:** Python 3.12, faster-whisper, pytest.

**Spec:** `docs/superpowers/specs/2026-04-16-whisper-prompt-design.md`

---

### Task 1: Token budget regression test

**Files:**
- Create: `rag_demo_system/tests/test_whisper_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
"""Regression tests for Whisper initial_prompt: token budget and key-term inclusion."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.whisper_server import _DEFAULT_INITIAL_PROMPT  # noqa: E402


def test_prompt_fits_224_token_budget() -> None:
    """Whisper's initial_prompt is capped at 224 tokens. Anything above silently truncates."""
    from faster_whisper.tokenizer import Tokenizer

    class _StubModel:
        is_multilingual = True
        num_languages = 99

    tokenizer = Tokenizer(
        tokenizer=None,
        multilingual=True,
        task="transcribe",
        language="ru",
    )
    tokens = tokenizer.encode(_DEFAULT_INITIAL_PROMPT)
    assert len(tokens) < 224, f"Prompt is {len(tokens)} tokens, cap is 224"


def test_prompt_contains_critical_vocabulary() -> None:
    """Key terms missing before this change; must all appear at least once now."""
    required = [
        "Ксения",
        "линейный",
        "аннуитет",
        "дифференцированный",
        "нагрузка",
        "переплата",
        "физик",
        "юрик",
        "ипэшник",
        "лизингополучатель",
        "выкупной",
    ]
    for term in required:
        assert term in _DEFAULT_INITIAL_PROMPT, f"Missing required term: {term}"


def test_prompt_places_bot_name_near_end() -> None:
    """Whisper keeps only the last 224 tokens; Ксения must be in the final half to be guaranteed."""
    mid = len(_DEFAULT_INITIAL_PROMPT) // 2
    assert "Ксения" in _DEFAULT_INITIAL_PROMPT[mid:], "Ксения must appear in second half of prompt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rag_demo_system && python -m pytest tests/test_whisper_prompt.py -v`
Expected: All tests fail — `Ксения` is absent from current prompt; `линейный` missing; token count likely exceeds 224 (current prompt ~275 tokens).

If the tokenizer import fails (e.g. faster-whisper not installed in current venv), the test should be skipped with `pytest.importorskip("faster_whisper.tokenizer")` — add at top of test file:

```python
import pytest
pytest.importorskip("faster_whisper")
```

- [ ] **Step 3: Commit the failing test**

```bash
git add rag_demo_system/tests/test_whisper_prompt.py
git commit -m "test(whisper): add regression tests for initial_prompt token budget and key vocab"
```

---

### Task 2: Replace the initial prompt

**Files:**
- Modify: `rag_demo_system/services/whisper_server.py:14-35`

- [ ] **Step 1: Replace `_DEFAULT_INITIAL_PROMPT`**

Open `rag_demo_system/services/whisper_server.py`. Replace lines 14-35 (the comment block above `_DEFAULT_INITIAL_PROMPT` plus the whole string) with:

```python
# Domain vocabulary for Whisper: biases transcription toward the bot name,
# Belarusian leasing vocabulary, car brand aliases and graph types.
# Whisper initial_prompt is capped at 224 tokens; high-ROI terms placed at
# the end (guaranteed to survive truncation). See test_whisper_prompt.py.
_DEFAULT_INITIAL_PROMPT = (
    "Микро Лизинг, лизинг в Беларуси. Помощница Ксения. "
    "Города: Минск, Гомель, Брест, Витебск, Гродно, Могилёв. "
    "Марки: Volkswagen Фольксваген, Toyota Тойота, BMW бэха, "
    "Mercedes мерс, Audi аудюха, Hyundai Хёндай, Kia Киа, "
    "Skoda Шкода, Lada Лада ВАЗ, ГАЗ ГАЗель, МАЗ, "
    "Geely Джили, Chery Чери, Haval Хавал, BYD. "
    "Документы: паспорт, водительские права, УНП, ИНН, VIN, "
    "НДС, КАСКО, ОСАГО, новый, б/у. "
    "Валюты: белорусский рубль, доллар, евро, российский рубль. "
    "Предметы: легковой автомобиль, грузовой автомобиль, "
    "спецтехника, оборудование, недвижимость, прочий транспорт, "
    "тягач, полуприцеп, погрузчик, автобус. "
    "Клиенты: физическое лицо, физлицо, физик, ИП, ипэшник, "
    "индивидуальный предприниматель, юридическое лицо, юрлицо, юрик. "
    "Термины: аванс, срок лизинга, ежемесячный платёж, "
    "выкупной платёж, график платежей, нагрузка, переплата, "
    "удорожание, общая сумма, итого, лизингодатель, лизингополучатель. "
    "Графики: аннуитетный, аннуитет, линейный, дифференцированный. "
    "Голосовая помощница Ксения, Ксения."
)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd rag_demo_system && python -m pytest tests/test_whisper_prompt.py -v`
Expected: All three tests PASS.

If `test_prompt_fits_224_token_budget` still fails (exceeds budget), shorten the prompt by removing the least critical car brand (likely `Skoda Шкода` first). Re-run until passing.

- [ ] **Step 3: Commit the prompt change**

```bash
git add rag_demo_system/services/whisper_server.py
git commit -m "feat(whisper): tighten initial_prompt for Ксения, graph types, leasing slang"
```

---

### Task 3: Manual smoke validation

- [ ] **Step 1: Deploy and run a voice call**

After server redeploy (end-of-round deploy command, not per-task), call in and say:
- "Ксения, я физлицо" — transcript should contain "Ксения" (not "Сеня" / "Синяя").
- "Линейный график на 60 месяцев" — transcript contains "линейный".
- "Какая нагрузка по платежам" — transcript contains "нагрузка".

- [ ] **Step 2: Capture results in call audit log**

Tail `.state/backend.log` for 5-10 subsequent calls. Record recognition rate for "Ксения" in vocative position. Target: ≥ 80% (from estimated current ~20%).

No code changes at this task — it is an observational checkpoint.

---

## Self-review

**Spec coverage:**
- New prompt string — Task 2 ✓
- Token budget guard — Task 1 ✓
- Key-term presence — Task 1 ✓
- V2 queue note — already in spec, no code ✓

**Placeholders:** none.

**Type consistency:** single string constant; nothing to cross-reference.
