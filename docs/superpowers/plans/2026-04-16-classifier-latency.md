# Classifier / SessionAgent Latency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move SessionAgent (formerly classifier) to a dedicated Qwen3-4B vLLM instance on port 8788 to eliminate scheduler contention with the main 35B model and enable prefix caching for the stable system prompt.

**Architecture:** Second vLLM process on 8788 serving Qwen3-4B-Instruct-FP8 (util 0.08). Backend reads new env vars `SESSIONAGENT_BASE_URL` / `SESSIONAGENT_MODEL`. SessionAgent helper extracted from `app.py` into a dedicated async function that uses the new URL. Main vLLM util drops 0.60 → 0.55 to create headroom. No functional behavior change; latency only.

**Tech Stack:** Python 3.12, vLLM 0.19.0, Qwen3-4B-Instruct-FP8, asyncio, pydantic-settings.

**Spec:** `docs/superpowers/specs/2026-04-16-classifier-latency-design.md`

---

### Task 1: Add SessionAgent env variables to settings

**Files:**
- Modify: `rag_demo_system/backend/config.py` (add fields)
- Modify: `rag_demo_system/.env.example` (add keys)

- [ ] **Step 1: Write the failing test**

Create `rag_demo_system/tests/test_session_agent_config.py`:

```python
"""SessionAgent config defaults and override precedence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_session_agent_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SESSIONAGENT_BASE_URL", raising=False)
    monkeypatch.delenv("SESSIONAGENT_MODEL", raising=False)
    from backend.config import Settings

    s = Settings()
    assert s.session_agent_base_url == "http://127.0.0.1:8788/v1"
    assert s.session_agent_model == "Qwen/Qwen3-4B-Instruct-FP8"


def test_session_agent_config_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SESSIONAGENT_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("SESSIONAGENT_MODEL", "Qwen/Qwen3-1.7B-Instruct")
    from backend.config import Settings

    s = Settings()
    assert s.session_agent_base_url == "http://127.0.0.1:9999/v1"
    assert s.session_agent_model == "Qwen/Qwen3-1.7B-Instruct"
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_session_agent_config.py -v`
Expected: FAIL — `Settings` has no `session_agent_base_url` attribute.

- [ ] **Step 3: Add fields to `Settings`**

Open `rag_demo_system/backend/config.py`. Find the class `Settings` or equivalent. Add two fields next to the existing `rag_llm_*` / `rag_llm_fast_*` fields (same pattern):

```python
session_agent_base_url: str = Field(
    default="http://127.0.0.1:8788/v1",
    description="OpenAI-compatible base URL for SessionAgent (classifier+profile extractor).",
)
session_agent_model: str = Field(
    default="Qwen/Qwen3-4B-Instruct-FP8",
    description="Model name served by the SessionAgent vLLM instance.",
)
```

Add matching env mapping in whatever BaseSettings `env_prefix` / field alias convention the file uses. Keep pattern identical to `rag_llm_base_url`.

- [ ] **Step 4: Run test, verify PASS**

Run: `cd rag_demo_system && python -m pytest tests/test_session_agent_config.py -v`
Expected: PASS.

- [ ] **Step 5: Update `.env.example`**

Append to `rag_demo_system/.env.example`:

```
# SessionAgent (classifier + profile extractor) — dedicated small model
SESSIONAGENT_BASE_URL=http://127.0.0.1:8788/v1
SESSIONAGENT_MODEL=Qwen/Qwen3-4B-Instruct-FP8
```

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/config.py rag_demo_system/.env.example rag_demo_system/tests/test_session_agent_config.py
git commit -m "feat(config): add SESSIONAGENT_BASE_URL and SESSIONAGENT_MODEL settings"
```

---

### Task 2: Extract SessionAgent helper

**Files:**
- Modify: `rag_demo_system/backend/app.py:703-798` (the current classifier block)

- [ ] **Step 1: Write the failing test**

Append to `rag_demo_system/tests/test_session_agent_config.py`:

```python
import asyncio
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock


def test_run_session_agent_uses_dedicated_url(monkeypatch) -> None:
    """SessionAgent helper must route to SESSIONAGENT_BASE_URL, not to main LLM URL."""
    monkeypatch.setenv("SESSIONAGENT_BASE_URL", "http://127.0.0.1:8788/v1")
    monkeypatch.setenv("SESSIONAGENT_MODEL", "Qwen/Qwen3-4B-Instruct-FP8")
    # re-import settings with fresh env
    import importlib
    import backend.config
    importlib.reload(backend.config)
    import backend.app
    importlib.reload(backend.app)

    captured = {}

    def fake_call(*, base_url, model, **kwargs):
        captured["base_url"] = base_url
        captured["model"] = model
        return SimpleNamespace(text='{"intent":"RAG"}')

    with patch("backend.llm.call_openai_compatible", side_effect=fake_call):
        fake_session = SimpleNamespace(tool_calls_this_turn=[])
        fake_chat_session = SimpleNamespace(transcript=[])
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            backend.app._run_session_agent(
                "Здравствуйте",
                fake_chat_session,
                fake_session,
                tool_schemas=[{}],
                session_id="test",
            )
        )
        loop.close()

    assert captured["base_url"] == "http://127.0.0.1:8788/v1"
    assert captured["model"] == "Qwen/Qwen3-4B-Instruct-FP8"
    assert result["intent"] == "RAG"
```

- [ ] **Step 2: Run test, verify FAIL**

Run: `cd rag_demo_system && python -m pytest tests/test_session_agent_config.py::test_run_session_agent_uses_dedicated_url -v`
Expected: FAIL — `backend.app._run_session_agent` does not exist.

- [ ] **Step 3: Extract classifier to `_run_session_agent`**

Open `rag_demo_system/backend/app.py`. Find the classifier block starting at `_t_classify_start = time.time()` (currently ~line 703). Extract everything from that line through the `print(f"[Classifier] result: ...")` line (currently ~798) into a new async function, placed just before `_stream_voice_response`:

```python
async def _run_session_agent(
    message: str,
    chat_session,
    session,
    tool_schemas: list,
    session_id: str,
) -> dict:
    """Call SessionAgent LLM to classify intent and extract fields.

    Routes to settings.session_agent_base_url (independent from main LLM) to
    avoid scheduler contention. Returns a dict with keys: intent, subject,
    cost, currency, client_type, prepaid, term, action, plus needs_tool boolean.
    """
    from .llm import call_openai_compatible

    # ... move the existing classifier body here, replacing:
    #   base_url=effective_base_url  ->  base_url=settings.session_agent_base_url
    #   model=effective_model        ->  model=settings.session_agent_model
    # Return a dict rather than mutating caller state.

    return {
        "intent": "TOOL" if needs_tool else "RAG",
        "needs_tool": needs_tool,
        "hints": _extracted_hints,
        "latency_ms": _t_classify_ms,
    }
```

Then at the original classifier call site, replace the inline block with:

```python
_session_agent_out = await _run_session_agent(
    message, chat_session, session,
    tool_schemas=tool_schemas, session_id=session_id,
)
needs_tool = _session_agent_out["needs_tool"]
_extracted_hints = _session_agent_out["hints"]
```

Preserve: fast-skip logic (currently lines 672-680) remains at the call site — it's a pre-filter that avoids invoking the helper entirely. Preserve: `has_sms_intent` override (currently line 792) remains at the call site.

- [ ] **Step 4: Run full test suite, verify no regression + new test passes**

Run: `cd rag_demo_system && python -m pytest tests/ -v -x`
Expected: All tests PASS including the new one.

Run the app locally (or on dev server) with `SESSIONAGENT_BASE_URL=http://127.0.0.1:8787/v1` pointing to the main LLM (fallback). Fire one test utterance. Verify logs show `[Classifier]` output format unchanged.

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/tests/test_session_agent_config.py
git commit -m "refactor(app): extract SessionAgent call to _run_session_agent helper"
```

---

### Task 3: Add second vLLM instance to deploy scripts

**Files:**
- Modify: `rag_demo_system/scripts/regenerate_env_and_restart.sh`
- Modify: `rag_demo_system/scripts/provision_server.sh`
- Modify: `rag_demo_system/scripts/restart_all.sh`

- [ ] **Step 1: Locate STACK_QWEN_CMD in regenerate_env_and_restart.sh**

Open `rag_demo_system/scripts/regenerate_env_and_restart.sh`. Find the line setting `STACK_QWEN_CMD=...` with `--gpu-memory-utilization 0.60`.

- [ ] **Step 2: Reduce main model utilization and add small model command**

Replace the value:

```
--gpu-memory-utilization 0.60
```

with:

```
--gpu-memory-utilization 0.55
```

Below the `STACK_QWEN_CMD=...` line, add a new env:

```bash
STACK_SESSIONAGENT_CMD="./.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-FP8 \
  --port 8788 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.08 \
  --enable-prefix-caching \
  --download-dir /workspace/models"
```

- [ ] **Step 3: Update supervisor / background launch in provision_server.sh**

Find the section in `provision_server.sh` that launches `STACK_QWEN_CMD`. Directly below, add a parallel launch for `STACK_SESSIONAGENT_CMD` using the same supervision pattern (same `nohup` / supervisor config). Add a corresponding health-check:

```bash
echo "Waiting for SessionAgent (Qwen3-4B) on :8788..."
for i in {1..60}; do
    if curl -sSf http://127.0.0.1:8788/health >/dev/null 2>&1; then
        echo "SessionAgent ready."
        break
    fi
    sleep 2
done
```

- [ ] **Step 4: Ensure restart_all.sh kills and restarts both vLLM instances**

Open `rag_demo_system/scripts/restart_all.sh`. Wherever it kills the main vLLM process (by port 8787), add symmetric kill for port 8788. Wherever it starts the main vLLM, add the sessionagent start.

Pattern to kill:

```bash
for port in 8787 8788; do
    pid=$(lsof -ti:$port || true)
    if [[ -n "$pid" ]]; then
        kill -9 "$pid" || true
    fi
done
```

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/scripts/regenerate_env_and_restart.sh rag_demo_system/scripts/provision_server.sh rag_demo_system/scripts/restart_all.sh
git commit -m "ops(vllm): add Qwen3-4B SessionAgent instance on :8788, main util 0.60->0.55"
```

---

### Task 4: Deploy validation

- [ ] **Step 1: Pull and restart on dev server**

On server: `cd /ephemeral/leasing/rag_demo_system && git pull && bash scripts/restart_all.sh`

- [ ] **Step 2: Verify both vLLM instances healthy**

```bash
curl -sS http://127.0.0.1:8787/health
curl -sS http://127.0.0.1:8788/health
nvidia-smi --query-gpu=memory.used --format=csv
```

Expected: both return OK. GPU memory used < 60000 MiB.

- [ ] **Step 3: Fire a SIP test call, inspect logs**

Tail `.state/backend.log`. Make a SIP call, say *"хочу рассчитать легковой автомобиль за 30 тысяч"*. Expected log lines:

```
[Classifier] result: intent=TOOL hints={'subject': 'Легковой автомобиль', 'cost': 30000, ...} (XXXms)
```

Compare XXXms against the baseline recorded before this change. Target: < 150ms p50.

No code changes here — observational step.

---

## Self-review

**Spec coverage:**
- GPU budget table → Task 3 ✓
- Env vars → Task 1 ✓
- Helper extraction → Task 2 ✓
- vLLM command with flags → Task 3 ✓
- Prefix cache explicit flag → Task 3 ✓ (`--enable-prefix-caching`)
- Latency measurement → Task 4 (observational) ✓
- Plan B (CPU) — not in this plan, documented in spec as fallback only

**Placeholders:** none.

**Type consistency:** `_run_session_agent` returns `{"intent", "needs_tool", "hints", "latency_ms"}` — consumed consistently at call site.
