# Tool Use System Design

**Date:** 2026-04-06
**Branch:** `feature/tool-use` (from `feature/voice-pipeline` at `1453dfe`)
**Safe revert:** `1453dfe`

## Overview

Add tool/function-calling capabilities to the voice pipeline. The LLM (Qwen3.5-35B via vLLM) calls tools mid-conversation using the native OpenAI-compatible `tools=[]` API. Tools execute server-side and results are injected back into the LLM context for natural spoken presentation.

### Tools in scope

| Tool | Priority | Phase | Purpose |
|------|----------|-------|---------|
| `calculator` | 1 | Now | Calculate leasing payment schedule via 1C API |
| `send_sms` | 1 | Now | Send full payment schedule to client via SMS |
| `escalate_to_human` | 2 | Later | Collect client data, summarize session, create AMO CRM lead |

### Out of scope

- 1C ERP integration (explicitly deferred by client)
- SIP telephony (separate phase after tool use)
- Email delivery (not requested)

---

## 1. Tool Framework Architecture

### New modules

```
rag_demo_system/backend/
  tools/
    __init__.py          # Tool registry
    base.py              # ToolDefinition base class
    calculator.py        # Payment calculator tool
    sms_sender.py        # SMS delivery tool
    escalation.py        # Human escalation tool (phase 2)
```

### ToolDefinition base class (`base.py`)

Abstract base class. Each tool implements:

```python
class ToolDefinition(ABC):
    @abstractmethod
    def schema(self) -> dict:
        """OpenAI-compatible function schema for tools=[] parameter."""

    @abstractmethod
    def defaults(self) -> dict:
        """Default parameter values."""

    def fill_defaults(self, params: dict) -> tuple[dict, list[str]]:
        """Merge user params with defaults. Returns (filled_params, defaulted_field_names)."""

    @abstractmethod
    async def execute(self, params: dict, session_context: dict) -> dict:
        """Execute the tool. Returns structured result."""

    @abstractmethod
    def format_voice_summary(self, result: dict) -> str:
        """Concise Russian text for LLM to base spoken response on."""

    def format_sms_body(self, result: dict) -> str | None:
        """Full text for SMS. Returns None if tool does not support SMS."""
```

### Tool Registry (`__init__.py`)

Simple dict-based registry. No auto-discovery. Three explicit imports.

```python
from .calculator import CalculatorTool
from .sms_sender import SmsSenderTool
from .escalation import EscalationTool

_TOOLS = {
    "calculator": CalculatorTool(),
    "send_sms": SmsSenderTool(),
    "escalate_to_human": EscalationTool(),
}

def get_tool_schemas() -> list[dict]:
    """All tool schemas for LLM tools=[] parameter."""

def get_tool(name: str) -> ToolDefinition:
    """Get tool instance by name."""
```

---

## 2. Tool Loop in the Streaming Pipeline

### Integration point

The tool loop lives inside `_stream_voice_response()` in `app.py`. It modifies the existing producer-consumer pattern.

### Flow

```
Build messages (+ tools=get_tool_schemas()) -> LLM call (streaming)
  |
  +-- LLM returns content tokens:
  |     Existing flow: SentenceDetector -> TTS queue -> Audio out
  |
  +-- LLM returns tool_calls:
        1. Parse tool name + arguments from response
        2. Orchestrator sends filler phrase to TTS (bypasses LLM)
        3. tool.fill_defaults(params) -> filled_params, defaulted_fields
        4. tool.execute(filled_params, session_context) -> result
        5. Append to message list:
           - assistant message with tool_calls
           - tool message with result + "Defaulted fields: ..."
        6. Call LLM again (same messages, streaming) with tools=get_tool_schemas()
        7. Second call streams content -> SentenceDetector -> TTS -> Audio
        8. If second call also returns tool_calls: loop (max 3 iterations)
```

### Filler phrases

Hardcoded per tool, randomly selected. Sent directly to TTS synthesis, bypassing LLM.

- **calculator**: "Секундочку, рассчитываю...", "Один момент, считаю для вас...", "Сейчас посчитаю..."
- **send_sms**: "Отправляю сообщение...", "Секунду, отправляю СМС..."
- **escalate_to_human**: "Передаю информацию специалисту..."

### Constraints

- **Max iterations**: 3 tool calls per user turn.
- **Tool timeout**: 10 seconds per execution. On timeout, return error result; LLM tells client gracefully.
- **Barge-in during tool execution**: If `session.interrupted` is set while tool executes, tool result is stored in session but spoken response is skipped. Client's new utterance takes priority.

---

## 3. LLM Integration

### Request modification (`llm.py`)

Add optional `tools` parameter to `iter_openai_compatible_stream_events()`:

```python
payload = {
    "model": model,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "stream": True,
    "messages": messages,
    "chat_template_kwargs": {"enable_thinking": False},
}
if tools:
    payload["tools"] = tools
```

### Response parsing

When streaming, vLLM returns tool calls as:
```json
{"choices": [{"delta": {"tool_calls": [{"function": {"name": "calculator", "arguments": "{...}"}}]}, "finish_reason": "tool_calls"}]}
```

The streaming parser accumulates tool_call deltas (arguments may be streamed token-by-token) and yields a complete tool_call object when `finish_reason` is `"tool_calls"`.

### Tool result message format

```json
{
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "Результат расчёта (параметры по умолчанию отмечены *):\n..."
}
```

---

## 4. Payment Calculator Tool

### API

Base URL: `https://personal.mikro-leasing.by/calculator/api/`
Auth: Bearer token (to be provided by client)

### Parameters

**Required** (tool rejects without): `subject`, `cost`

**Defaulted:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `client_type` | "Физическое лицо" | Most callers are individuals |
| `condition_new` | 1 | New item |
| `currency` | "BYN" | Local currency |
| `prepaid` | 30 | Advance percentage |
| `term` | 36 | Months |
| `type_schedule` | "0" | Annuity (equal payments) |
| `age` | 0 | New item, no age |

When `condition_new=0` (used) and `age` is not provided, the tool returns an error asking the LLM to collect the age.

### API call

Single call to `/1.0/calculate/` with all parameters. Returns full payment schedule.

Fallback: if `/1.0/calculate/` returns empty/error, do NOT automatically call `/1.0/terms/` in v1. Return the error to the LLM; it tells the client to adjust parameters.

### Voice summary format (for LLM context)

```
Результат расчёта (параметры по умолчанию отмечены *):
- Тип клиента: Физическое лицо *
- Предмет: Легковой автомобиль
- Состояние: Новый *
- Валюта: BYN *
- Стоимость: 30 000
- Аванс: 30% * -> 9 000 BYN
- Срок: 36 мес. *
- График: аннуитет (равные платежи) *

Аванс (платёж 0): 9 000 BYN
Ежемесячный платёж: ~850 BYN (36 платежей)
Выкупной платёж: 100 BYN
Общая сумма: ~39 700 BYN
```

The LLM reads this and formulates a natural spoken response, including which fields were assumed.

### SMS format (full schedule)

```
Микро Лизинг: график платежей
Легковой автомобиль, 30 000 BYN
Тип: Физическое лицо, аннуитет

Аванс: 9 000,00
1: 850,00
2: 850,00
...
36: 850,00
Выкуп: 100,00

Итого: ~39 700,00 BYN
+375 17 322 77 00
```

Every payment row from the API included. Multi-part SMS (3-4 segments) is acceptable since client explicitly requested it.

---

## 5. SMS Sender Tool

### Purpose

Sends SMS via sms-assistent.by API. Never called independently; always follows a calculator result when client agrees.

### Schema

```json
{
    "name": "send_sms",
    "parameters": {
        "phone": "string, required, international format",
        "message": "string, required, full text body"
    }
}
```

### Phone number source

- SIP mode (future): from transport layer automatically
- Returning client: from stored profile
- New client (browser): model asks at SMS send time, not before

### Error handling

- API error/timeout: return error, LLM tells client it could not send
- Invalid phone format: validate before API call, return error, LLM asks to repeat

---

## 6. Client Session Persistence

### Data model

Client profiles stored as JSON files, keyed by phone hash.

```
.state/clients/
  {phone_hash}.json
```

### Profile structure

```json
{
    "phone_hash": "sha256:...",
    "phone_last4": "7700",
    "pin_hash": "bcrypt:...",
    "name": "Иван",
    "client_type": "Физическое лицо",
    "created_at": "2026-04-06T14:00:00Z",
    "last_session_at": "2026-04-06T14:35:00Z",
    "sessions": [
        {
            "session_id": "abc123",
            "timestamp": "2026-04-06T14:00:00Z",
            "summary": "AI-generated 2-3 sentence summary",
            "topics": ["легковой автомобиль", "расчёт платежей"],
            "calculations": [
                {
                    "subject": "Легковой автомобиль",
                    "cost": 25000,
                    "currency": "USD",
                    "term": 36,
                    "monthly_payment": 580.00
                }
            ],
            "pending_actions": ["клиент обещал перезвонить"]
        }
    ]
}
```

### Summary generation

Runs after session ends (no latency impact on conversation). LLM call with summarization prompt extracts: name, topics, calculations, pending actions, 2-3 sentence summary. Only summaries stored, never raw transcripts.

### Returning client flow

1. Phone number arrives (SIP) or client provides it
2. Lookup `{phone_hash}.json`
3. If found: ask for 4-digit PIN (max 3 attempts, then new-client flow)
4. PIN verified: inject into LLM context as memory block:

```
[Информация о клиенте]
Имя: Иван
Последний звонок: 6 апреля 2026
Краткое содержание: ...
Предыдущие расчёты: ...
Незавершённые вопросы: ...
```

5. Model opens naturally: "Иван, рада снова слышать вас! В прошлый раз мы обсуждали..."

### PIN management

- Offered at session end (optional). Client chooses whether to be remembered.
- 4 digits, stored as bcrypt hash.
- No recovery in v1. Forgotten PIN = start fresh.
- "Forget me" command deletes profile.

### Privacy

- No raw transcripts stored, only AI summaries.
- Phone stored as hash (lookup) + last 4 digits (display only).
- PIN protects access.

---

## 7. Escalation Tool (Phase 2, Design Only)

### Trigger conditions

- Client explicitly asks for a specialist/manager
- Model exhausted its ability to help
- Objection handling leads to "transfer to specialist"

### Data collection

1. Phone number (from SIP, profile, or ask + confirm)
2. Name (only if not already known)
3. No other mandatory fields

### Execution sequence

1. Generate session summary (LLM call with full transcript)
2. Create lead in AMO CRM (POST with structured data)
3. Return confirmation to LLM

### Post-escalation

Session enters wind-down. Summary generated and stored in client profile.

### Open items (waiting for client)

- AMO CRM field mapping and pipeline configuration
- Callback SLA (what timeframe to promise)
- Duplicate lead handling strategy

---

## 8. System Prompt Changes

### Replace integration placeholder (lines 123-128)

Replace the current `# Integration Points (Note)` section with tool usage instructions:

```
# Инструменты

Вам доступны инструменты для выполнения действий во время разговора. Система сообщит вам
какие инструменты доступны. Используйте их когда это уместно.

## Калькулятор (calculator)
- Используйте когда клиент хочет узнать суммы платежей, рассчитать график, или спрашивает
  "сколько будет стоить".
- Обязательные данные от клиента: предмет лизинга и стоимость. Остальное можно рассчитать
  с параметрами по умолчанию.
- После получения результата ОБЯЗАТЕЛЬНО назовите какие параметры были выбраны по умолчанию
  и предложите пересчитать с другими значениями.
- После озвучивания результата предложите отправить полный график платежей по СМС.

## Отправка СМС (send_sms)
- Используйте ТОЛЬКО после того, как клиент явно согласился получить СМС.
- Никогда не отправляйте СМС без согласия клиента.
- Если номер телефона неизвестен, спросите его перед отправкой.

## Передача специалисту (escalate_to_human)
- Используйте когда клиент просит связать со специалистом или когда вы не можете помочь.
- Перед вызовом убедитесь, что у вас есть номер телефона клиента.
- Спросите имя, если оно ещё неизвестно.
```

### Fix rates guardrail (line 29)

Before: "Не называть процентные ставки и переплаты. Условия рассчитываются индивидуально менеджером."
After: "Не придумывать процентные ставки и переплаты. Если калькулятор вернул результат, можно озвучить суммы платежей из расчёта. Не называть ставки если они не были явно возвращены калькулятором."

### Fix escalation trigger (line 64)

Before: "Вопрос требует индивидуального расчета, который вы не можете сделать"
After: "Вопрос требует индивидуального расчёта, выходящего за возможности калькулятора (нестандартные условия, акции, специальные программы)"

### Update consultation flow (lines 85-93)

Add: "На шаге расчёта используйте инструмент calculator. После расчёта предложите отправить график по СМС."

### KB updates

Pass through KB to update entries that say "contact a specialist for calculations" to reflect that the system can now calculate. Specific entries to review:
- "данные для расчета" intent
- Any "менеджер рассчитает" / "индивидуальный расчёт" phrasing
- Payment schedule references that redirect to humans

---

## 9. Development Approach

### Local development (all tool framework work)

- Tool modules, base class, registry: pure Python, no GPU
- Calculator API integration: test with real API (if credentials available) or mock responses
- SMS integration: mock until credentials available
- LLM tool-calling response parsing: mock vLLM responses locally
- Streaming pipeline changes: test with mock LLM that returns tool_calls
- System prompt and KB changes: text editing, no server needed
- Client persistence: pure Python, file I/O
- Unit tests for all tool modules

### Server integration testing (final phase)

Spin up GPU server only for end-to-end voice testing: STT -> LLM with real tool calling -> tool execution -> TTS. This is the last step after everything else works locally.

---

## 10. Dependencies and Risks

### External dependencies

| Dependency | Status | Blocker for |
|------------|--------|-------------|
| Calculator API bearer token | Waiting for client | Calculator tool testing with real API |
| SMS API credentials | Waiting for client | SMS tool |
| AMO CRM contract | Waiting for client | Escalation tool (phase 2) |
| vLLM tool calling with Qwen3.5 FP8 | Needs verification | LLM integration |

### Risks

1. **vLLM + Qwen3.5 tool calling**: Native function calling may behave unexpectedly with FP8 quantization. Mitigation: test early on server; if broken, fall back to prompt-driven approach (Approach B from design discussion).
2. **Calculator API instability**: External service, no SLA known. Mitigation: 10-second timeout, graceful error handling, LLM tells client to try again.
3. **SMS cost**: Multi-part SMS at scale. Mitigation: only sent on explicit client request, never automatic.
4. **PIN UX in voice**: Saying a 4-digit PIN aloud may feel awkward. Mitigation: offer as optional; evaluate after user testing.
