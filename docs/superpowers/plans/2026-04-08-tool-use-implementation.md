# Tool Use System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tool/function-calling to the voice pipeline so the LLM can call a payment calculator API and send SMS during conversations.

**Architecture:** Python tool modules with a base class and registry. Native OpenAI function calling via vLLM `tools=[]`. Tool loop inside the existing streaming pipeline with orchestrator-injected filler phrases. All local-first development; server integration last.

**Tech Stack:** Python 3.11+, FastAPI, httpx (async HTTP), pytest. External: calculator API (REST + Bearer), sms-assistent.by (REST + login/password).

**Spec:** `docs/superpowers/specs/2026-04-06-tool-use-system-design.md`

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `backend/tools/__init__.py` | Tool registry: `get_tool_schemas()`, `get_tool(name)` |
| `backend/tools/base.py` | `ToolDefinition` abstract base class |
| `backend/tools/calculator.py` | `CalculatorTool`: calls calculator API, formats results |
| `backend/tools/sms_sender.py` | `SmsSenderTool`: sends SMS via sms-assistent.by |
| `backend/tools/filler.py` | Filler phrases per tool, random selection |
| `tests/test_tool_base.py` | Tests for base class `fill_defaults` logic |
| `tests/test_calculator.py` | Tests for calculator tool (mock HTTP) |
| `tests/test_sms_sender.py` | Tests for SMS tool (mock HTTP) |
| `tests/test_tool_registry.py` | Tests for registry functions |
| `tests/test_tool_loop.py` | Tests for streaming tool call parsing |

### Modified files

| File | What changes |
|------|-------------|
| `backend/llm.py` | Add `tools` and `messages` params to streaming functions |
| `backend/llm_stream.py` | Add tool_call delta accumulation to event parser |
| `backend/app.py` | Tool loop in `_stream_voice_response()`, filler injection |
| `backend/voice_session.py` | Add `tool_calls_this_turn` field |
| `backend/settings.py` | Add `ToolsConfig` dataclass for env vars |
| `config/app.yaml` | Add `tools:` section |
| `config/system_prompt_ru_v2.txt` | Replace integration placeholder, fix guardrails |
| `.env.example` | Already done (calculator + SMS placeholders) |

---

### Task 1: Tool base class and registry

**Files:**
- Create: `rag_demo_system/backend/tools/__init__.py`
- Create: `rag_demo_system/backend/tools/base.py`
- Test: `rag_demo_system/tests/test_tool_base.py`

- [ ] **Step 1: Write the failing test for fill_defaults**

```python
# tests/test_tool_base.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.base import ToolDefinition


class DummyTool(ToolDefinition):
    def schema(self) -> dict:
        return {"type": "function", "function": {"name": "dummy", "parameters": {}}}

    def defaults(self) -> dict:
        return {"color": "red", "size": 10}

    def execute(self, params: dict, session_context: dict) -> dict:
        return {"ok": True}

    def format_voice_summary(self, result: dict) -> str:
        return "done"


def test_fill_defaults_merges_missing():
    tool = DummyTool()
    filled, defaulted = tool.fill_defaults({"color": "blue"})
    assert filled == {"color": "blue", "size": 10}
    assert defaulted == ["size"]


def test_fill_defaults_no_defaults_needed():
    tool = DummyTool()
    filled, defaulted = tool.fill_defaults({"color": "blue", "size": 5})
    assert filled == {"color": "blue", "size": 5}
    assert defaulted == []


def test_fill_defaults_all_defaulted():
    tool = DummyTool()
    filled, defaulted = tool.fill_defaults({})
    assert filled == {"color": "red", "size": 10}
    assert sorted(defaulted) == ["color", "size"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd rag_demo_system && python -m pytest tests/test_tool_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.tools'`

- [ ] **Step 3: Write base.py**

```python
# backend/tools/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolDefinition(ABC):
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """OpenAI-compatible function schema for the tools=[] parameter."""

    @abstractmethod
    def defaults(self) -> dict[str, Any]:
        """Default parameter values. Keys must match schema property names."""

    def fill_defaults(self, params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Merge user-provided params with defaults.

        Returns (filled_params, list_of_defaulted_field_names).
        """
        result = dict(params)
        defaulted: list[str] = []
        for key, value in self.defaults().items():
            if key not in result:
                result[key] = value
                defaulted.append(key)
        return result, defaulted

    @abstractmethod
    def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool. Returns structured result dict."""

    @abstractmethod
    def format_voice_summary(self, result: dict[str, Any]) -> str:
        """Concise Russian text for LLM to base spoken response on."""

    def format_sms_body(self, result: dict[str, Any]) -> str | None:
        """Full text for SMS delivery. None if tool does not support SMS."""
        return None
```

- [ ] **Step 4: Create empty __init__.py**

```python
# backend/tools/__init__.py
from __future__ import annotations

from typing import Any

from .base import ToolDefinition


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return all tool schemas for the LLM tools=[] parameter."""
    return [tool.schema() for tool in _TOOLS.values()]


def get_tool(name: str) -> ToolDefinition:
    """Get a tool instance by name. Raises KeyError if not found."""
    return _TOOLS[name]


def get_all_tools() -> dict[str, ToolDefinition]:
    """Return the full tool registry."""
    return dict(_TOOLS)


# Registry populated as tools are implemented.
# Import lines added in later tasks.
_TOOLS: dict[str, ToolDefinition] = {}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd rag_demo_system && python -m pytest tests/test_tool_base.py -v`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/tools/__init__.py rag_demo_system/backend/tools/base.py rag_demo_system/tests/test_tool_base.py
git commit -m "feat: add tool base class and registry skeleton"
```

---

### Task 2: Calculator tool

**Files:**
- Create: `rag_demo_system/backend/tools/calculator.py`
- Modify: `rag_demo_system/backend/tools/__init__.py`
- Modify: `rag_demo_system/backend/settings.py`
- Modify: `rag_demo_system/config/app.yaml`
- Test: `rag_demo_system/tests/test_calculator.py`

- [ ] **Step 1: Add ToolsConfig to settings.py**

Add after `QueryRewriteConfig` (line 83 of `settings.py`):

```python
@dataclass
class ToolsConfig:
    calculator_api_base_url: str
    calculator_api_token: str
    sms_api_login: str
    sms_api_password: str
    sms_sender_name: str
    crm_webhook_url: str
    crm_webhook_token: str
```

Add `tools: ToolsConfig` to the `Settings` dataclass (after `voice`).

Add to `load_settings()` after voice config:

```python
tools_cfg = payload.get("tools", {})
# ... in the Settings() constructor:
tools=ToolsConfig(
    calculator_api_base_url=os.getenv("CALCULATOR_API_BASE_URL", tools_cfg.get("calculator_api_base_url", "")),
    calculator_api_token=os.getenv("CALCULATOR_API_TOKEN", tools_cfg.get("calculator_api_token", "")),
    sms_api_login=os.getenv("SMS_API_LOGIN", tools_cfg.get("sms_api_login", "")),
    sms_api_password=os.getenv("SMS_API_PASSWORD", tools_cfg.get("sms_api_password", "")),
    sms_sender_name=os.getenv("SMS_SENDER_NAME", tools_cfg.get("sms_sender_name", "MikroLizing")),
    crm_webhook_url=os.getenv("CRM_WEBHOOK_URL", tools_cfg.get("crm_webhook_url", "")),
    crm_webhook_token=os.getenv("CRM_WEBHOOK_TOKEN", tools_cfg.get("crm_webhook_token", "")),
),
```

- [ ] **Step 2: Add tools section to app.yaml**

Append to `config/app.yaml`:

```yaml
tools:
  calculator_api_base_url: $CALCULATOR_API_BASE_URL
  calculator_api_token: $CALCULATOR_API_TOKEN
  sms_api_login: $SMS_API_LOGIN
  sms_api_password: $SMS_API_PASSWORD
  sms_sender_name: $SMS_SENDER_NAME
  crm_webhook_url: $CRM_WEBHOOK_URL
  crm_webhook_token: $CRM_WEBHOOK_TOKEN
```

- [ ] **Step 3: Write the failing test for calculator tool**

```python
# tests/test_calculator.py
from pathlib import Path
import sys
import json
from unittest.mock import patch, MagicMock
import asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.calculator import CalculatorTool

SAMPLE_RESPONSE = {
    "0": {
        "URL": "https://mikro-leasing.by/graphic/?57030126",
        "id": "57030126",
        "increase_factor": 0.12914956,
        "increase_percent": 13.0,
        "number": 0,
        "sum": 9000.0,
    },
    "1": {"number": 1, "sum": 897.87},
    "2": {"number": 2, "sum": 897.87},
    "3": {"number": 3, "sum": 897.87},
    "999": {"number": 999, "sum": 300.0},
}


def test_schema_has_required_fields():
    tool = CalculatorTool(base_url="http://test", token="tok")
    schema = tool.schema()
    func = schema["function"]
    assert func["name"] == "calculator"
    props = func["parameters"]["properties"]
    assert "subject" in props
    assert "cost" in props
    assert func["parameters"]["required"] == ["subject", "cost"]


def test_defaults():
    tool = CalculatorTool(base_url="http://test", token="tok")
    d = tool.defaults()
    assert d["client_type"] == "Физическое лицо"
    assert d["currency"] == "BYN"
    assert d["prepaid"] == 30
    assert d["term"] == 36
    assert d["type_schedule"] == "0"


def test_fill_defaults_marks_defaulted():
    tool = CalculatorTool(base_url="http://test", token="tok")
    filled, defaulted = tool.fill_defaults({"subject": "Легковой автомобиль", "cost": 30000})
    assert filled["currency"] == "BYN"
    assert "currency" in defaulted
    assert "subject" not in defaulted
    assert "cost" not in defaulted


def test_execute_calls_api():
    tool = CalculatorTool(base_url="http://test", token="tok")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SAMPLE_RESPONSE

    with patch("backend.tools.calculator.httpx.get", return_value=mock_resp) as mock_get:
        result = tool.execute(
            {"subject": "Легковой автомобиль", "cost": 30000, "client_type": "Физическое лицо",
             "condition_new": 1, "currency": "BYN", "prepaid": 30, "term": 36, "type_schedule": "0"},
            {},
        ))
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "Authorization" in call_args.kwargs.get("headers", call_args[1].get("headers", {}))

    assert result["ok"] is True
    assert result["url"] == "https://mikro-leasing.by/graphic/?57030126"
    assert result["calculation_id"] == "57030126"
    assert result["advance_sum"] == 9000.0
    assert result["buyout_sum"] == 300.0
    assert result["increase_percent"] == 13.0
    assert len(result["payments"]) == 3  # 3 monthly payments in sample


def test_execute_handles_404():
    tool = CalculatorTool(base_url="http://test", token="tok")
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "Not Found"

    with patch("backend.tools.calculator.httpx.get", return_value=mock_resp):
        result = tool.execute(
            {"subject": "Легковой автомобиль", "cost": 30000, "client_type": "Физическое лицо",
             "condition_new": 1, "currency": "BYN", "prepaid": 30, "term": 36, "type_schedule": "0"},
            {},
        ))

    assert result["ok"] is False
    assert "error" in result


def test_voice_summary_contains_key_numbers():
    tool = CalculatorTool(base_url="http://test", token="tok")
    result = {
        "ok": True,
        "params": {"subject": "Легковой автомобиль", "cost": 30000, "currency": "BYN",
                    "client_type": "Физическое лицо", "condition_new": 1, "prepaid": 30,
                    "term": 36, "type_schedule": "0"},
        "defaulted": ["currency", "prepaid", "term"],
        "advance_sum": 9000.0,
        "monthly_payment_avg": 897.87,
        "monthly_payment_first": 897.87,
        "monthly_payment_last": 897.87,
        "buyout_sum": 300.0,
        "total_sum": 41623.46,
        "increase_percent": 13.0,
        "num_payments": 36,
        "url": "https://mikro-leasing.by/graphic/?57030126",
        "calculation_id": "57030126",
        "payments": [],
    }
    summary = tool.format_voice_summary(result)
    assert "9 000" in summary or "9000" in summary
    assert "897" in summary
    assert "300" in summary
    assert "*" in summary  # defaulted fields marked


def test_format_sms_body_contains_link():
    tool = CalculatorTool(base_url="http://test", token="tok")
    result = {
        "ok": True,
        "params": {"subject": "Легковой автомобиль", "cost": 30000, "currency": "BYN",
                    "term": 36, "prepaid": 30},
        "url": "https://mikro-leasing.by/graphic/?57030126",
        "advance_sum": 9000.0,
        "monthly_payment_avg": 897.87,
        "num_payments": 36,
        "increase_percent": 13.0,
        "total_sum": 41623.46,
        "buyout_sum": 300.0,
    }
    body = tool.format_sms_body(result)
    assert body is not None
    assert "https://mikro-leasing.by/graphic/?57030126" in body
    assert "Микро Лизинг" in body
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.tools.calculator'`

- [ ] **Step 5: Implement calculator.py**

```python
# backend/tools/calculator.py
from __future__ import annotations

from typing import Any

import httpx

from .base import ToolDefinition

# Map human-readable names to labels used in voice summary
_SCHEDULE_TYPES = {"0": "аннуитет (равные платежи)", "1": "убывающий"}
_CONDITION = {1: "Новый", 0: "Б/У"}


class CalculatorTool(ToolDefinition):
    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": (
                    "Рассчитать график лизинговых платежей. Вызывайте когда клиент хочет "
                    "узнать суммы платежей, ежемесячный платёж, или просит рассчитать лизинг."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": (
                                "Предмет лизинга. Один из: Легковой автомобиль, Грузовой автомобиль, "
                                "Прочий транспорт, Оборудование, Спецтехника, Недвижимость"
                            ),
                        },
                        "cost": {
                            "type": "number",
                            "description": "Стоимость предмета лизинга",
                        },
                        "client_type": {
                            "type": "string",
                            "description": "Тип клиента: Физическое лицо или Юридическое лицо. По умолчанию Физическое лицо.",
                        },
                        "condition_new": {
                            "type": "integer",
                            "description": "Новый (1) или б/у (0). По умолчанию 1 (новый).",
                        },
                        "age": {
                            "type": "integer",
                            "description": "Возраст предмета лизинга в годах. Обязателен если б/у (condition_new=0).",
                        },
                        "currency": {
                            "type": "string",
                            "description": "Валюта: BYN, USD, EUR, RUB. По умолчанию BYN.",
                        },
                        "prepaid": {
                            "type": "number",
                            "description": "Аванс в процентах. По умолчанию 30.",
                        },
                        "term": {
                            "type": "integer",
                            "description": "Срок лизинга в месяцах. По умолчанию 36.",
                        },
                        "type_schedule": {
                            "type": "string",
                            "description": "Тип графика: 0 = аннуитет (равные платежи), 1 = убывающий. По умолчанию 0.",
                        },
                    },
                    "required": ["subject", "cost"],
                },
            },
        }

    def defaults(self) -> dict[str, Any]:
        return {
            "client_type": "Физическое лицо",
            "condition_new": 1,
            "currency": "BYN",
            "prepaid": 30,
            "term": 36,
            "type_schedule": "0",
        }

    def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
        # Validate: used item must have age
        if params.get("condition_new") == 0 and "age" not in params:
            return {
                "ok": False,
                "error": "Для б/у предмета необходимо указать возраст (age). Спросите у клиента год выпуска.",
            }

        url = f"{self._base_url}/1.0/calculate/"
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=10.0)
        except httpx.TimeoutException:
            return {"ok": False, "error": "Сервис расчёта временно недоступен. Попробуйте позже."}
        except httpx.RequestError as exc:
            return {"ok": False, "error": f"Ошибка подключения к сервису расчёта: {exc}"}

        if resp.status_code == 404:
            return {
                "ok": False,
                "error": "Подходящих условий не найдено для указанных параметров. Предложите клиенту изменить аванс, срок или стоимость.",
            }
        if resp.status_code != 200:
            return {"ok": False, "error": f"Ошибка сервиса расчёта (HTTP {resp.status_code})."}

        data = resp.json()
        return self._parse_response(data, params)

    def _parse_response(self, data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        payment_0 = data.get("0", {})
        payment_999 = data.get("999", {})

        # Collect monthly payments (keys 1..N, excluding 0 and 999)
        monthly = []
        for key in sorted(data.keys(), key=lambda k: int(k) if k.isdigit() else 9999):
            if key in ("0", "999"):
                continue
            entry = data[key]
            if isinstance(entry, dict) and "sum" in entry:
                monthly.append(entry["sum"])

        total = sum(v["sum"] for v in data.values() if isinstance(v, dict) and "sum" in v)

        return {
            "ok": True,
            "params": params,
            "url": payment_0.get("URL", ""),
            "calculation_id": str(payment_0.get("id", "")),
            "advance_sum": payment_0.get("sum", 0),
            "increase_factor": payment_0.get("increase_factor", 0),
            "increase_percent": payment_0.get("increase_percent", 0),
            "buyout_sum": payment_999.get("sum", 0),
            "payments": monthly,
            "num_payments": len(monthly),
            "monthly_payment_first": monthly[0] if monthly else 0,
            "monthly_payment_last": monthly[-1] if monthly else 0,
            "monthly_payment_avg": round(sum(monthly) / len(monthly), 2) if monthly else 0,
            "total_sum": round(total, 2),
        }

    def format_voice_summary(self, result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return result.get("error", "Ошибка расчёта.")

        p = result["params"]
        defaulted = result.get("defaulted", [])

        def _mark(key: str, val: str) -> str:
            return f"{val} *" if key in defaulted else val

        lines = [
            "Результат расчёта (параметры по умолчанию отмечены *):",
            f"- Тип клиента: {_mark('client_type', p.get('client_type', ''))}",
            f"- Предмет: {_mark('subject', p.get('subject', ''))}",
            f"- Состояние: {_mark('condition_new', _CONDITION.get(p.get('condition_new', 1), 'Новый'))}",
            f"- Валюта: {_mark('currency', p.get('currency', ''))}",
            f"- Стоимость: {p.get('cost', 0):,.0f}".replace(",", " "),
            f"- Аванс: {_mark('prepaid', str(p.get('prepaid', 0)) + '%')} = {result['advance_sum']:,.0f} {p.get('currency', '')}".replace(",", " "),
            f"- Срок: {_mark('term', str(p.get('term', 0)) + ' мес.')}",
            f"- График: {_mark('type_schedule', _SCHEDULE_TYPES.get(str(p.get('type_schedule', '0')), 'аннуитет'))}",
            f"- Удорожание: {result.get('increase_percent', 0)}%",
            "",
            f"Аванс (платёж 0): {result['advance_sum']:,.2f} {p.get('currency', '')}".replace(",", " "),
        ]

        first = result.get("monthly_payment_first", 0)
        last = result.get("monthly_payment_last", 0)
        n = result.get("num_payments", 0)
        cur = p.get("currency", "")

        if abs(first - last) < 1.0:
            lines.append(f"Ежемесячный платёж: {first:,.2f} {cur} ({n} платежей)".replace(",", " "))
        else:
            lines.append(f"Первый платёж: {first:,.2f} {cur}, последний: {last:,.2f} {cur} ({n} платежей)".replace(",", " "))

        lines.append(f"Выкупной платёж: {result.get('buyout_sum', 0):,.2f} {cur}".replace(",", " "))
        lines.append(f"Общая сумма: {result.get('total_sum', 0):,.2f} {cur}".replace(",", " "))
        lines.append(f"Ссылка на график: {result.get('url', '')}")

        return "\n".join(lines)

    def format_sms_body(self, result: dict[str, Any]) -> str | None:
        if not result.get("ok"):
            return None
        p = result.get("params", {})
        cur = p.get("currency", "BYN")
        url = result.get("url", "")
        return (
            f"Микро Лизинг: расчёт лизинга\n"
            f"{p.get('subject', '')}, {p.get('cost', 0):,.0f} {cur}\n".replace(",", " ")
            + f"Аванс {p.get('prepaid', 0)}%: {result.get('advance_sum', 0):,.0f} {cur}\n".replace(",", " ")
            + f"Срок: {result.get('num_payments', 0)} мес.\n"
            + f"Удорожание: {result.get('increase_percent', 0)}%\n"
            + f"График платежей: {url}\n"
            + f"+375 17 322 77 00"
        )
```

- [ ] **Step 6: Register calculator in __init__.py**

Add to `backend/tools/__init__.py`:

```python
from .calculator import CalculatorTool

# Populated after settings are loaded
_TOOLS: dict[str, ToolDefinition] = {}


def init_tools(settings: Any) -> None:
    """Initialize tools with settings. Called once at app startup."""
    _TOOLS["calculator"] = CalculatorTool(
        base_url=settings.tools.calculator_api_base_url,
        token=settings.tools.calculator_api_token,
    )
```

Replace the empty `_TOOLS` dict and add `init_tools`.

- [ ] **Step 7: Run tests**

Run: `cd rag_demo_system && python -m pytest tests/test_calculator.py -v`
Expected: All tests PASS

- [ ] **Step 8: Run a live API test (no mock)**

```bash
cd rag_demo_system && python -c "
import asyncio
from backend.tools.calculator import CalculatorTool
import os

# Load env
for line in open('.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

tool = CalculatorTool(
    base_url=os.environ['CALCULATOR_API_BASE_URL'],
    token=os.environ['CALCULATOR_API_TOKEN'],
)
params, defaulted = tool.fill_defaults({'subject': 'Легковой автомобиль', 'cost': 25000})
result = asyncio.run(tool.execute(params, {}))
result['defaulted'] = defaulted
print(tool.format_voice_summary(result))
print('---')
print(tool.format_sms_body(result))
"
```

Expected: Formatted voice summary and SMS body with real numbers.

- [ ] **Step 9: Commit**

```bash
git add rag_demo_system/backend/tools/calculator.py rag_demo_system/backend/tools/__init__.py rag_demo_system/backend/settings.py rag_demo_system/config/app.yaml rag_demo_system/tests/test_calculator.py
git commit -m "feat: add calculator tool with API integration and formatting"
```

---

### Task 3: SMS sender tool

**Files:**
- Create: `rag_demo_system/backend/tools/sms_sender.py`
- Modify: `rag_demo_system/backend/tools/__init__.py`
- Test: `rag_demo_system/tests/test_sms_sender.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sms_sender.py
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import asyncio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools.sms_sender import SmsSenderTool


def test_schema():
    tool = SmsSenderTool(login="test", password="test", sender="MikroLizing")
    schema = tool.schema()
    func = schema["function"]
    assert func["name"] == "send_sms"
    assert "phone" in func["parameters"]["properties"]
    assert "message" in func["parameters"]["properties"]
    assert func["parameters"]["required"] == ["phone", "message"]


def test_execute_success():
    tool = SmsSenderTool(login="test", password="test", sender="MikroLizing")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "1454982446"

    with patch("backend.tools.sms_sender.httpx.get", return_value=mock_resp):
        result = tool.execute(
            {"phone": "375291224557", "message": "Test message"},
            {},
        ))

    assert result["ok"] is True
    assert result["message_id"] == "1454982446"


def test_execute_auth_failure():
    tool = SmsSenderTool(login="test", password="wrong", sender="MikroLizing")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "-2"

    with patch("backend.tools.sms_sender.httpx.get", return_value=mock_resp):
        result = tool.execute(
            {"phone": "375291224557", "message": "Test"},
            {},
        ))

    assert result["ok"] is False
    assert "error" in result


def test_phone_validation():
    tool = SmsSenderTool(login="test", password="test", sender="MikroLizing")
    result = asyncio.run(tool.execute({"phone": "123", "message": "Test"}, {}))
    assert result["ok"] is False
    assert "номер" in result["error"].lower() or "phone" in result["error"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rag_demo_system && python -m pytest tests/test_sms_sender.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement sms_sender.py**

```python
# backend/tools/sms_sender.py
from __future__ import annotations

import re
from typing import Any

import httpx

from .base import ToolDefinition

_SMS_API_URL = "https://userarea.sms-assistent.by/api/v1/send_sms/plain"

# Error codes from sms-assistent.by docs
_ERROR_CODES = {
    "-1": "Недостаточно средств на балансе SMS-сервиса",
    "-2": "Ошибка авторизации SMS-сервиса",
    "-10": "SMS API не активирован",
    "-13": "Трафик заблокирован",
}

# Phone must be 12 digits (375XXXXXXXXX)
_PHONE_RE = re.compile(r"^375\d{9}$")


class SmsSenderTool(ToolDefinition):
    def __init__(self, login: str, password: str, sender: str) -> None:
        self._login = login
        self._password = password
        self._sender = sender

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "send_sms",
                "description": (
                    "Отправить СМС клиенту. Используйте ТОЛЬКО после явного согласия клиента. "
                    "Обычно отправляется ссылка на график платежей после расчёта."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "Номер телефона клиента в формате 375XXXXXXXXX (12 цифр без +)",
                        },
                        "message": {
                            "type": "string",
                            "description": "Текст СМС сообщения",
                        },
                    },
                    "required": ["phone", "message"],
                },
            },
        }

    def defaults(self) -> dict[str, Any]:
        return {}

    def execute(self, params: dict[str, Any], session_context: dict[str, Any]) -> dict[str, Any]:
        phone = re.sub(r"[^\d]", "", params.get("phone", ""))
        if phone.startswith("+"):
            phone = phone[1:]
        message = params.get("message", "")

        if not _PHONE_RE.match(phone):
            return {"ok": False, "error": f"Некорректный номер телефона: {phone}. Ожидается формат 375XXXXXXXXX."}

        if not message:
            return {"ok": False, "error": "Текст сообщения пустой."}

        try:
            resp = httpx.get(
                _SMS_API_URL,
                params={
                    "user": self._login,
                    "password": self._password,
                    "recipient": phone,
                    "message": message,
                    "sender": self._sender,
                },
                timeout=10.0,
            )
        except httpx.TimeoutException:
            return {"ok": False, "error": "SMS-сервис временно недоступен."}
        except httpx.RequestError as exc:
            return {"ok": False, "error": f"Ошибка подключения к SMS-сервису: {exc}"}

        body = resp.text.strip()

        # Positive number = message ID (success)
        if body.lstrip("-").isdigit() and int(body) > 0:
            return {"ok": True, "message_id": body}

        # Negative number = error code
        error_msg = _ERROR_CODES.get(body, f"Ошибка SMS-сервиса (код {body})")
        return {"ok": False, "error": error_msg}

    def format_voice_summary(self, result: dict[str, Any]) -> str:
        if result.get("ok"):
            return "СМС успешно отправлено."
        return f"Не удалось отправить СМС: {result.get('error', 'неизвестная ошибка')}."
```

- [ ] **Step 4: Register SMS tool in __init__.py**

Add to `init_tools()` in `backend/tools/__init__.py`:

```python
from .sms_sender import SmsSenderTool

# Inside init_tools():
_TOOLS["send_sms"] = SmsSenderTool(
    login=settings.tools.sms_api_login,
    password=settings.tools.sms_api_password,
    sender=settings.tools.sms_sender_name,
)
```

- [ ] **Step 5: Run tests**

Run: `cd rag_demo_system && python -m pytest tests/test_sms_sender.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/tools/sms_sender.py rag_demo_system/backend/tools/__init__.py rag_demo_system/tests/test_sms_sender.py
git commit -m "feat: add SMS sender tool with phone validation"
```

---

### Task 4: Filler phrases

**Files:**
- Create: `rag_demo_system/backend/tools/filler.py`

- [ ] **Step 1: Create filler.py**

```python
# backend/tools/filler.py
from __future__ import annotations

import random

_FILLERS: dict[str, list[str]] = {
    "calculator": [
        "Секундочку, рассчитываю.",
        "Один момент, считаю для вас.",
        "Сейчас посчитаю.",
    ],
    "send_sms": [
        "Отправляю сообщение.",
        "Секунду, отправляю СМС.",
    ],
    "escalate_to_human": [
        "Передаю информацию специалисту.",
        "Секунду, связываю со специалистом.",
    ],
}

_DEFAULT_FILLER = "Один момент."


def get_filler(tool_name: str) -> str:
    """Return a random filler phrase for the given tool."""
    phrases = _FILLERS.get(tool_name, [_DEFAULT_FILLER])
    return random.choice(phrases)
```

- [ ] **Step 2: Commit**

```bash
git add rag_demo_system/backend/tools/filler.py
git commit -m "feat: add filler phrases for tool execution latency"
```

---

### Task 5: LLM streaming with tool call support

**Files:**
- Modify: `rag_demo_system/backend/llm_stream.py`
- Modify: `rag_demo_system/backend/llm.py`
- Test: `rag_demo_system/tests/test_tool_loop.py`

- [ ] **Step 1: Write the failing test for tool call parsing**

```python
# tests/test_tool_loop.py
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.llm_stream import iter_openai_stream_events, parse_tool_calls_from_events


def test_parse_tool_call_from_stream():
    """Simulate vLLM streaming a tool_call response."""
    events = [
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "function": {"name": "calculator", "arguments": ""},
                    }]
                },
                "finish_reason": None,
            }]
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"subject": "Лег'},
                    }]
                },
                "finish_reason": None,
            }]
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": 'ковой автомобиль", "cost": 30000}'},
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        },
    ]
    result = parse_tool_calls_from_events(events)
    assert len(result) == 1
    assert result[0]["id"] == "call_123"
    assert result[0]["function"]["name"] == "calculator"
    args = json.loads(result[0]["function"]["arguments"])
    assert args["subject"] == "Легковой автомобиль"
    assert args["cost"] == 30000


def test_parse_content_no_tool_calls():
    """Regular content response should return empty list."""
    events = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
    ]
    result = parse_tool_calls_from_events(events)
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rag_demo_system && python -m pytest tests/test_tool_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_tool_calls_from_events'`

- [ ] **Step 3: Add parse_tool_calls_from_events to llm_stream.py**

Append to `backend/llm_stream.py`:

```python
def parse_tool_calls_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accumulate streamed tool_call deltas into complete tool_call objects.

    Returns a list of tool call dicts: [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
    Returns empty list if no tool calls in the events.
    """
    calls: dict[int, dict[str, Any]] = {}
    for event in events:
        choice = (event.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        tool_calls = delta.get("tool_calls")
        if not tool_calls:
            continue
        for tc in tool_calls:
            idx = tc.get("index", 0)
            if idx not in calls:
                calls[idx] = {
                    "id": tc.get("id", ""),
                    "function": {"name": "", "arguments": ""},
                }
            if tc.get("id"):
                calls[idx]["id"] = tc["id"]
            func = tc.get("function") or {}
            if func.get("name"):
                calls[idx]["function"]["name"] = func["name"]
            if func.get("arguments"):
                calls[idx]["function"]["arguments"] += func["arguments"]
    return list(calls.values())
```

- [ ] **Step 4: Modify iter_openai_compatible_stream_events in llm.py to accept messages and tools**

Replace the current `iter_openai_compatible_stream_events` function in `backend/llm.py` (lines 87-119):

```python
def iter_openai_compatible_stream_events(
    base_url: str,
    model: str,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 120,
    timeout_sec: int = 60,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    if not base_url:
        raise ValueError("RAG_LLM_BASE_URL is not set")
    if not model:
        raise ValueError("RAG_LLM_MODEL is not set")
    url = base_url.rstrip("/") + "/chat/completions"

    # Build messages: either from explicit list or from system+user prompts
    if messages is not None:
        msg_list = messages
    else:
        msg_list = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ]

    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": msg_list,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(url, json=payload, timeout=timeout_sec, stream=True)
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        for event in iter_openai_stream_events([line]):
            yield event
```

Keep the existing `call_openai_compatible` and `iter_openai_compatible_stream` functions unchanged.

- [ ] **Step 5: Run all tests**

Run: `cd rag_demo_system && python -m pytest tests/test_tool_loop.py tests/test_llm_stream.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/llm_stream.py rag_demo_system/backend/llm.py rag_demo_system/tests/test_tool_loop.py
git commit -m "feat: add tool call parsing and tools param to LLM streaming"
```

---

### Task 6: Tool loop in the streaming pipeline

**Files:**
- Modify: `rag_demo_system/backend/app.py` (the `_stream_voice_response` function and `llm_producer`)
- Modify: `rag_demo_system/backend/voice_session.py`

This is the core integration. The `llm_producer` inside `_stream_voice_response` must detect tool calls, execute tools, send filler phrases, and re-call the LLM.

- [ ] **Step 1: Add tool tracking to VoiceSession**

In `backend/voice_session.py`, add field:

```python
tool_calls_this_turn: list[dict] = field(default_factory=list)
```

(Change `VoiceSession` from `@dataclass` to `@dataclass` with `from dataclasses import dataclass, field`)

- [ ] **Step 2: Modify _stream_voice_response in app.py**

This is the largest change. The key modifications to `_stream_voice_response`:

**2a.** After building `user_prompt` (line ~606), build a message list instead of relying on system_prompt + user_prompt strings:

```python
from .tools import get_tool_schemas, get_tool
from .tools.filler import get_filler
from .llm_stream import parse_tool_calls_from_events

# Build initial messages list for tool-aware LLM call
llm_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
tool_schemas = get_tool_schemas()
```

**2b.** Replace the `llm_producer` function with a tool-loop-aware version:

```python
async def llm_producer() -> None:
    nonlocal t_llm_first_token
    max_tool_iterations = 3

    for iteration in range(max_tool_iterations + 1):
        detector = SentenceDetector()
        collected_events: list[dict] = []
        has_content = False

        try:
            stream = iter_openai_compatible_stream_events(
                base_url=effective_base_url,
                model=effective_model,
                messages=llm_messages,
                temperature=settings.llm.temperature,
                max_tokens=voice_max_tokens if not has_content else 220,
                timeout_sec=settings.llm.timeout_sec,
                tools=tool_schemas if iteration < max_tool_iterations else None,
            )
            _sentinel = object()
            while True:
                if session.interrupted:
                    break
                event = await asyncio.to_thread(next, stream, _sentinel)
                if event is _sentinel:
                    break
                collected_events.append(event)
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}

                # Regular content token
                token = delta.get("content") or ""
                if token:
                    has_content = True
                    if t_llm_first_token is None:
                        t_llm_first_token = time.time()
                    for sent in detector.feed(token):
                        cleaned = clean_answer(sent)
                        if cleaned:
                            await sentence_queue.put(cleaned)

        except Exception as exc:
            state.log({"event": "llm_error", "error": str(exc), "session_id": session_id})
            break

        # Flush remaining content
        remaining = detector.flush()
        if remaining and not session.interrupted:
            cleaned = clean_answer(remaining)
            if cleaned:
                await sentence_queue.put(cleaned)

        # If we got content, we are done (no tool call)
        if has_content or session.interrupted:
            break

        # Check for tool calls
        tool_calls = parse_tool_calls_from_events(collected_events)
        if not tool_calls:
            break

        # Process each tool call
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                func_args = {}

            # Send filler phrase to TTS
            filler = get_filler(func_name)
            await sentence_queue.put(filler)

            # Execute tool
            try:
                tool = get_tool(func_name)
                filled_params, defaulted = tool.fill_defaults(func_args)
                result = await asyncio.to_thread(
                    tool.execute, filled_params, {"session_id": session_id}
                )
                result["defaulted"] = defaulted
                session.tool_calls_this_turn.append({
                    "tool": func_name, "params": filled_params, "result": result,
                })

                # Format result for LLM
                summary = tool.format_voice_summary(result)
            except KeyError:
                summary = f"Инструмент '{func_name}' не найден."
                result = {"ok": False, "error": summary}
            except Exception as exc:
                summary = f"Ошибка выполнения инструмента: {exc}"
                result = {"ok": False, "error": str(exc)}

            # Append tool call + result to messages for next LLM iteration
            llm_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc.get("id", f"call_{func_name}"),
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(func_args, ensure_ascii=False),
                    },
                }],
            })
            llm_messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{func_name}"),
                "content": summary,
            })

    await sentence_queue.put(None)
```

Note on `asyncio.to_thread(asyncio.run, tool.execute(...))`: the tool's `execute` is async (uses httpx), but `_stream_voice_response` already runs in the event loop. We need to call synchronous httpx from a thread. **Simpler alternative:** change tool `execute` to use synchronous `httpx.get` (which it already does) and call via `asyncio.to_thread(tool.execute_sync, ...)`. Let the implementation task decide the cleanest approach; the key contract is: tool execution must not block the event loop.

- [ ] **Step 3: Initialize tools at app startup**

In `backend/app.py`, in the startup section (after `settings = load_settings()`), add:

```python
from .tools import init_tools
init_tools(settings)
```

- [ ] **Step 4: Test locally with mock tool call response**

Create a quick script to test the tool call parsing and execution flow without a real LLM:

```bash
cd rag_demo_system && python -c "
import asyncio, json
from backend.tools import init_tools, get_tool_schemas, get_tool
from backend.tools.filler import get_filler
from backend.llm_stream import parse_tool_calls_from_events

# Mock settings
class MockTools:
    calculator_api_base_url = ''
    calculator_api_token = ''
    sms_api_login = ''
    sms_api_password = ''
    sms_sender_name = 'MikroLizing'
    crm_webhook_url = ''
    crm_webhook_token = ''
class MockSettings:
    tools = MockTools()

init_tools(MockSettings())

# Check schemas are valid
schemas = get_tool_schemas()
print(f'Registered tools: {len(schemas)}')
for s in schemas:
    print(f'  - {s[\"function\"][\"name\"]}')

# Check filler
print(f'Filler for calculator: {get_filler(\"calculator\")}')

# Simulate tool call events from LLM
events = [
    {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'calculator', 'arguments': ''}}]}, 'finish_reason': None}]},
    {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': json.dumps({'subject': 'Легковой автомобиль', 'cost': 30000})}}]}, 'finish_reason': 'tool_calls'}]},
]
calls = parse_tool_calls_from_events(events)
print(f'Parsed tool calls: {json.dumps(calls, ensure_ascii=False, indent=2)}')
print('OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/backend/voice_session.py
git commit -m "feat: integrate tool loop into voice streaming pipeline"
```

---

### Task 7: System prompt and KB updates

**Files:**
- Modify: `rag_demo_system/config/system_prompt_ru_v2.txt`
- Modify: `rag_demo_system/knowledge_base/kb_faq_ru_v2.md`

- [ ] **Step 1: Update system prompt**

**1a.** Replace lines 123-128 (Integration Points section) with the full tool instructions block from the spec (Section 8 of the design doc).

**1b.** Fix line 29: change "Не называть процентные ставки и переплаты. Условия рассчитываются индивидуально менеджером." to "Не придумывать процентные ставки и переплаты. Если калькулятор вернул результат, можно озвучить суммы платежей из расчёта. Не называть ставки если они не были явно возвращены калькулятором."

**1c.** Fix line 64: change "Вопрос требует индивидуального расчета, который вы не можете сделать" to "Вопрос требует индивидуального расчёта, выходящего за возможности калькулятора (нестандартные условия, акции, специальные программы)"

**1d.** Update consultation flow (line ~88): add "На шаге расчёта используйте инструмент calculator. После расчёта предложите отправить график по СМС."

- [ ] **Step 2: Review KB for conflicting entries**

Search `kb_faq_ru_v2.md` and `kb_faq_ru.yaml` / `kb_faq_ru.json` for entries that say "менеджер рассчитает", "индивидуальный расчёт", "данные для расчета" and update them to reflect that the system can now calculate. Do NOT delete entries; soften the redirect language.

- [ ] **Step 3: Verify prompt loads correctly**

```bash
cd rag_demo_system && python -c "
from pathlib import Path
prompt = Path('config/system_prompt_ru_v2.txt').read_text(encoding='utf-8')
assert 'Калькулятор (calculator)' in prompt
assert 'Отправка СМС (send_sms)' in prompt
assert 'Не придумывать процентные ставки' in prompt
print(f'Prompt length: {len(prompt)} chars')
print('Tool section found: OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add rag_demo_system/config/system_prompt_ru_v2.txt rag_demo_system/knowledge_base/
git commit -m "feat: update system prompt and KB for tool use"
```

---

### Task 8: Tool registry test

**Files:**
- Test: `rag_demo_system/tests/test_tool_registry.py`

- [ ] **Step 1: Write registry test**

```python
# tests/test_tool_registry.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tools import init_tools, get_tool_schemas, get_tool, get_all_tools


class MockTools:
    calculator_api_base_url = "http://test"
    calculator_api_token = "tok"
    sms_api_login = "login"
    sms_api_password = "pass"
    sms_sender_name = "Test"
    crm_webhook_url = ""
    crm_webhook_token = ""


class MockSettings:
    tools = MockTools()


def test_init_tools_registers_all():
    init_tools(MockSettings())
    tools = get_all_tools()
    assert "calculator" in tools
    assert "send_sms" in tools


def test_get_tool_schemas_returns_list():
    init_tools(MockSettings())
    schemas = get_tool_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) >= 2
    names = [s["function"]["name"] for s in schemas]
    assert "calculator" in names
    assert "send_sms" in names


def test_get_tool_by_name():
    init_tools(MockSettings())
    calc = get_tool("calculator")
    assert calc is not None
    assert calc.schema()["function"]["name"] == "calculator"


def test_get_tool_unknown_raises():
    init_tools(MockSettings())
    try:
        get_tool("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
```

- [ ] **Step 2: Run all tests**

Run: `cd rag_demo_system && python -m pytest tests/test_tool_base.py tests/test_calculator.py tests/test_sms_sender.py tests/test_tool_loop.py tests/test_tool_registry.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add rag_demo_system/tests/test_tool_registry.py
git commit -m "test: add tool registry integration tests"
```

---

### Task 9: End-to-end local integration test

**Files:**
- No new files. Run manual verification.

- [ ] **Step 1: Test calculator tool with real API**

```bash
cd rag_demo_system && python -c "
import asyncio, os
from backend.tools.calculator import CalculatorTool

for line in open('.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

tool = CalculatorTool(
    base_url=os.environ['CALCULATOR_API_BASE_URL'],
    token=os.environ['CALCULATOR_API_TOKEN'],
)

# Test 1: Normal calculation
params, defaulted = tool.fill_defaults({'subject': 'Легковой автомобиль', 'cost': 20000, 'currency': 'USD'})
result = asyncio.run(tool.execute(params, {}))
result['defaulted'] = defaulted
assert result['ok'], f'Failed: {result}'
print('Test 1 PASS: Normal calculation')
print(tool.format_voice_summary(result))
print()

# Test 2: 404 (impossible params)
result2 = asyncio.run(tool.execute({
    'subject': 'Легковой автомобиль', 'cost': 1, 'client_type': 'Физическое лицо',
    'condition_new': 0, 'age': 99, 'currency': 'BYN', 'prepaid': 99, 'term': 1, 'type_schedule': '0',
}, {}))
assert not result2['ok'], 'Expected 404 for impossible params'
print(f'Test 2 PASS: 404 handled: {result2[\"error\"]}')
print()

# Test 3: Used without age
result3 = asyncio.run(tool.execute({
    'subject': 'Легковой автомобиль', 'cost': 20000, 'condition_new': 0,
    'client_type': 'Физическое лицо', 'currency': 'BYN', 'prepaid': 30, 'term': 36, 'type_schedule': '0',
}, {}))
assert not result3['ok']
assert 'возраст' in result3['error'].lower()
print(f'Test 3 PASS: Used without age rejected: {result3[\"error\"]}')
print()

# Test 4: SMS body
sms = tool.format_sms_body(result)
assert 'mikro-leasing.by/graphic' in sms
print(f'Test 4 PASS: SMS body contains link')
print(sms)
"
```

- [ ] **Step 2: Test SMS with real API (send to your number)**

```bash
cd rag_demo_system && python -c "
import asyncio, os
from backend.tools.sms_sender import SmsSenderTool

for line in open('.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

tool = SmsSenderTool(
    login=os.environ['SMS_API_LOGIN'],
    password=os.environ['SMS_API_PASSWORD'],
    sender=os.environ['SMS_SENDER_NAME'],
)
result = asyncio.run(tool.execute({
    'phone': '375291224557',
    'message': 'Тестовое сообщение из системы инструментов. Если получили, значит интеграция работает.',
}, {}))
print(f'Result: {result}')
assert result['ok'], f'SMS failed: {result}'
print('SMS sent successfully')
"
```

- [ ] **Step 3: Run the full test suite**

Run: `cd rag_demo_system && python -m pytest tests/ -v --ignore=tests/test_whisper_server.py --ignore=tests/test_vad.py --ignore=tests/test_audio_input.py --ignore=tests/test_rtc_audio.py -x`

Ignore tests that require GPU/audio hardware. All tool tests must pass.

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: tool use system complete - calculator, SMS, streaming integration"
```

---

### Task 10: Server integration testing (when GPU server is available)

**Files:**
- No code changes. Verification on server.

- [ ] **Step 1: Deploy to server**

```bash
ssh -i ~/.ssh/jarvislabs sesterce@<SERVER_IP>
cd ~/leasing/rag_demo_system
git pull origin feature/tool-use
# Copy .env with real credentials
```

- [ ] **Step 2: Test vLLM tool calling with Qwen3.5**

On the server, test that vLLM correctly handles `tools=[]` with Qwen3.5-35B-A3B-FP8:

```bash
curl -X POST http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B-FP8",
    "messages": [
      {"role": "system", "content": "Ты голосовой помощник лизинговой компании."},
      {"role": "user", "content": "Рассчитай лизинг на машину за 30000 долларов"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "Рассчитать график лизинговых платежей",
          "parameters": {
            "type": "object",
            "properties": {
              "subject": {"type": "string"},
              "cost": {"type": "number"}
            },
            "required": ["subject", "cost"]
          }
        }
      }
    ],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Expected: Response contains `tool_calls` with `calculator` and appropriate arguments.

- [ ] **Step 3: Test end-to-end voice flow**

Open the demo UI, switch to voice mode, and test:
1. "Рассчитай лизинг на Kia Sportage за 25 тысяч долларов" - should trigger calculator tool
2. After getting results: "Отправь мне график по СМС" - should trigger send_sms tool
3. "Какие документы нужны?" - should NOT trigger any tool (regular RAG)
4. "Сколько будет стоить грузовик за 50 тысяч?" - should trigger calculator with subject "Грузовой автомобиль"

- [ ] **Step 4: If vLLM tool calling fails**

Fall back to prompt-driven approach: add tool call markers to the system prompt, parse them from text output. This is a separate task only if needed.
