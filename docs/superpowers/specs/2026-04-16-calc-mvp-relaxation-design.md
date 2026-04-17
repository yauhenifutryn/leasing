# Spec 1: Calculator MVP Relaxation

**Cluster:** A — Calculator
**Depends on:** Spec 2 (ClientProfile must exist as gate)
**Blocks:** —

## Context

Transcript 779154b4 shows calculator firing with hallucinated parameters (line
18:05:30: `calculator(cost=20000)` while client never stated cost), ignoring
linear graph requests (18:17-21), and rejecting USD from individuals without
recovery path (18:07). The current tool (`backend/tools/calculator.py:86-94`)
ships with hardcoded defaults that mask missing data and produce wrong output.

## Problem

1. `CalculatorTool.defaults()` returns `client_type=Физическое лицо, prepaid=30,
   term=36, currency=BYN, condition_new=1, type_schedule=0`. These silently fill
   missing parameters and hide the bug that SessionAgent did not collect them.
2. Classifier prompt (`app.py:746`) enforces `prepaid >= 10` client-side. Real
   minimum per Excel is 0% for new cars. Clients are told "невозможно" falsely.
3. System prompt (`config/system_prompt_ru_v2.txt:150`) states "срок 36 месяцев"
   as default. Bot then parrots "максимум 36" to clients (transcript 18:11:31),
   while real maximum is 84.
4. `type_schedule` is a declared parameter but the DirectTool path (`app.py:964`)
   never forwards it. Client's "линейный график" request gets lost.
5. `prepaid` accepts percent only on input. Output shows amount only. Clients
   asking "аванс пятнадцать тысяч" get converted awkwardly by LLM.
6. USD + Физическое лицо returns HTTP 404. Bot emits generic error. No fallback
   to BYN conversion.

## Goals

- Remove all default values from `CalculatorTool`. Invocation with missing
  required fields raises `IncompleteProfileError`.
- Accept `prepaid_pct` XOR `prepaid_amount` on input. Return both in result.
- Forward `type_schedule` from `ClientProfile` through the DirectTool path.
- Hardcoded 3 BYN/USD conversion for MVP only, applied before calculator call
  when `client_type=Физическое лицо AND currency=USD`.
- Explicit disclosure: "по курсу 3 рубля за доллар это X BYN".
- Widen classifier accepted ranges: `prepaid 0..40`, `term 12..84`.
- Reject EUR/RUB/other for Физическое лицо with clear voice message.
- Present all fields in the post-calc summary, marking none as `defaulted`
  (because defaults no longer exist).

## Non-goals

- Excel age × subject × advance matrix validation (production, Spec 7)
- NBRB live currency API (production, Spec 7)
- Structured error contract (production, Spec 7)
- Session persistence / cross-call profile (future)

## Design

### Component changes

**`backend/tools/calculator.py`**
- Delete `defaults()` method or return `{}`.
- `execute()` asserts all 8 required fields present; raises
  `IncompleteProfileError("missing: [field1, field2]")` otherwise.
- Accept `prepaid_pct` or `prepaid_amount` (not both). Convert `prepaid_amount`
  to percentage using `cost` before API call. Validate 0 ≤ pct ≤ 40.
- Response parser: always populate `prepaid_pct` and `prepaid_amount` in the
  output dict.

**`backend/app.py` — DirectTool path (line 952-1020)**
- Build `_direct_params` from `session.client_profile` (not from ad-hoc
  `_extracted_hints`). Profile is the single source of truth.
- Pre-step: if `profile.client_type == "Физическое лицо"` and `profile.currency
  == "USD"`, apply conversion: `cost_byn = cost_usd * USD_BYN_RATE`, override
  `cost` and set `currency="BYN"`, note conversion in `result.currency_conversion`.
- Pre-step: if `profile.client_type == "Физическое лицо"` and `profile.currency
  in ("EUR", "RUB", "CNY")`, raise `UnsupportedCurrencyError` — bot tells client
  "сейчас поддерживаются BYN и USD для физлиц".
- Forward `type_schedule`, `prepaid_pct` or `prepaid_amount`, `age` from profile.

**`config/system_prompt_ru_v2.txt`**
- Strip every mention of "30%", "36 месяцев", "по умолчанию".
- Replace `# Инструменты` section with collection-protocol language pointing to
  the read-back gate in Spec 2.
- Add: "Если клиент назвал USD, вы пересчитаете по курсу 3 рубля за доллар и
  обязательно озвучите конвертацию".

**`backend/app.py` — classifier prompt (line 709-748)**
- Relax: `prepaid` valid range `0..40`, no more `invalid_param` at 5%.
- Relax: `term` valid range `12..84`.
- Remove "минимум 10%" text.
- Add `type_schedule` to output JSON: `"type_schedule": "0/1 or null"`.
- Extraction rule: "аннуитет/аннуитетный" → `"0"`; "линейный/убывающий/
  дифференцированный" → `"1"`.

**Post-calc summary (`app.py:1089-1117`)**
- Always announce: subject, cost+currency, client_type, prepaid (both % and
  amount), term, schedule type, monthly payment, buyout, total, удорожание.
- No "(по умолчанию)" tags — defaults don't exist.
- Offer: "Поменять аванс, срок или график? Или отправить по СМС?".

### Data contract

```python
# ClientProfile → calculator.execute() params
{
  "subject": str,              # from profile
  "cost": float,               # converted to BYN if USD+ФЛ
  "currency": str,             # BYN always after conversion for ФЛ
  "client_type": str,          # "Физическое лицо" or "Юридическое лицо"
  "condition_new": int,        # 1 or 0
  "age": int | None,           # required when condition_new=0
  "prepaid_pct": float | None, # XOR
  "prepaid_amount": float | None,
  "term": int,                 # 12..84
  "type_schedule": str,        # "0" or "1"
}
```

### Error surfacing

| Situation | Bot says |
|---|---|
| `UnsupportedCurrencyError` (EUR/RUB for ФЛ) | "Сейчас могу считать в белорусских рублях или долларах. В какой валюте стоимость?" |
| Currency conversion (USD→BYN) | "По курсу 3 рубля за доллар это X BYN. Продолжаем?" |
| API returns 404 | "По таким параметрам условия не нашла. Попробуем другой аванс или срок?" |
| `IncompleteProfileError` | Should never happen in prod — ClientProfile gate prevents it. Logs `ERROR` and tells user "секундочку, уточним". |

## Files to change

- `rag_demo_system/backend/tools/calculator.py`
- `rag_demo_system/backend/app.py` (classifier prompt; DirectTool path; summary)
- `rag_demo_system/backend/session.py` (add `USD_BYN_RATE` env read; optional)
- `rag_demo_system/config/system_prompt_ru_v2.txt`
- `rag_demo_system/.env.example` (add `USD_BYN_RATE=3.0`)
- Tests: `rag_demo_system/tests/test_calculator_tool.py` (new)

## Testing

**Unit — `test_calculator_tool.py`**
1. `execute({}, {})` → raises `IncompleteProfileError`, lists all 8 missing fields.
2. `execute({..., "prepaid_amount": 14000, "cost": 70000, ...})` → pct=20 sent
   to API; output contains both pct and amount.
3. `execute({..., "client_type": "Физическое лицо", "currency": "EUR", ...})`
   → raises `UnsupportedCurrencyError`.
4. `execute({..., "client_type": "Физическое лицо", "currency": "USD",
   "cost": 24300, ...})` → pre-converts to 72900 BYN, API called with BYN.
5. `execute({..., "type_schedule": "1", ...})` → API called with `type_schedule=1`.

**Integration — transcript replay**
1. Replay client turns from transcript 779154b4. Assert:
   - At line 18:05:30, no calculator call (cost not yet in profile).
   - At line 18:07 (USD mention), conversion happens before call.
   - At line 18:11:27 (300 months), bot reports max 84, not 36.
   - At line 18:17:51 (linear graph), API receives `type_schedule=1`.

**Metrics**
- Log every `IncompleteProfileError` to stderr. Target: zero in first 48h post-deploy.
- Log every `UnsupportedCurrencyError`. Count per day.
- Log USD→BYN conversions. Expected: any physlico + USD call.

## Risks

| Risk | Mitigation |
|---|---|
| Breaking change: calculator refuses calls without profile | Must deploy Spec 2 simultaneously — ClientProfile is the feeding gate |
| 3 BYN/USD rate becomes wrong | Env var, one-line redeploy to change. Production Spec 7 replaces with NBRB |
| Classifier emits wrong `type_schedule` | Post-calc read-back says "график аннуитетный" explicitly; client can correct |

## Rollback

Single commit per cluster. If breakage: `git revert`, redeploy, profile remains
orphan structure but non-fatal.
