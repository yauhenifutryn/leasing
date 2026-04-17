# KB Audit Report — 2026-04-16

> **Status (2026-04-16):** patches applied to `knowledge_base/kb_faq_ru_v2.md`
> based on client-verified facts. Re-index required (`python scripts/index_kb.py`
> on server, or automatic via `smoke_test.sh`).
>
> **Applied:**
> - Added canonical "адреса офисов" section (all 6 cities with addresses)
> - Added "что такое нагрузка" definition
> - Added "типы графиков платежей" section (annuity + linear with synonyms)
> - Fixed office hours: all offices Mon-Fri only; call-center adds Saturday 10-16
>   (previous KB incorrectly said Minsk office works weekends)
> - Removed "минимальный аванс 30%" hardcoded escalation rule (now triggers on
>   calculator-reported failure instead)
> - Updated stale "24–36 месяцев" examples to "от 12 до 84 месяцев"
>
> **Deferred to future audit** (client said "not needed for MVP"):
> - Per-subject matrix (spectech, equipment, real estate) — calculator handles
> - Additional narrative 10%/30% examples — left intact (not rule-statements)
> - More granular S5-S8 stale content — only fixed blockers



**Source:** client test call transcript 779154b4 (Сергей, 18:00-18:25)
**KB file:** `knowledge_base/kb_faq_ru_v2.md`
**Method:** manual extraction of 20 client questions + keyword-overlap diff against KB paragraphs + regex scan for stale operational values.

## Summary

- 20 client questions analyzed
- **3 true GAPS** (no KB content addresses the topic)
- **8 STALE content blocks** (KB contradicts the new business rules from 2026-04-01 Excel)
- **9 PARTIAL/COVERED** (KB addresses but inconsistently or with dated numbers)

Estimated KB patch scope: ~15-20 line edits + 2 new sections (linear/annuity explanation, nagruzka definition).

---

## 1. True GAPS — add to KB

### G1. "Что такое нагрузка" (financial term)

**Client asked** at 18:03:13: "А что такое нагрузка, Ксения?"
**Bot improvised** an answer: "Нагрузка — это сумма ваших будущих платежей, которая зависит от стоимости предмета, срока лизинга и размера аванса."
**KB:** `нагрузка` appears 15 times in KB but ONLY as operational phrasing ("нагрузка по текущим обязательствам"), never as a definition.

**Proposed KB addition:**

```markdown
## Вопрос: Что такое нагрузка по лизингу

"Нагрузка" — общая сумма, которую клиент заплатит по договору лизинга за весь
срок: аванс + сумма всех ежемесячных платежей + выкупной платёж. Нагрузка
зависит от стоимости предмета, аванса, срока и ставки удорожания. Её также
называют "общая сумма сделки".

Например, для автомобиля стоимостью 70 000 BYN при авансе 20% и сроке 84
месяца общая нагрузка будет около 151 000 BYN (включая аванс 14 000 BYN и
84 платежа).

Синонимы, которые могут встретиться: переплата, общая сумма сделки, итого
по договору.
```

### G2. "Линейный график" / "дифференцированный график"

**Client asked** 5× (18:17-21): "Можешь сделать линейный график?"
**Bot failed** to honor the request even after multiple attempts.
**KB:** `линейн` has 0 hits. KB uses "график на уменьшение" and "классический" as synonyms without explaining.

**Proposed KB addition:**

```markdown
## Типы графиков платежей

"Микро Лизинг" поддерживает два типа графиков платежей:

1. **Аннуитетный** (равные платежи). Каждый месяц клиент платит одинаковую
   сумму. Удобно для планирования семейного бюджета. В начале срока большая
   часть платежа идёт на проценты, к концу — на погашение тела.

2. **Линейный** (также называется "дифференцированный", "классический",
   "график на уменьшение", "убывающий"). Тело лизинга погашается равными
   долями, а проценты начисляются на остаток. Первые платежи больше, к
   концу срока они уменьшаются. Общая переплата по линейному графику обычно
   меньше чем по аннуитетному.

По умолчанию калькулятор считает аннуитет. Для линейного графика клиент
должен явно попросить "линейный" или "дифференцированный" график.
```

### G3. Могилёвский офис — адрес и часы работы

**Client asked** at 18:24:19: "Могу ли я приехать в Могилёв?" + "Со скольки до скольки работает офис?"
**Bot answered** with specific address "улица Комсомольская, 10а" and "9:00 до 17:30 в будни" — but I couldn't find these specific facts in the KB.
**Risk:** Bot may have hallucinated the address or time — this is serious.
**Action:** Verify with client whether these are correct. If yes, add to KB as hard facts.

**Proposed verification question for you:** Is "улица Комсомольская, 10а" the actual Mogilev office address? Are the hours 9:00-17:30 Mon-Fri correct?

Once verified, add to KB as:

```markdown
## Адреса и часы работы офисов

### Могилёв
Адрес: улица Комсомольская, 10а
Часы работы: будние дни с 9:00 до 17:30
Телефон: +375 17 322 77 00

(Аналогично для Минска, Гомеля, Бреста, Витебска, Гродно — добавить
фактические адреса и часы по предоставленным клиентом данным.)
```

---

## 2. STALE content — needs update

### S1. "Минимальный аванс 30%" (KB escalation rule)

**Location:** KB paragraph mentioning "если клиент не может внести минимальный аванс 30%" as a reason to escalate.
**Problem:** Per 2026-04-01 Excel, minimum advance is **0%** for new cars, 10% for used up to 5 years, 15-25% for older. "30%" is a stale default, not a business rule.
**Fix:** Replace "минимальный аванс 30%" with language that references the Excel matrix:

```markdown
- если аванс, который клиент может внести, ниже минимального по матрице 
  условий для данной комбинации (возраст/состояние/тип предмета)
```

### S2. "до 36 месяцев" language (if present)

**Count:** 2 hits for `до 36 месяц` in KB. KB states max term as 36 somewhere.
**Problem:** Real max per Excel is up to 84 мес for certain configurations.
**Fix:** Remove explicit numeric caps in generic language; rely on calculator API.

### S3. "аванс 10%" / "аванс 30%" as facts

**Count:** 10%: 25 hits, 30%: 14 hits in KB.
**Problem:** Many are narrative example numbers, not all stale. But some function as business rules (e.g., "минимальный 10% для юрлица").
**Action needed:** Manual pass to distinguish narrative examples (keep) from rule statements (update). Estimated 5-8 rule statements need updating.

### S4. "аннуитетный, классический и т.п."

**Location:** KB says graph types are "аннуитетный, классический и т.п.".
**Problem:** "и т.п." is vague, "классический" is a synonym for linear that clients don't use.
**Fix:** Replace with "аннуитетный, линейный (также называют дифференцированным или графиком на уменьшение)".

### S5-S8. Other likely stale content

Manual review recommended for:
- Mentions of fixed maximum terms per subject type
- Mentions of fixed minimum advances per subject type
- Currency restrictions ("только BYN") that should reflect the new MVP policy (BYN + USD conversion for physlico)
- Any mention of "специалист уточнит срок" when calculator should handle it

---

## 3. Partial / Covered — no action needed

| Client question | KB coverage |
|---|---|
| "Что такое лизинг и чем отличается от кредита" | Covered — KB has intro paragraph |
| "Лизинг это очень дорого" | Covered as objection-handling |
| "Можно профинансировать землю" | Covered — "земля не финансируется, но коммерческая недвижимость — да" |
| "Какие типы предметов" | Covered — full list present |
| "В каких валютах" | Partial — KB lists BYN/USD/EUR/RUB but the USD-for-physlico MVP flow isn't in KB |
| "Минимальный аванс" | Partial — needs S1 + S3 fixes |
| "Вы большая компания" | Partial — KB mentions 2009 founding but not scale |
| "Могу приехать в Могилёв" | Partial — city listed in IVR but office hours/address gap (G3) |

---

## 4. Operational integration

After applying patches:

1. **Re-index the KB:**
   ```bash
   # On server:
   cd /ephemeral/leasing/rag_demo_system
   python scripts/index_kb.py
   ```

2. **Verify retrieval on the new content:**
   ```bash
   python scripts/voice_lab.py --query "что такое нагрузка"
   python scripts/voice_lab.py --query "линейный график"
   python scripts/voice_lab.py --query "минимальный аванс физлицу легковой"
   ```

3. **Re-run `kb_gap_report.py` after a few more calls** to verify the G1-G3 topics drop out of the gap ranking.

---

## 5. Patch proposals

These are 3 concrete KB sections to add or edit. I recommend you review them with the client before applying, especially:
- G3 (Mogilev address/hours) — must verify facts
- S1 (escalation rule change)
- S4 (schedule type language)

Once you approve, I can produce a diff patch against `kb_faq_ru_v2.md`.
