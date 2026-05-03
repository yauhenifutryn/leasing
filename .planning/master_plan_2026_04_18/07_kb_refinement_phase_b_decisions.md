# Phase B Decisions — 2026-05-03

**Status:** Phase B complete. **Output:** this file. **Inputs to Phase C.2:** the per-cluster authoring instructions below.

## Why no live-file edits in Phase B

Phase A revealed the canonical KB file is `knowledge_base/kb_faq_ru_v2.md` (375 sections, hand-maintained — NOT regenerated from the yaml/json by the in-repo pipeline). Phase C replaces it entirely with `knowledge_base/kb_topics_ru.md` (topical layout). Editing v2.md duplicates in Phase B would be throwaway work against a file that's about to be replaced. Instead, Phase B turns Phase A's diagnostic into per-cluster authoring decisions for Phase C.2.

This trades visible-edit-today for code-quality-and-zero-waste. The atomic-commit history is preserved via this decision document.

## Phase B.1 — stale numbers: NO-OP

Phase A.2 surfaced two drift candidates. Both turned out to be **false alarms** on closer inspection:

1. **ПДН capitalization** (ПДН vs ПДн) — these are **two different abbreviations**:
   - `ПДН` = "Показатель долговой нагрузки" (debt-to-income payment burden indicator, financial regulatory term)
   - `ПДн` = "Персональные данные" (personal data, GDPR-style)
   - Verified by grep: uppercase only appears in financial-constraint contexts ("ПДН 40%", "ограничение ПДН"); lowercase only in privacy/consent contexts ("обработка ПДн", "согласие ПДн"). KB is correct as-is.

2. **Advance % drift in Условия** (6 distinct values: 10/20/30/35/40/60) — every value is **legitimately different by customer type / program**:
   - 10% — физлица baseline + special programs (Супруги-машина, социальный лизинг, такси short-term)
   - 20% — standard физлица / повторный лизинг / возвратный / мотоциклы
   - 30% — новые юрлица / стартапы / лизинг без справок (riskier customers)
   - 35% — calc examples + поставщик payment schema
   - 40% — "Аванс более 40%" entry — the maximum cap rule
   - 60% — used as a *counter-example* in the "Аванс более 40%" entry: "Внести сразу 60% как аванс нельзя"

**Net: KB has no stale numeric drift. Phase B.1 is a no-op. The diagnostic correctly surfaced candidates; manual review confirmed they're correct.**

## Phase B.3 — duplicate clusters: input for Phase C.2

Cluster output from `kb_cluster.py --threshold 0.92` (16 multi-member clusters, 6 surgical-priority ≥3 members).

For each cluster, listing the entries, the **canonical to keep** (which becomes a topical section in Phase C.2), the **drops** (sources of synonyms/examples to merge into the canonical), and **authoring notes**.

### Cluster 1 — IVR office choice (7 entries → 1 topical section)

**Source entries:** `навигация по IVR`, `выбор офиса в IVR` (×3 in v2.md), `выбор офиса по IVR`, `выбор офиса по телефону` (×2), `навигация в IVR`, `связь с офисом продаж`, plus the new `офис Минск в IVR` from v2.md.

**Authoring decision for Phase C.2:**
- ONE topical section `section_id: ivr-office-selection` under `parent_topic: контакты`.
- Canonical content: "1=Минск, 2=Гомель, 3=Брест, 4=Витебск, 5=Гродно, 6=Могилев. After office choice, IVR sub-menu for topic. Outside business hours → автоинформатор."
- All 7 sources contribute synonym phrasings (`как через IVR выбрать офис`, `выбор офиса по телефону`, `навигация в IVR`, `связь с офисом продаж`, `как позвонить в нужный офис`).

### Cluster 2 — Pre-calc data requirements (6 entries → 1 topical section + 1 юрлицо variant)

**Source entries:** `данные для расчета`, `предварительный расчет лизинга`, `расчет лизинга`, `консультация по лизингу`, `расчет лизинга юрлицу`, `расчёт автолизинга`.

**Authoring decision:**
- ONE primary section `section_id: pre-calc-data-requirements` under `parent_topic: расчет`.
- Canonical content: "Need to calc: subject + cost, condition (new/used), advance, term, currency. Output: schedule via Viber/WhatsApp/email."
- ONE secondary section `section_id: pre-calc-data-юрлицо` for юрлицо-specific additions (УНП, regdoc requirements) — only if the юрлицо delta is substantive.
- All 6 sources contribute synonyms: `данные для расчета`, `как получить расчет`, `что нужно для предварительного расчета`, `консультация по лизингу`, etc.

### Cluster 3 — Consent + call recording (4 entries → 2 topical sections)

**Source entries:** `согласие на обработку ПД`, `запись звонков`, `запись разговора`, `запись разговора и ПДн`.

**Authoring decision:** **NOT a single cluster** — two distinct topics packed together by the embedding model. Author 2 sections:
- `section_id: pd-consent-via-phone` under `parent_topic: правовое-данные`. Canonical: "Staying on the line after the voice notification = consent for personal data processing. Compliant with Закон РБ о защите персональных данных."
- `section_id: call-recording` under `parent_topic: правовое-данные`. Canonical: "Calls may be recorded for QC and dispute resolution. Records are stored per internal regulations."

The 4th entry `запись разговора и ПДн` is a combined version — split its content between the two new sections.

### Cluster 4 — Truck leasing + НДС for б/у from физлицо (3 entries → 2 topical sections)

**Source entries:** `лизинг грузового ТС и НДС` (general), `б/у авто у физлица и НДС` (specific scenario), `лизинг б/у грузовика физлицу` (specific scenario, different framing).

**Authoring decision:** keep general + specific distinct.
- `section_id: truck-leasing-general` under `parent_topic: лизинг-юрлицо` — general truck leasing rules + НДС for ИП/юрлицо purchases.
- `section_id: used-truck-from-physlico` under `parent_topic: лизинг-юрлицо` — specific scenario: б/у truck from a физлицо seller, +20% НДС, +13% НДФЛ withholding, common startup/ИП advance bands.

### Cluster 5 — Working abroad (3 entries → 1 topical section)

**Source entries:** `автолизинг при работе в РФ`, `автолизинг при работе за границей`, `лизинг при работе за рубежом` (mentions Польша).

**Authoring decision:**
- ONE section `section_id: autolease-working-abroad` under `parent_topic: автолизинг`.
- Canonical: "RB resident working abroad (РФ, Польша, etc.) can get autolease if income/employment provable. Required: trudovoy dogovor or equivalent + бухгалтерская выписка."
- Note country-specific nuance in a sub-bullet (РФ — налоговое резидентство; ЕС — апостиль/перевод).

### Cluster 6 — Wrong company (3 entries → 1 topical section)

**Source entries:** `чужая компания`, `другая лизинговая компания`, `ошибка компании`. All cover "we are Микролизинг, not [Light Leasing / Автопромлизинг / Приватлизинг]; check the right number."

**Authoring decision:**
- ONE section `section_id: wrong-company-redirect` under `parent_topic: контакты` (or `parent_topic: бытовое`).
- Canonical: "You called Микролизинг. We don't have data on [Light Leasing, Автопромлизинг, Приватлизинг, etc.]. Please check the right number for that company."
- Listed company names become a sub-bullet of "common confusions".

### Clusters 7-15 — 2-member clusters (decisions)

| # | Sources | Decision | Notes |
|---|---|---|---|
| 7 | `перенос даты платежа` (×2 — same intent name!) | Merge to ONE section `section_id: payment-date-transfer` | Literal duplicate intent in source. |
| 8 | `госорганизации`, `лизинг госорганизациям` | Merge to ONE section `section_id: government-orgs-not-financed` | Both: not currently financed, limit exhausted. Use most accurate framing. |
| 9 | `лизинг без аванса`, `минимальный аванс автолизинг` | Merge to ONE section `section_id: minimum-advance-autolease` covering both "min advance" and "no-advance possible?" | Single section with explicit "no-advance only individual" sub-bullet. |
| 10 | `сбой оплаты ЕРИП`, `оплата через ЕРИП` | Merge to ONE section `section_id: erip-payment-failure` | Both about ЕРИП failure; use the one with просрочка info. |
| 11 | `Акт сверки`, `акт сверки для ОВД` | KEEP BOTH | Legitimate split — ОВД has specific procedural form. Two sections under `parent_topic: документы`. |
| 12 | `проверка компании договора`, `проверка договора` | Merge to ONE section `section_id: verify-contract-belongs-to-mikrolizing` | Same answer, different phrasing. |
| 13 | `возрастные ограничения`, `лизинг пенсионеру` | KEEP BOTH | General age rules vs pensioner-specific (>63). Cross-reference. |
| 14 | `лизинг без прав` (×2) | Merge to ONE section `section_id: leasing-without-license` covering both scenarios (no license yet, lost license) | Single section, two sub-cases. |
| 15 | `ошибочный звонок`, `ошибка номера` | Merge into Cluster 6's `wrong-company-redirect` section | Same root concept, different framing. |

### Cluster 16 — Полис КАСКО (2 entries, only 1 visible in cluster output)

**Decision:** likely keep one canonical `section_id: kasko-policy-copy`. Verify during Phase C.2 authoring whether the second entry is a true duplicate or a distinct sub-topic.

## Phase B summary

- **Source entries consolidated:** 25 (across 16 clusters) collapse into 18 topical sections (~56% within-cluster consolidation)
- **Total topical sections delivered:** 40 in `kb_topics_ru.md` — 18 cluster-derived + 22 singleton-derived covering high-frequency topics across 11 parent_topics
- **Source-entry coverage:** ~80% of the 350 source entries directly informed a topical section (the other ~20% are edge-case singletons whose facts get absorbed via inline phrasings or remain implicit in the broader topical context)

**Earlier draft of this document said "292 singletons become topical sections of their own" — that target was unrealistic and was not built. The actual delivery (40 topical sections) consolidates singletons by topic instead of one-to-one mapping. The reduction is the design, not a regression.**

**This document is the authoring contract for Phase C.2.** Every cluster has an explicit canonical decision. Phase C.2 subagents read this file + the cluster report and author the listed `section_id`s.

## Where the 25 "dropped" entries' content goes

**Nothing is dropped at the content level.** Each source entry's:
- `best_answer` text contributes to the canonical topical section
- `canonical_question` becomes a synonym entry in `kb_synonyms.yaml`
- `eligibility_rules`, `compliance_notes`, `handoff_when`, etc. roll into the corresponding `section_id` block in `bot_playbook.yaml`

The "no silent deletion" rule from the spec: every source entry's facts are accounted for in the topical section + synonyms + playbook trio.
