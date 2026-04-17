# Client Feedback Round — Verbal Test Playbook

**Date:** 2026-04-16
**Branch:** `feature/voice-pipeline`
**Revert point:** git tag `pre-client-feedback-round-2026-04-16`

Audience: tester on a SIP client (Zoiper / softphone).

Prerequisites before picking up this playbook:

- `bash scripts/smoke_test.sh` → PASSED
- `bash scripts/terminal_tests.sh` → all 8 PASS

Those scripts cover everything that can be verified from the shell. This
document is **voice only**: real calls, real audio, real barge-in.

---

## How to use this playbook

1. Start a call, go through the scenarios in order. Sections 3.1–3.7 are
   independent but build on each other — keep the same SIP session where
   possible so `ClientProfile` accumulates realistically.
2. For every turn, compare the bot's actual response against "Expected".
   Don't abort on mismatches — note them and keep going. We want the
   whole run's signal, not a short-circuit.
3. Write comments directly on the audio (timestamp + issue) or in a
   notes file. You will send me that, plus the transcript and logs that
   the final command in section 4 bundles up.

---

## 1. Baseline sanity

| # | You say | Expected bot response | Why this matters |
|---|---|---|---|
| 1 | Press 1 to consent | "Спасибо за согласие! Меня зовут Ксения..." | Consent flow + bot introduces itself |
| 2 | "Здравствуйте, меня зовут Сергей." | "Добрый день, Сергей! Чем могу помочь?" | Name captured into ClientProfile on first turn |
| 3 | "Как меня зовут?" | "Сергей." | Memory across turns (previously re-asked) |
| 4 | "Ты что, не слышала?" (any no-info filler) | Bot politely deflects / asks clarifying question | No hallucinated answer, no spurious tool call |

## 2. Calculator — full flow with read-back + change-confirm

Same session. The bot must collect all fields before calling the API,
read them back, confirm, then calculate. If it calls the calculator
before confirmation — **bug**.

| # | You say | Expected bot response |
|---|---|---|
| 5 | "Хочу рассчитать лизинг на легковой автомобиль." | Asks the next missing field (type of client, or стоимость). Does **not** run the calculator yet. |
| 6 | "Я физлицо, машина новая, 70 тысяч рублей." | Records those. Asks for срок / аванс / тип графика (whichever are missing). |
| 7 | "Срок 84 месяца, аванс 20 процентов, аннуитет." | **Read-back**: "Итак: легковой автомобиль, новый, 70 тысяч рублей, физлицо, аванс 20%, срок 84 месяца, аннуитет. Всё верно?" |
| 8 | "Да, всё верно." | Runs calculator. Gives: сумма аванса, ежемесячный платёж, итоговая сумма, удорожание. |
| 9 | "Сделай линейный график." | **Change-confirm**: "Меняю тип графика на линейный, всё остальное оставляю. Всё верно?" (single-field change, doesn't re-ask everything) |
| 10 | "Да." | Recalculates. Answer differs from turn 8 (linear ≠ annuity). |

**Red flags to note:**
- Bot runs calculator before turn 7 read-back → regression
- Bot re-asks client_type/subject/cost after turn 6 → profile hygiene bug
- Turn 9 skips confirmation and recalculates immediately → change-confirm missing
- Turn 10 answer identical to turn 8 → `type_schedule` not forwarded

## 3. Currency policy

New session (hang up + call back, or tell bot "начнём заново").

| # | You say | Expected bot response |
|---|---|---|
| 11 | "Я физлицо, хочу легковой автомобиль за двадцать четыре тысячи триста долларов." | Bot: "По курсу 3 рубля за доллар это 72 900 рублей. Продолжаем в рублях?" (explicit disclosure of conversion) |
| 12 | "Да." | Collects remaining fields, reads back with `cost=72900, currency=BYN`, then calculates. |
| 13 | (new session) "Я физлицо, машина за двадцать пять тысяч евро." | "Для физлиц сейчас поддерживаются расчёты в белорусских рублях и в долларах. В какой валюте указать стоимость?" |
| 14 | "Тогда сорок тысяч рублей." | Continues normally in BYN. |

**Red flags:**
- USD silently converted without telling user → disclosure bug
- EUR accepted as a valid currency → rejection missing
- Bot says "USD не поддерживается" for физлицо → USD conversion missing

## 4. Relaxed prepaid + term ranges

New session.

| # | You say | Expected bot response |
|---|---|---|
| 15 | "Хочу машину без аванса на 84 месяца." | Bot should **not** say "минимум 10% / 30%". It may echo back the API's decision if the calculator refuses, but it must not block on its own. |
| 16 | "Аванс сорок процентов, срок 60 месяцев." | Accepted. Classifier allows 0–40%. |
| 17 | "Можно срок 96 месяцев?" | Bot says max is 84 months. Does NOT say 36 (old stale value). |

## 5. Semantic stop command (barge-in + listen_mode)

Same session.

| # | You say | Expected bot response |
|---|---|---|
| 18 | Ask a long question so the bot starts a multi-sentence answer. Mid-answer, say: **"Стоп."** | TTS cuts off within ~300 ms. Bot goes SILENT — no new reply is generated. |
| 19 | Stay silent for ~3 seconds | Bot emits: "Слушаю Вас." (auto-exit from listen_mode on timeout) |
| 20 | "Хорошо, продолжим про лизинг." | Normal conversation resumes |
| 21 | Again mid-response, say: **"Помолчи, пожалуйста."** | Same behavior as turn 18 (semantic stop, not literal "стоп") |

**Red flags:**
- Bot keeps talking past "Стоп" → barge-in broken
- Bot answers "Стоп" as if it were a question → intent classifier missing |
- Bot stays silent forever → listen_mode timeout broken

## 6. Whisper recognition quality

Any session. These are spot-checks for STT vocabulary we added this round.
Listen for the bot to echo the transcribed word correctly.

| # | You say | What we're checking |
|---|---|---|
| 22 | "Ксения, что такое нагрузка?" | Bot heard "Ксения" (not "Сеня"/"Синяя") AND "нагрузка" (not "на грузовик" or similar) |
| 23 | "Сделай линейный график." | "линейный" transcribed correctly |
| 24 | "Какой будет аннуитетный платёж?" | "аннуитетный" transcribed correctly |
| 25 | "Я ипэшник, работаю в сфере IT." | "ипэшник" transcribed, bot classifies as `client_type='ИП'` |

## 7. Address / abbreviation read-out

Any session. Tests TTS abbreviation expansion (ул. → улица, etc.).

| # | You say | Expected TTS output |
|---|---|---|
| 26 | "Где у вас офис в Минске?" | Bot reads full words: "проспект Победителей, 57" (not "пр-т") |
| 27 | "А в Могилёве?" | "улица ..., дом ..." — not "ул. ..., д. ..." |
| 28 | "Какой у вас телефон?" | "телефон +375..." — not "тел. +375..." |

---

## Collect everything for review

When the call run is done, run this on the server to bundle logs + the
session's transcript + the analyzer report into one tarball you can send me:

```bash
cd /ephemeral/leasing/rag_demo_system && \
  TS=$(date -u +%Y%m%d-%H%M%S) && \
  OUT=/tmp/client-feedback-run-${TS} && \
  mkdir -p "$OUT" && \
  cp .state/backend.log "$OUT/backend.log" 2>/dev/null; \
  cp -r .state/transcripts "$OUT/transcripts" 2>/dev/null; \
  cp .state/analysis/session_reports.jsonl "$OUT/session_reports.jsonl" 2>/dev/null; \
  cp .env "$OUT/env.txt" 2>/dev/null && \
  sed -i "s/\(TOKEN\|PASSWORD\|LOGIN\)=.*/\1=<redacted>/g" "$OUT/env.txt"; \
  grep -E 'Classifier|\[Profile\]|DirectTool|listen_mode|is_stop_request|USD->BYN|currency_policy|BARGE-IN' .state/backend.log > "$OUT/markers.log" 2>/dev/null; \
  ./.venv/bin/python scripts/kb_gap_report.py > "$OUT/kb_gap_report.txt" 2>&1; \
  tar czf "${OUT}.tar.gz" -C /tmp "$(basename $OUT)" && \
  echo "=== bundle ready: ${OUT}.tar.gz ===" && \
  ls -lh "${OUT}.tar.gz"
```

Then send me:
1. `${OUT}.tar.gz` — the bundle
2. Your notes file or timestamped audio comments
3. The recording itself if you have it

I'll diff expected vs actual for each scenario, cross-check against
`markers.log` for the events we instrumented, and map any failures back
to the specs in `docs/superpowers/specs/2026-04-16-*.md`.

---

## If the stack breaks mid-test

Quick rollback to the pre-round tag:

```bash
cd /ephemeral/leasing/rag_demo_system
git fetch --tags
git reset --hard pre-client-feedback-round-2026-04-16
bash scripts/regenerate_env_and_restart.sh
```

This reverts to commit `f64356c` (last known-good before this round).
