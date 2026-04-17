# Client Feedback Round — Deploy & Test Playbook

**Date:** 2026-04-16
**Branch:** `feature/voice-pipeline`
**Head commit at time of writing:** `1215396` (1 commit after `032c429`, the main integration commit)
**Revert point:** git tag `pre-client-feedback-round-2026-04-16` (commit `f64356c`)

Audience: you (voice pipeline operator) + your tester.

---

## 0. What shipped in this round

Seven specs were designed, five were implemented (one deferred, one is
a deliverable for the client's calculator team). Summary of concrete
behavior changes you will observe on the server after deploy:

| Area | Before | After |
|---|---|---|
| Whisper bot-name recognition | "Ксения" → Сеня / Синяя / Алексей | Bot biased toward "Ксения" (3× in prompt, final position) |
| Whisper leasing vocab | Missing "линейный", "нагрузка", "дифференцированный" | Present |
| Classifier latency | Queues behind main LLM | Runs on dedicated Qwen3-4B on port 8788 (when env var set) |
| Calculator defaults | Silent 30% advance / 36mo / BYN / annuity | No defaults: raises `IncompleteProfileError` on missing fields |
| Prepaid min/max | Hardcoded 10% floor | 0-40% range |
| Term max bot advertises | 36 mo (wrong) | 12-84 mo |
| Linear graph | Request ignored, always annuity | `type_schedule` extracted by SessionAgent and forwarded |
| Prepaid % vs amount | Only percent accepted | Either accepted; both returned |
| USD for physlico | Hard 404 error, no recovery | Auto-converted to BYN at 3:1 with explicit disclosure |
| EUR/RUB for physlico | Generic error | Clear rejection: "сейчас поддерживаются BYN и USD" |
| Bot re-asking ФЛ/ИП/ЮЛ | Every turn | Stored in `ClientProfile` session-wide |
| Semantic "stop" word | Ignored | Triggers `listen_mode`, bot goes silent |
| System prompt default language | "аванс 30% по умолчанию" etc. | Collection protocol + readback + change-confirmation |

## 1. Deploy on server

You have two supported deploy flows. Choose one based on whether you
want a minimal restart (1.1) or a full idempotent re-provision (1.2).

**Before you start:** export `HF_TOKEN` from your local credentials store
(it is intentionally NEVER committed to this repo). The token is stored
in your Claude memory file `reference_api_credentials.md` — copy the
value from there into your shell before running the commands below:

```bash
export HF_TOKEN=<paste-from-your-credentials-memory>
```

All `HF_TOKEN="$HF_TOKEN"` references below read it from that environment
variable.


### 1.1 Fast path — fetch, regenerate .env, restart

For a routine update that pulls this round's commits, rewrites `.env`
to pick up new variables (SESSIONAGENT_*, USD_BYN_RATE, turn_taking
knobs, revision pins), and restarts the supervisor stack:

```bash
ssh -i ~/.ssh/jarvislabs sesterce@<SERVER_IP>

cd /ephemeral/leasing/rag_demo_system
git pull origin feature/voice-pipeline

# Regenerate .env (picks up all new vars) then full clean restart.
# HF_TOKEN is needed ONCE to download Qwen3-4B-Instruct-FP8 on first boot.
HF_TOKEN="$HF_TOKEN" bash scripts/regenerate_env_and_restart.sh
```

**After first-boot HF download completes** (the 4B model is ~8 GB, 3-6 min
on a fast link), set `HF_HUB_OFFLINE=1` in `.env` to lock future starts
to cached weights only. This follows the project's pin-all-versions
rule — once the model is cached we never re-download.

### 1.2 Clean re-provision — idempotent, skips already-done work

Use this when you want to verify a clean install path end-to-end, or
after major changes to provisioning itself (new model, new Docker
images, new services). `provision_server.sh` is idempotent: it skips
apt packages that are installed, skips venvs that exist, skips cached
models. The one thing it **always** does is rewrite `.env` from the
current template, then restart the stack.

```bash
ssh -i ~/.ssh/jarvislabs sesterce@<SERVER_IP>

cd /ephemeral/leasing/rag_demo_system
git pull origin feature/voice-pipeline

# Re-run provision — downloads Qwen3-4B if missing, rewrites .env,
# restarts stack. Safe to run multiple times.
HF_TOKEN="$HF_TOKEN" bash scripts/provision_server.sh

# After provision completes, run smoke test (waits for services, verifies KB index).
bash scripts/smoke_test.sh

# Then deploy SIP accounts (Jambonz).
bash scripts/deploy_jambonz.sh
```

### 1.3 Optional: pin exact model revisions

If you want full reproducibility (weights frozen at specific commits),
look up the HF commit SHAs for the two models and export before running
provision/regenerate:

```bash
# Lookup on https://huggingface.co/Qwen/Qwen3-4B-Instruct-FP8/commits/main
# and https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8/commits/main
# Use the latest commits you have tested.

QWEN_MAIN_REVISION=<sha-for-35B-main-model> \
QWEN_SESSIONAGENT_REVISION=<sha-for-4B-sessionagent> \
HF_TOKEN="$HF_TOKEN" \
bash scripts/regenerate_env_and_restart.sh
```

Unset (empty) means "track main branch" — which is the current default
and what was tested during this round. Leave empty unless you've tested
a specific SHA and want to lock it.

### 1.4 Post-deploy: add back credentials

`provision_server.sh` rewrites `.env` from the template which has empty
`CALCULATOR_API_TOKEN`, `SMS_API_LOGIN`, `SMS_API_PASSWORD`. After
provision completes, append your real credentials:

```bash
cat >> /ephemeral/leasing/rag_demo_system/.env <<'EOF'
CALCULATOR_API_TOKEN='OrS0Xtm32f3o]T[96EAr'
SMS_API_LOGIN='Mikro_Lizing'
SMS_API_PASSWORD='4T5Nf879'
EOF

# Restart backend to pick them up.
./.venv/bin/supervisorctl -c scripts/supervisord.conf restart backend
```

(These are your production secrets — they are NEVER committed to git
and are stored only in your memory files + the server's `.env`.)

### 1.5 If GPU is too small for SessionAgent side-by-side

On GPUs below 75 GB the provision script disables SessionAgent
automatically (`STACK_SESSIONAGENT_CMD=""`). The backend then falls
back to the main LLM for classifier calls — latency wins are lost
but everything still works. Check the log line during provision:

```
GPU: ... -> main 0.50, sessionagent 0.00 (disabled); classifier uses main LLM
```

### 1.6 Verify the deploy

```bash
# On the server:
curl -s http://127.0.0.1:8787/health         # main LLM (Qwen 35B)
curl -s http://127.0.0.1:8788/health         # SessionAgent (Qwen 4B) — returns 200 if enabled
curl -s http://127.0.0.1:50002/health        # Whisper
curl -s http://127.0.0.1:50006/health        # Silero TTS
curl -s http://127.0.0.1:8000/api/health     # backend

# Live tailing:
tail -f .state/backend.log | grep -E 'Classifier|Profile|DirectTool|BARGE-IN|listen_mode|currency'
```

Look for these log markers on the very first call:
- `[Classifier] result: intent=TOOL hints={...} (XXms)` — XX should be < 400 ms p50.
- `[Profile] patched: {...}` — whenever SessionAgent extracts a new field.
- `[DirectTool] USD->BYN: ...` — if the client quoted USD.

## 2. Rollback

If anything goes wrong:

```bash
cd /ephemeral/leasing/rag_demo_system
git fetch --tags
git reset --hard pre-client-feedback-round-2026-04-16
bash scripts/regenerate_env_and_restart.sh
```

This reverts to commit `f64356c` (last known-good from 2026-04-16 before
this round).

## 3. Testing script — what to say, what to listen for

Use Zoiper or any SIP client. All voice tests start with the same
consent prompt (press 1 for "yes, record this call").

### 3.1 Baseline sanity

| # | You say | Expected bot response | Pass/Fail criteria |
|---|---|---|---|
| 1 | Press 1 to consent | "Спасибо за согласие! Меня зовут Ксения..." | Bot introduces herself |
| 2 | "Здравствуйте, меня зовут Сергей." | "Добрый день, Сергей! Чем могу помочь?" | Bot uses the name, doesn't re-ask |
| 3 | "Как меня зовут?" | "Сергей." or "Вас зовут Сергей" | Bot remembers name within session |

**Log check:** `grep '[Profile] patched' .state/backend.log` should show `name=Сергей`.

### 3.2 Calculator — full flow

| # | You say | Expected bot response |
|---|---|---|
| 4 | "Хочу рассчитать лизинг на легковой автомобиль." | Bot asks for the next missing field (type of client, or cost). Does NOT invoke calculator yet (no defaults). |
| 5 | "Я физлицо, машина новая, 70 тысяч рублей." | Bot records those fields, asks for remaining (срок, аванс, график). |
| 6 | "Срок 84 месяца, аванс 20 процентов, аннуитет." | Bot does a read-back: "Итак, легковой автомобиль, новый, 70 тысяч рублей, физлицо, аванс 20 процентов, срок 84 месяца, аннуитетный график. Всё верно?" |
| 7 | "Да, всё верно." | Bot invokes calculator, responds with aванс сумма + monthly + total + updorожание |
| 8 | "Сделай линейный график." | Bot asks to confirm: "Меняю тип графика на линейный, всё верно?" |
| 9 | "Да." | Bot recalculates with `type_schedule="1"` (verify via log) |

**Log check:**
```
grep 'DirectTool' .state/backend.log | tail -20
```
Should show two calculator calls, the second with `"type_schedule":"1"`.

### 3.3 USD + Физ лицо → auto-conversion

| # | You say | Expected bot response |
|---|---|---|
| 10 | "Хочу посчитать легковой автомобиль за двадцать четыре тысячи триста долларов." | Bot: "По курсу 3 рубля за доллар это 72 900 рублей. Продолжаем в рублях?" |
| 11 | "Да." | Calculator fires with `cost=72900, currency=BYN` (verify via log) |

**Log check:** `grep 'USD->BYN' .state/backend.log`.

### 3.4 EUR / RUB → polite rejection

| # | You say | Expected bot response |
|---|---|---|
| 12 | "Я физлицо, посчитай машину за двадцать пять тысяч евро." | "Для физлиц сейчас поддерживаются расчёты в белорусских рублях и в долларах. В какой валюте стоимость?" |

**Log check:** `grep 'currency_policy: reject' .state/backend.log`.

### 3.5 Relaxed advance + term ranges

| # | You say | Expected bot response |
|---|---|---|
| 13 | "Хочу машину без аванса на 84 месяца." | Bot should NOT say "минимум 10%" (limit was removed). Calculator may return 404 if API rules block it, but the bot should parrot the API's message, not its own hardcoded "10% minimum". |
| 14 | "На 60 месяцев с авансом 40 процентов." | Bot accepts this; classifier prompt allows 0-40% range. |

**Log check:** prepaid 0 or 40 should reach calculator, not get blocked upstream.

### 3.6 Semantic stop command

| # | You say | Expected bot response |
|---|---|---|
| 15 | While bot is mid-sentence, say: "Стоп." | TTS cancels mid-word. Bot goes SILENT. No response is generated. |
| 16 | After ~3 seconds of silence | Bot emits: "Слушаю Вас." (auto-exit from listen_mode) |
| 17 | "Хорошо, продолжим." | Normal conversation resumes |

**Log check:** `grep 'listen_mode\|is_stop_request' .state/backend.log`.

### 3.7 Whisper improvements

| # | You say | Expected transcription in log |
|---|---|---|
| 18 | "Ксения, что такое нагрузка?" | Log shows "Ксения" (not "Сеня"/"Синяя"), "нагрузка" correctly |
| 19 | "Сделай линейный график." | Log shows "линейный" (not corruption) |
| 20 | "Какой будет аннуитетный платёж?" | Log shows "аннуитетный" |

**Log check:** `grep 'transcription\|whisper' .state/backend.log`.

## 4. Self-improvement & KB audit

### 4.1 Self-improvement pipeline (extended this round)

Every SIP call automatically saves:
- Full transcript → `.state/transcripts/<session_id>.json`
- LLM-generated quality report → `.state/analysis/session_reports.jsonl`

The analyzer prompt (`backend/session_analyzer.py`) was extended this round
with new dimensions that specifically measure this round's fixes:

| Dimension | What it tracks |
|---|---|
| `profile_hygiene.repeat_asks` | Fields the bot re-asked after client already stated them |
| `stop_command_events` | Times client said стоп/подожди/помолчи + whether bot respected |
| `tool_calls.readback_before_first_calc` | Did bot read back all params before first calc? |
| `tool_calls.change_confirmed_before_recalc` | Did bot confirm before recalc on field change? |
| `tool_calls.usd_to_byn_conversion_done` | Was USD auto-converted for физлицо? |
| `tool_calls.eur_rub_rejected_cleanly` | Was EUR/RUB rejected with clear message? |
| `tool_calls.type_schedule_forwarded_correctly` | Did linear-graph request reach calculator? |
| `tool_calls.linear_requests_count` / `linear_successes_count` | Linear-graph success rate |
| `defaults_assumed` | Cases where bot stated a default without confirmation |

**Aggregation command (run weekly):**

```bash
cd /ephemeral/leasing/rag_demo_system
python scripts/kb_gap_report.py
```

This prints:
- KB gaps (topics clients asked about, no KB answer) ranked by frequency
- Recurring issues (severity-tagged)
- Operational signals: readback compliance %, change-confirm compliance %,
  USD conversion count, EUR rejection count, linear-graph success rate,
  stop-command respect %, profile-hygiene repeat-ask counts per field,
  defaults-assumed incidents

**Target metrics after this round:**
- Readback compliance: ≥ 80% of calc calls should have a read-back
- Change-confirm compliance: ≥ 90% of recalcs should have a single-field confirm
- Linear-graph success rate: ≥ 90% (was 0% pre-deploy)
- Stop-command respected: ≥ 85%
- Profile repeat-asks for client_type: ≤ 1 per session (was 3-7 per session)
- Defaults-assumed incidents: ≤ 2 per session (was high before)

### 4.2 KB audit (done this round)

Full report: [`docs/kb-audit-report-2026-04-16.md`](kb-audit-report-2026-04-16.md)

**Summary of findings:**
- 3 true gaps: "нагрузка" as financial term, linear/annuity explanation, Mogilev office details
- 8 stale blocks: "минимальный аванс 30%", stale max-term numbers, "аннуитетный, классический и т.п." vague language
- 9 partial/covered: most client questions have some KB coverage but need minor enrichment

**Next step:** review the audit report with the client, apply proposed patches
to `knowledge_base/kb_faq_ru_v2.md`, re-index:

```bash
# After patches land:
python scripts/index_kb.py
# Verify retrieval on the added content:
python scripts/voice_lab.py --query "что такое нагрузка"
python scripts/voice_lab.py --query "линейный график"
```

### 4.3 Version pinning

All Python deps pinned via pip (`vllm==0.19.0`, `faster-whisper==1.2.1`,
`silero==0.5.5`). Docker images for Jambonz all pinned to `0.9.6`.

Model weights can now be pinned to specific HF commit SHAs via env vars
`QWEN_MAIN_REVISION` and `QWEN_SESSIONAGENT_REVISION` (see section 1.3).
Leave empty to track `main`, which is the tested default.

---

## 5. Known limitations (for follow-up session)

Some items from the plans were scoped down to ship in this round.
These need a follow-up session to complete:

1. **Full profile-gated read-back state machine**: currently the LLM
   uses the system prompt instructions to do read-back. For fully
   deterministic flow (code-level gate), the `_direct_tool_from_profile`
   helper needs to be wired to the TTS path. This is a correctness
   improvement, not a functional bug.

2. **Pre-response hold + VAD silence bump** (Spec 4 partial):
   `VAD_SILENCE_MS` default is still 500ms (not 700). Pre-response
   hold (300ms) not yet added. This means clients can still be cut
   off mid-sentence on long natural pauses. Workaround: ask client to
   keep sentences compact.

3. **Transcript questions KB audit** (Spec 6): deferred. Requires you
   to provide the list of client questions from the transcripts so I
   can diff against the KB and flag gaps.

4. **Russian calculator API spec (Spec 7)**: already delivered at
   `docs/calculator-api-production-spec-ru.md`. You can forward this
   to the client's developer team.

## 6. Where the artifacts live

- Specs: `docs/superpowers/specs/2026-04-16-*.md` (6 files + index)
- Plans: `docs/superpowers/plans/2026-04-16-*.md` (5 files)
- Russian production spec: `docs/calculator-api-production-spec-ru.md`
- Testing playbook: this file
- Memory files (global):
  - `project_calculator_production_backlog.md`
  - `project_stt_v2_roadmap.md`
- PROJECT_LOG.md entry: 2026-04-16 (later)

## 7. Quick reference commands

```bash
# Tail backend log for this round's markers
tail -f .state/backend.log | grep -E 'Classifier|Profile|DirectTool|listen_mode|BARGE-IN|USD'

# Verify SessionAgent is actually serving classifier traffic
grep '\[Classifier\]' .state/backend.log | tail -5
# Look at latency_ms field; should be < 400ms when using 8788

# Verify no default parameters leaking through (sanity check)
grep 'defaulted' .state/backend.log
# Should be empty or only show []

# Verify currency policy is running
grep -E 'USD->BYN|currency_policy' .state/backend.log

# Run the weekly KB gap + operational-metrics aggregation
python scripts/kb_gap_report.py

# Inspect the most recent session quality report
tail -1 .state/analysis/session_reports.jsonl | python3 -m json.tool

# Quick operational counters from the last 24h of calls (one-liner)
python3 -c "
import json
from pathlib import Path
reports = Path('.state/analysis/session_reports.jsonl').read_text().splitlines()
readback_ok = readback_skip = 0
linear_asked = linear_ok = 0
stop_ok = stop_skip = 0
for line in reports:
    try: r = json.loads(line)
    except: continue
    tc = r.get('tool_calls') or {}
    if tc.get('readback_before_first_calc') is True: readback_ok += 1
    elif tc.get('readback_before_first_calc') is False: readback_skip += 1
    linear_asked += int(tc.get('linear_requests_count') or 0)
    linear_ok += int(tc.get('linear_successes_count') or 0)
    for e in (r.get('stop_command_events') or []):
        if e.get('bot_respected'): stop_ok += 1
        else: stop_skip += 1
print(f'readback: {readback_ok}/{readback_ok+readback_skip}')
print(f'linear: {linear_ok}/{linear_asked}')
print(f'stop: {stop_ok}/{stop_ok+stop_skip}')
"
```
