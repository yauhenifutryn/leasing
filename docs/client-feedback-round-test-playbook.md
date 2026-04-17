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

### 1.1 If the server already has the voice pipeline running

SSH into the server and run these commands:

```bash
ssh -i ~/.ssh/jarvislabs sesterce@<SERVER_IP>

cd /ephemeral/leasing/rag_demo_system
git pull

# Regenerates .env with new SESSIONAGENT_*, USD_BYN_RATE, and 
# launch command for the second vLLM instance. Then restarts the stack.
bash scripts/regenerate_env_and_restart.sh
```

First restart after this update will **download Qwen3-4B-Instruct-FP8
(~8 GB)** from HuggingFace. Expect the `sessionagent` service to take
5-15 minutes on first boot. The main LLM (Qwen3.5-35B) is already
cached on disk and will start in ~2 min.

If HuggingFace is configured in offline mode on your server, you may
need to either:
- Temporarily set `HF_HUB_OFFLINE=0` in `.env`, restart, wait for
  download, then restore `HF_HUB_OFFLINE=1`, OR
- Pre-download with `huggingface-cli download Qwen/Qwen3-4B-Instruct-FP8
  --local-dir /ephemeral/models/Qwen3-4B-Instruct-FP8`.

### 1.2 If GPU is too small for SessionAgent side-by-side

On GPUs below 75 GB the provision script disables SessionAgent
automatically (`STACK_SESSIONAGENT_CMD=""`). The backend then falls
back to the main LLM for classifier calls — latency wins are lost
but everything still works. Check the log line during provision:

```
GPU: ... -> main 0.50, sessionagent 0.00 (disabled)
```

### 1.3 Verify the deploy

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

## 4. Known limitations (for follow-up session)

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

## 5. Where the artifacts live

- Specs: `docs/superpowers/specs/2026-04-16-*.md` (6 files + index)
- Plans: `docs/superpowers/plans/2026-04-16-*.md` (5 files)
- Russian production spec: `docs/calculator-api-production-spec-ru.md`
- Testing playbook: this file
- Memory files (global):
  - `project_calculator_production_backlog.md`
  - `project_stt_v2_roadmap.md`
- PROJECT_LOG.md entry: 2026-04-16 (later)

## 6. Quick reference commands

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
```
