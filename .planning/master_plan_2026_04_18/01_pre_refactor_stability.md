# Section 1 — Pre-refactor stability (quick client wins)

**Status**: pending
**Prereqs**: none
**Estimated effort**: ~1 day
**Base commit**: `mvp-2026-04-18` (`a20aa4f`)

## Goal

Ship the small, isolated fixes that directly address client feedback without touching the orchestrator architecture. Each fix is ≤3 hours, independently revertible, and produces a visible improvement in the next client test.

## Required memories (retrieve before starting)

- `project_mvp_complete_2026_04_18.md` — current state, commit chain
- `project_orchestrator_refactor_pending.md` — context on why we're not refactoring yet
- `project_client_feedback_fixes.md` — prior client feedback history
- `project_currency_semantics.md` — BYN vs RUB rule; informs USD disclosure
- `feedback_no_postprocessing_hacks.md` — NEVER regex-patch LLM output
- `feedback_universal_fixes.md` — fixes must be systemic, not special-case
- `project_barge_in_plan.md` + `project_barge_in_remaining.md` — VAD threshold context

## Primary skill

`superpowers:systematic-debugging` for each fix (each is a targeted behavioral issue). `superpowers:test-driven-development` for new test cases. `superpowers:verification-before-completion` before claiming each fix done.

## Scope — four isolated fixes

### Fix 1.1 — Deterministic numeric rendering (LLM must never produce numbers)

**Problem**: LLM-generated summaries mix up USD/BYN ("стоимость 20 тысяч рублей" when user said $20k), fabricate monthly payments when calc didn't fire, produce inconsistent readbacks.

**Scope**:
- Audit every LLM code path in `backend/app.py` that has access to profile numbers or calc results
- Replace direct LLM interpolation with deterministic `render_profile_summary(profile)` / `render_calc_result(result)` helpers in `backend/profile_prompts.py`
- LLM's job: add tone/empathy markers, never produce or transform numbers
- Add a lint check (grep script) that fails if LLM prompt strings include `{cost}`, `{prepaid}`, `{term}` etc. directly

**Done criteria**:
- `grep -rE '\{(cost|prepaid|term|payment_min|advance_sum)' backend/*.py | grep -v profile_prompts.py` returns nothing
- New test `test_llm_never_sees_numeric_templates` passes
- Live call with calc fail produces "параметры не подходят" without fabricated numbers

### Fix 1.2 — USD always disclosed with BYN equivalent

**Problem**: For Физлицо + USD, code converts 3:1 internally and stores BYN. Readback says bare "20 тысяч рублей" — user is confused. Client: "doesn't say currency in USD case."

**Scope**:
- In `ClientProfile`, add fields `original_cost` + `original_currency` alongside the stored (BYN) cost
- When USD→BYN conversion happens in `app.py` direct-call path, capture original values before conversion
- `build_readback_text` always renders as `"20 тысяч долларов, или 60 тысяч рублей по курсу 3:1"` when original_currency=USD
- `render_calc_result` same pattern
- SMS body: same pattern
- Classifier prompt: acknowledge "USD автоматически конвертируется в BYN 3:1" once in the prompt so LLM knows

**Done criteria**:
- New test `test_usd_disclosure_in_readback` passes (profile with original_currency=USD → readback contains both amounts)
- Live call with "$20k машина" → bot readback says "20 тысяч долларов (или 60 тысяч рублей)"
- SMS body includes both

### Fix 1.3 — "б/у" robustness

**Problem**: Classifier misses "бу", "б/у-корабль", "бэу", "с пробегом", "не новый". Client: "не понимает слово б/у."

**Scope**:
- Expand `_CONDITION_NEW_CUE_RE` in `profile_hygiene.py`: add `бэу`, `б-у`, `пробег\w*`, `не\s+нов\w+`, `старый`
- Add `condition_new=0` examples to classifier prompt: "бэу/с пробегом/подержанный/не новый/б-у-корабль → 0"
- Add phonetic variants to Whisper `initial_prompt`: "бу, бэу, б/у, подержанный, с пробегом"
- Test cases covering every variant

**Done criteria**:
- New test file `test_condition_new_variants.py` with 8+ variants all → condition_new=0
- Live call test: user says "бэу мотоцикл" → profile.condition_new=0 after one turn

### Fix 1.4 — Barge-in bump (quick interim only)

**Problem**: VAD silence threshold 500ms too aggressive. User pauses to think → bot jumps in. Client complaint.

**Scope** (MVP only — full natural turn-taking is Section 4):
- Change `VAD_SILENCE_MS` default from 500 to 900 in `.env.example` and restart script
- Document the tradeoff in README: +400ms perceived latency in exchange for thought-pause tolerance
- Do NOT touch adaptive endpointing or fillers here — that's Section 4

**Done criteria**:
- Config changed, server restart uses new value
- Live test: 1.5s pause mid-sentence doesn't trigger bot response

## Checkpoints (update in CHECKPOINTS.md as work progresses)

- CP-1.1: Fix 1.1 shipped + live-verified
- CP-1.2: Fix 1.2 shipped + live-verified
- CP-1.3: Fix 1.3 shipped + live-verified
- CP-1.4: Fix 1.4 shipped + live-verified
- CP-1.5: Section closed out — commit tagged `stability-v1`, client complaints list updated

## Suggested commit cadence

One commit per fix. Each commit references its client-complaint mapping:

```
fix(voice): 1.1 — deterministic numeric rendering (client: "LLM hallucinates numbers")
fix(voice): 1.2 — USD disclosure with BYN conversion (client: "doesn't say currency")
fix(voice): 1.3 — б/у variant recognition (client: "doesn't understand б/у")
fix(voice): 1.4 — VAD silence threshold 500→900ms (client: "kicks in when I pause")
```

## Rollback

```bash
git reset --hard mvp-2026-04-18
git push --force-with-lease origin feature/voice-pipeline
```

## Handoff to next section

After CP-1.5, update `CHECKPOINTS.md` with commit SHAs and move to Section 2. Section 2 does NOT depend on Section 1 — they can run in parallel in separate branches if desired.
