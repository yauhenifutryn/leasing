# Section 3.5 — Calculator Funnel API Integration

**Status**: pending
**Prereqs**: Section 3 fully closed (CP-3.4..CP-3.7 done, `refactor-v1` tag pushed, legacy 5-gate block deleted, Section 3 merged into `feature/voice-pipeline`)
**Effort**: ~2–3 days
**Priority**: HIGH — ships real bug fixes (drift bugs A/B/C in spec §1), unblocks Section 4 live-test signal quality
**Spec**: `docs/superpowers/specs/2026-04-24-calculator-funnel-design.md`
**Branch**: `feature/section-3.5-calculator-funnel` off post-merge `feature/voice-pipeline`
**Tag on completion**: `calc-funnel-v1`

## Why this section exists

The Mikro Leasing calculator API exposes five endpoints. We call one (`/calculate/`). The other four (`/subjects/`, `/currencies/`, `/ranges/`, `/terms/`) give us the rule matrix and per-scenario constraints live. Our voice bot has been duplicating the rule matrix in frozen sets, hardcoded constants, and subject maps — all of which are drifting from the real rules.

Three concrete drift bugs verified against prod API on 2026-04-24:

- `_COMMERCIAL_SUBJECTS` misses Оборудование and Недвижимость → физлицо can request equipment leasing, `/calculate/` 404s with no actionable steer.
- `COST_MIN=1, COST_MAX=100_000_000` vs real USD car range `2000..300000`.
- Hardcoded `TERM_MAX=84` hides the per-prepaid-tier cap ("10% prepaid → max 60 mo"). Combinations like `term=72 + prepaid=10` look valid to us but 404 at calc time.

Fix: query the API progressively as the client talks. Pre-validate before every readback. Fall back to today's permissive behavior on any fetch failure.

## Required memories

- `project_calculator_production_backlog.md` — prior deferred work, now partially unblocked
- `project_currency_converter_plan.md` — NBRB rate; deferred in this section (§7 of spec)
- `project_tools_remaining_issues.md` — current tool-use orchestration state
- `project_section_3_phase_D_complete.md` — Section 3 handoff + `apply_turn` architecture
- `reference_api_credentials.md` — calculator API base URL + token (for tests against sandbox)

## Primary skill

`superpowers:writing-plans` (create the implementation plan from the spec)
`superpowers:executing-plans` (execute the plan in a separate session)

## Checkpoints

- CP-3.5.1 — `backend/calc_limits.py` with `subjects`, `currencies`, `ranges` fetchers + 24h process cache + 20+ unit tests green
- CP-3.5.2 — `terms` fetcher + `disagreement` logic + conflict-suggestion generation + unit tests green
- CP-3.5.3 — `EmitConstraintConflict` TurnAction variant + `CalcLimitsAdapter` + execute_action EmitReadback handler upgraded + flag-gated (`CALCULATOR_FUNNEL_ENABLED`, default `0`) + all existing tests green on both flag values
- CP-3.5.4 — LLM-side rule deletion per spec §7 (COMMERCIAL_SUBJECTS, range constants, _SUBJECT_MAP, _VALID_SUBJECTS/CURRENCIES; `_CLIENT_TYPE_MAP` ИП→Юр and 3:1 USD hack KEPT — deferred)
- CP-3.5.5 — Live SIP regression green on `CALCULATOR_FUNNEL_ENABLED=1`: happy path + combo-conflict scenario both pass
- CP-3.5.6 — Flag and legacy branches removed; `calc-funnel-v1` tag pushed; section closed

## Rollback

- **Intra-section**: `CALCULATOR_FUNNEL_ENABLED=0` on server env, restart. Legacy rule branches still resident through CP-3.5.5.
- **Section revert (within 7 days)**: `git reset --hard refactor-v1`.
- **Nuclear**: `git reset --hard mvp-2026-04-18`.

## Handoff prompt for fresh session

> Read `.planning/master_plan_2026_04_18/INDEX.md` then `.planning/master_plan_2026_04_18/CHECKPOINTS.md`. Resume Section 3.5 (calculator funnel API integration). Spec at `docs/superpowers/specs/2026-04-24-calculator-funnel-design.md`. Use the memories and skills listed in `03_5_calculator_funnel.md`. Prereq: Section 3 must be fully closed (`refactor-v1` tag pushed, merged to `feature/voice-pipeline`).
