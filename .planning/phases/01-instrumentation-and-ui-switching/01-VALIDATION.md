---
phase: 1
slug: instrumentation-and-ui-switching
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-25
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `python -m pytest tests/ -x -q` |
| **Full suite command** | `python -m pytest tests/ -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/ -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | INST-01 | unit | `python -m pytest tests/test_instrumentation.py -k test_log_fields` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | INST-02 | unit | `python -m pytest tests/test_instrumentation.py -k test_primary_kpi` | ❌ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | INST-03 | unit | `python -m pytest tests/test_instrumentation.py -k test_stack_id` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | SWITCH-01 | unit | `python -m pytest tests/test_voice_session.py -k test_selectors` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | SWITCH-02 | unit | `python -m pytest tests/test_voice_session.py -k test_live_switch` | ❌ W0 | ⬜ pending |
| 1-02-03 | 02 | 1 | SWITCH-03 | unit | `python -m pytest tests/test_voice_session.py -k test_stack_id_auto` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_instrumentation.py` — stubs for INST-01, INST-02, INST-03
- [ ] `tests/test_voice_session.py` — stubs for SWITCH-01, SWITCH-02, SWITCH-03
- [ ] `tests/conftest.py` — shared fixtures (mock VoiceSession, StateStore)
- [ ] pytest install — if not already available

*Existing infrastructure covers framework; test files need creation.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| UI selectors render correctly | SWITCH-01 | Browser DOM interaction | Open frontend, verify dropdowns for RAG backend, brain model, STT, TTS appear and are clickable |
| Live switching takes effect without restart | SWITCH-02 | Requires active WebSocket session | Switch a selector mid-session, verify next voice turn uses new config |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
