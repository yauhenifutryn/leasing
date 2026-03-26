---
phase: 4
slug: qwen3-omni-hybrid
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | rag_demo_system/tests/conftest.py |
| **Quick run command** | `cd rag_demo_system && python -m pytest tests/test_omni_adapter.py -x -q` |
| **Full suite command** | `cd rag_demo_system && python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd rag_demo_system && python -m pytest tests/test_omni_adapter.py -x -q`
- **After every plan wave:** Run `cd rag_demo_system && python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|

*Status: pending / green / red / flaky*

*Filled during planning — planner maps tasks to verification commands.*

---

## Wave 0 Requirements

- [ ] `rag_demo_system/tests/test_omni_adapter.py` — stubs for OMNI-01, OMNI-02, OMNI-03
- [ ] Sidecar contract tests — /chat and /health endpoint stubs

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Russian audio output quality | OMNI-01 | Subjective quality assessment requires human listener | Play Omni audio response, verify Russian is intelligible and not code-switching |
| Out-of-scope refusal quality | OMNI-02 | Refusal phrasing quality is subjective | Ask out-of-scope question via UI, verify response is a polite refusal in Russian |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
