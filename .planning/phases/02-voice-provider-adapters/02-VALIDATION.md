---
phase: 2
slug: voice-provider-adapters
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-25
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | rag_demo_system/tests/ (existing test directory) |
| **Quick run command** | `cd rag_demo_system && python -m pytest tests/test_voice_adapters_official.py -x -q` |
| **Full suite command** | `cd rag_demo_system && python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd rag_demo_system && python -m pytest tests/test_voice_adapters_official.py -x -q`
- **After every plan wave:** Run `cd rag_demo_system && python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | TBD | TBD | VPROV-01 | unit | `pytest tests/test_voice_adapters_official.py -k qwen3_tts` | TBD | pending |
| TBD | TBD | TBD | VPROV-02 | unit | `pytest tests/test_voice_adapters_official.py -k qwen3_asr` | TBD | pending |
| TBD | TBD | TBD | VPROV-03 | unit | `pytest tests/test_voice_adapters_official.py -k voxtral` | TBD | pending |
| TBD | TBD | TBD | VPROV-04 | integration | `pytest tests/test_voice_adapters_official.py -x` | existing | pending |
| TBD | TBD | TBD | VPROV-05 | unit | `pytest tests/test_frontend_config_contract.py` | existing | pending |

*Status: pending / green / red / flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_voice_adapters_official.py` — add test stubs for qwen3_tts, qwen3_asr, voxtral
- [ ] Existing test infrastructure covers framework and fixtures

*Existing infrastructure covers framework requirements. New test cases needed for new providers.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Russian speech intelligibility (leasing terms) | VPROV-01 | Audio quality requires human ear | Play TTS output for domain terms, confirm intelligibility |
| Russian transcript accuracy | VPROV-02, VPROV-03 | Transcription quality requires human review | Speak Russian sentences, verify transcript accuracy |

---

## Validation Sign-Off

- [ ] All tasks have automated verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
