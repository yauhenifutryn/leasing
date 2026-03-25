---
phase: 3
slug: brain-upgrade-and-benchmark-framework
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-25
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | rag_demo_system/tests/ (existing test directory) |
| **Quick run command** | `cd rag_demo_system && python -m pytest tests/test_voice_session.py tests/test_instrumentation.py -x -q` |
| **Full suite command** | `cd rag_demo_system && python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd rag_demo_system && python -m pytest tests/test_voice_session.py tests/test_instrumentation.py -x -q`
- **After every plan wave:** Run `cd rag_demo_system && python -m pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | BRAIN-01 | unit | `pytest tests/test_voice_session.py -x -q` | TBD | ⬜ pending |
| TBD | 01 | 1 | BENCH-01 | unit | `pytest tests/test_benchmark_fixture.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | 02 | 1 | BENCH-02 | unit | `pytest tests/test_benchmark_runner.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | 02 | 1 | BENCH-03 | unit | `pytest tests/test_benchmark_runner.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | 03 | 2 | BENCH-04 | unit | `pytest tests/test_benchmark_compare.py -x -q` | ❌ W0 | ⬜ pending |
| TBD | 01 | 1 | DEPLOY-01 | file | `ls rag_demo_system/.env.bench.*` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `rag_demo_system/tests/test_benchmark_fixture.py` — stubs for BENCH-01 (fixture validation)
- [ ] `rag_demo_system/tests/test_benchmark_runner.py` — stubs for BENCH-02, BENCH-03 (runner output format)
- [ ] `rag_demo_system/tests/test_benchmark_compare.py` — stubs for BENCH-04 (comparison script)

*Existing infrastructure covers BRAIN-01 (voice_session tests exist) and DEPLOY-01 (file existence check).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Brain model actually routes inference to correct vLLM model | BRAIN-01 | Requires live vLLM server | Start vLLM with Qwen3.5-35B-A3B, select in UI, verify stack_id in log |
| Benchmark runner completes full question set via WebSocket | BENCH-02 | Requires live backend + WebSocket | Run `python benchmark_runner.py --fixture questions.jsonl`, verify JSONL output |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
