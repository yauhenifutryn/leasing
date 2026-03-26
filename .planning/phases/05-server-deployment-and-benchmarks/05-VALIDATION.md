---
phase: 5
slug: server-deployment-and-benchmarks
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3.4 (already in requirements.txt) |
| **Config file** | None present; pytest auto-discovers tests/ |
| **Quick run command** | `cd rag_demo_system && .venv/bin/pytest tests/ -x -q` |
| **Full suite command** | `cd rag_demo_system && .venv/bin/pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `bash -n rag_demo_system/scripts/provision_server.sh && bash -n rag_demo_system/scripts/benchmark_orchestrator.sh`
- **After every plan wave:** Run `cd rag_demo_system && .venv/bin/pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | DEPLOY-02 | unit (bash -n) | `bash -n rag_demo_system/scripts/provision_server.sh` | ❌ W0 | ⬜ pending |
| 05-01-02 | 01 | 1 | DEPLOY-02 | unit | `pytest tests/test_provision.py::test_ensure_venv_creates -x` | ❌ W0 | ⬜ pending |
| 05-01-03 | 01 | 1 | DEPLOY-02 | unit (mock) | `pytest tests/test_provision.py::test_driver_check_skip -x` | ❌ W0 | ⬜ pending |
| 05-02-01 | 02 | 1 | DEPLOY-03 | unit | `pytest tests/test_smoke_test.py::test_profile_sidecar_map -x` | ❌ W0 | ⬜ pending |
| 05-02-02 | 02 | 1 | DEPLOY-03 | unit | `pytest tests/test_smoke_test.py::test_vram_parse -x` | ❌ W0 | ⬜ pending |
| 05-02-03 | 02 | 2 | DEPLOY-03 | unit (mock) | `pytest tests/test_orchestrator.py::test_wait_healthy -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `rag_demo_system/scripts/provision_server.sh` -- skeleton script (does not exist)
- [ ] `rag_demo_system/scripts/benchmark_orchestrator.sh` -- skeleton script (does not exist)
- [ ] `rag_demo_system/results/.gitkeep` -- results dir must exist and be gitignored
- [ ] `rag_demo_system/tests/test_provision.py` -- new test file for provisioning helpers
- [ ] `rag_demo_system/tests/test_smoke_test.py` -- new test file for extended smoke test
- [ ] `rag_demo_system/tests/test_orchestrator.py` -- new test file for orchestrator helpers

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Full provisioning on TensorDock VM | DEPLOY-02 | Requires actual GPU VM with SSH access | SSH into fresh VM, run provision_server.sh, verify all services start |
| End-to-end benchmark run | DEPLOY-03 | Requires GPU, loaded models, live services | Run benchmark_orchestrator.sh on provisioned server, verify JSONL output |
| VRAM headroom with real models | DEPLOY-03 | Requires GPU with loaded models | Check nvidia-smi after model load, verify free VRAM > threshold |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
