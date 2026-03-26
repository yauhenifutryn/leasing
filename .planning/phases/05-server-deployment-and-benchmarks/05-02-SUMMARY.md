---
phase: 05-server-deployment-and-benchmarks
plan: 02
subsystem: infra
tags: [bash, benchmark, orchestrator, smoke-test, supervisord, nvidia-smi, vllm, vram]

# Dependency graph
requires:
  - phase: 03-brain-upgrade-and-benchmark-framework
    provides: "benchmark_runner.py and benchmark_compare.py CLI interfaces"
  - phase: 05-01-provisioning
    provides: "supervisord.conf program names (qwen, qwen3_omni, qwen3_tts, qwen3_asr, voxtral)"

provides:
  - "smoke_test.sh: extended with profile-aware sidecar health, vLLM readiness, and VRAM checks"
  - "benchmark_orchestrator.sh: sequential 4-step benchmark matrix runner with interactive winner selection"

affects:
  - server benchmark execution workflow

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Profile -> required sidecars mapping via bash case statement with 7 profiles"
    - "BEST_SPLIT_RESULT variable tracks best result across Steps 1-2 for correct Omni comparison"
    - "swap_model() centralizes stop -> VRAM check -> start -> wait_healthy sequence"
    - "RESULT_FOR_BRAIN_COMPARE set to dify result when dify wins Step 1 (avoids RAG-brain conflation)"
    - "read -rp interactive winner prompts after Steps 1 and 2 before conditional branching"
    - "SERVER_IP auto-detected via curl ifconfig.me with hostname fallback"

# Key files
key-files:
  created:
    - rag_demo_system/scripts/benchmark_orchestrator.sh
  modified:
    - rag_demo_system/scripts/smoke_test.sh

# Decisions
decisions:
  - "smoke_test.sh appended rather than rewritten: preserves the original 62-line checks verbatim per D-07; new blocks start after line 62"
  - "SUPERVISORCTL defined as variable (not inline): allows easy path override for non-standard .venv locations"
  - "vLLM skip for omni_hybrid: Omni uses its own model server on port 8002; checking vLLM /health would give false confidence"
  - "VRAM floor check at 10240 MiB: 10GB minimum signals a model is loaded; warns without aborting (nvidia-smi warns, low VRAM aborts)"
  - "sed -i.bak patch for brain model swap: modifies .env in-place with backup, restored after Step 2; avoids profile-specific vLLM command duplication"

# Metrics
metrics:
  duration: "~15 minutes"
  completed: "2026-03-26T17:02:00Z"
  tasks_completed: 2
  files_modified: 2
---

# Phase 05 Plan 02: Smoke Test Extension and Benchmark Orchestrator Summary

**One-liner:** Profile-aware smoke_test.sh (sidecar health + vLLM + VRAM) and 4-step benchmark_orchestrator.sh with interactive winner selection and model swaps via supervisorctl.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extend smoke_test.sh with profile-aware sidecar, vLLM, and VRAM checks | 6bac74d | rag_demo_system/scripts/smoke_test.sh |
| 2 | Create benchmark_orchestrator.sh (sequential 4-step benchmark runner) | 14cb9c4 | rag_demo_system/scripts/benchmark_orchestrator.sh |

## What Was Built

### Task 1: Extended smoke_test.sh (151 lines, was 62)

Four new sections appended after the existing checks:

**Section A: Profile detection and env loading**
- `BENCH_PROFILE="${BENCH_PROFILE:-baseline}"` for profile detection
- Sources `.env.bench.$BENCH_PROFILE` so sidecar BASE_URL overrides are available

**Section B: Profile-aware sidecar health (D-06)**
- `case` statement maps all 7 profiles to required sidecar URL pairs
- Loops over `REQUIRED_SIDECARS[@]` calling `/health`; exits 1 on failure
- baseline/dify_rag/brain_upgrade: SenseVoice + CosyVoice
- qwen3_tts: SenseVoice + Qwen3-TTS
- qwen3_asr: Qwen3-ASR + CosyVoice
- voxtral: Voxtral + CosyVoice
- omni_hybrid: SenseVoice + Qwen3-Omni

**Section C: vLLM readiness (D-05)**
- Health check via GET `/health`
- Trivial completion via POST `/completions` with `max_tokens=3`
- Skipped for `omni_hybrid` profile (Omni has its own model server on port 8002)

**Section D: VRAM headroom**
- `nvidia-smi --query-gpu=memory.used/total --format=csv,noheader,nounits`
- Warns if used < 10240 MiB (model may not be loaded)
- Gracefully skips if `nvidia-smi` is absent

### Task 2: benchmark_orchestrator.sh (365 lines)

**Configuration block:** SUPERVISORCTL, BENCHMARK_RUNNER, BENCHMARK_COMPARE variables with `.venv/bin/` paths; WS_URL, BACKEND_URL, VRAM_LIMIT_GB, HEALTH_TIMEOUT constants.

**Helper functions:**
- `log()`: timestamped prefix `[orch][HH:MM:SS]`
- `wait_healthy(url, timeout=300)`: 10-second polling with elapsed counter
- `check_vram(context)`: reports used/total GB; aborts if used >= 75 GB
- `swap_model(stop, start, health_url)`: stop -> sleep 5 -> VRAM check -> start -> wait_healthy -> VRAM check
- `run_benchmark(profile)`: loads `.env.bench.$profile`, invokes `benchmark_runner.py --ws-url ws://localhost:8000/ws/voice --backend-url http://localhost:8000 --timeout 30 --warmup 3`; returns output file path
- `run_comparison(file_a, file_b, label)`: invokes `benchmark_compare.py`, tees to stdout and `results/compare_<label>_<timestamp>.md`
- `print_scp(label, files...)`: prints ready-to-paste `scp root@SERVER_IP:...` command

**Orchestration flow:**
1. Pre-flight: `BENCH_PROFILE=baseline bash smoke_test.sh`
2. Step 1 (RAG): run baseline + dify_rag, compare, `read -rp` RAG winner; set `RESULT_FOR_BRAIN_COMPARE` and `BEST_SPLIT_RESULT` based on winner
3. Step 2 (brain): stop qwen, patch `.env` with sed, start qwen with Qwen3.5-35B, benchmark brain_upgrade, compare against `RESULT_FOR_BRAIN_COMPARE`, restore `.env`; `read -rp` brain winner; update `BEST_SPLIT_RESULT` if brain_upgrade wins
4. Step 3 (Omni): `swap_model qwen qwen3_omni`; benchmark omni_hybrid; compare against `BEST_SPLIT_RESULT` (not hardcoded baseline); `read -r USER_DECISION` for continue/skip
5. Step 4 (conditional): restart vLLM, start qwen3_tts/qwen3_asr/voxtral sidecars one at a time, benchmark each vs baseline, stop sidecar between runs

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - no placeholder data flows to any output.

## Self-Check: PASSED

- `rag_demo_system/scripts/smoke_test.sh`: exists, 151 lines, bash -n passes
- `rag_demo_system/scripts/benchmark_orchestrator.sh`: exists, 365 lines, bash -n passes, chmod +x applied
- Commit 6bac74d: confirmed in git log
- Commit 14cb9c4: confirmed in git log
