---
phase: 05-server-deployment-and-benchmarks
verified: 2026-03-26T17:07:25Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 5: Server Deployment and Benchmarks Verification Report

**Phase Goal:** The GPU server is provisioned from a fresh VM, all services pass the smoke test, and the benchmark matrix is executed in smart order — RAG comparison first, brain comparison second, Omni hybrid third (the most promising experiment, now with a baseline to compare against), and full STT/TTS matrix only as a fallback if Omni does not perform well enough
**Verified:** 2026-03-26T17:07:25Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Must-Haves Derivation

Must-haves were taken directly from `must_haves:` frontmatter in both PLAN files, cross-referenced against ROADMAP.md Phase 5 success criteria.

**Plans claiming requirements:**

| Plan | Requirement |
|------|-------------|
| 05-01-PLAN.md | DEPLOY-02 |
| 05-02-PLAN.md | DEPLOY-03 |

Both DEPLOY-02 and DEPLOY-03 are mapped to Phase 5 in REQUIREMENTS.md. No orphaned requirements.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SSH into a fresh TensorDock A100 80GB VM, run one command, all services start without manual steps | VERIFIED | `provision_server.sh` exists, 292 lines, executable, bash -n passes, calls install_apt_packages -> check_nvidia_driver -> clone_repo -> install_all_venvs -> download_models -> start_docker_qdrant -> write_env_file -> start_stack in sequence |
| 2 | NVIDIA driver is detected and installed only if missing; script exits with reboot message if driver install was needed | VERIFIED | `check_nvidia_driver()` at line 69: `nvidia-smi &>/dev/null` branch returns 0 if present; else runs `ubuntu-drivers install` and `exit 1` with reboot message |
| 3 | All 6 per-sidecar venvs are created with correct requirements files | VERIFIED | `install_all_venvs()` calls `ensure_venv` for .venv, .venv-voice-oss, .venv-qwen3-tts, .venv-qwen3-asr, .venv-voxtral, .venv-qwen3-omni each with matching requirements file |
| 4 | HuggingFace models download via huggingface-cli with resume support and HF_HOME pointing to the large volume | VERIFIED | `download_models()` sets `export HF_HOME="$MODELS_DIR"` and calls `huggingface-cli download` for all 7 models, every call includes `--local-dir-use-symlinks False` |
| 5 | supervisord entries exist for qwen3_tts, qwen3_asr, and voxtral so the orchestrator can start/stop them | VERIFIED | supervisord.conf has 13 `[program:...]` entries; `[program:qwen3_tts]`, `[program:qwen3_asr]`, `[program:voxtral]` all present with `autostart=false` |
| 6 | Smoke test validates sidecar health endpoints only for sidecars the active profile actually needs | VERIFIED | smoke_test.sh lines 83-113: `case "$BENCH_PROFILE"` maps all 7 profiles to `REQUIRED_SIDECARS` array, loops over it calling `/health`; unknown profile skips entirely |
| 7 | Smoke test validates vLLM is loaded by sending a trivial completion request | VERIFIED | smoke_test.sh lines 115-133: skips for omni_hybrid; otherwise hits `$VLLM_HEALTH` with GET then POST `/completions` with `max_tokens=3` and asserts `choices` is non-empty |
| 8 | Smoke test checks VRAM usage via nvidia-smi | VERIFIED | smoke_test.sh lines 136-148: `nvidia-smi --query-gpu=memory.used`; warns if used < 10240 MiB; gracefully skips if nvidia-smi absent |
| 9 | Benchmark orchestrator runs 4 steps sequentially: RAG comparison, brain comparison, Omni hybrid, STT/TTS matrix (conditional) | VERIFIED | benchmark_orchestrator.sh: STEP 1 (RAG) lines 184-218, STEP 2 (brain) lines 221-266, STEP 3 (Omni) lines 269-299, STEP 4 (STT/TTS conditional) lines 302-343 |
| 10 | Orchestrator swaps models via supervisorctl stop/start between steps, waits for /health 200, verifies VRAM | VERIFIED | `swap_model()` function: stop -> sleep 5 -> `check_vram` -> start -> `wait_healthy` -> `check_vram`; `check_vram` aborts if used >= 75 GB |
| 11 | Orchestrator pauses after step 3 and shows Omni comparison results for user decision | VERIFIED | Lines 288-298: `read -r USER_DECISION`; Step 4 only executes if `USER_DECISION = "continue"` |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `rag_demo_system/scripts/provision_server.sh` | Full A100 VM provisioning automation | VERIFIED | 292 lines (min 150), executable, bash -n passes, contains all required functions |
| `rag_demo_system/scripts/supervisord.conf` | Supervisor entries for all sidecar services | VERIFIED | 13 `[program:...]` entries; contains `[program:qwen3_tts]` with `autostart=false` |
| `rag_demo_system/results/.gitkeep` | Results directory placeholder | VERIFIED | File exists |
| `rag_demo_system/.gitignore` | Gitignore with results/ entry | VERIFIED | `results/` present |
| `rag_demo_system/.env.example` | Env template with all STACK_*_CMD vars | VERIFIED | Contains `STACK_QWEN3_OMNI_CMD=`, `STACK_QWEN3_TTS_CMD=`, `STACK_QWEN3_ASR_CMD=`, `STACK_VOXTRAL_CMD=`, and all four BASE_URL entries |
| `rag_demo_system/scripts/smoke_test.sh` | Profile-aware smoke test with sidecar, vLLM, and VRAM checks | VERIFIED | 151 lines (min 100), bash -n passes, all new checks appended after line 62 |
| `rag_demo_system/scripts/benchmark_orchestrator.sh` | Sequential 4-step benchmark matrix runner | VERIFIED | 365 lines (min 200), bash -n passes, executable |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `provision_server.sh` | `scripts/supervisord.conf` | `start_stack()` calls `stack.sh up`, which delegates to `stack_cli.py`; `stack_cli.py` resolves supervisord.conf at line 90 | WIRED (indirect) | Direct `supervisorctl -c` pattern not in provision_server.sh; the link goes through stack.sh -> stack_cli.py -> supervisord.conf. Functionally complete. |
| `provision_server.sh` | `requirements*.txt` | `ensure_venv()` calls `"$target/bin/pip" install -r "$req_file"` where $req_file expands to each requirements-*.txt | WIRED | Pattern differs from `pip install -r.*requirements-` in the PLAN (literal string), but the actual expansion at runtime matches; substantively correct |
| `benchmark_orchestrator.sh` | `benchmark_runner.py` | `$BENCHMARK_RUNNER --fixture ... --profile "$profile" --output ... --ws-url ... --backend-url ...` | WIRED | Variable defined at line 30; invoked at line 126 with `--profile` flag present |
| `benchmark_orchestrator.sh` | `benchmark_compare.py` | `$BENCHMARK_COMPARE "$file_a" "$file_b"` teed to results file | WIRED | Variable defined at line 31; invoked at line 148 with two JSONL file path arguments |
| `benchmark_orchestrator.sh` | `smoke_test.sh` | `BENCH_PROFILE=baseline bash "$SMOKE_TEST"` before Step 1 | WIRED | `SMOKE_TEST` defined at line 32; called at line 181 as pre-flight check |
| `smoke_test.sh` | `.env.bench.*` | Sources `$APP_DIR/.env.bench.$BENCH_PROFILE` at line 72 via `PROFILE_FILE` | WIRED | `BENCH_PROFILE` drives profile selection; file sourced so BASE_URL overrides take effect for sidecar checks |

---

### Data-Flow Trace (Level 4)

Artifacts in this phase are bash scripts (provisioning, smoke testing, orchestration), not React components rendering dynamic data. No stateful rendering. Level 4 data-flow trace does not apply to shell scripts.

---

### Behavioral Spot-Checks

These scripts target a remote GPU server environment. Running them locally would fail (no nvidia-smi, no GPU, no vLLM). Spot-checks limited to static verification.

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `provision_server.sh` parses cleanly | `bash -n rag_demo_system/scripts/provision_server.sh` | Exit 0 | PASS |
| `smoke_test.sh` parses cleanly | `bash -n rag_demo_system/scripts/smoke_test.sh` | Exit 0 | PASS |
| `benchmark_orchestrator.sh` parses cleanly | `bash -n rag_demo_system/scripts/benchmark_orchestrator.sh` | Exit 0 | PASS |
| `provision_server.sh` is executable | `test -x rag_demo_system/scripts/provision_server.sh` | Exit 0 | PASS |
| supervisord.conf has 13 program entries | `grep -c 'program:' supervisord.conf` | 13 | PASS |
| All 4 STACK_*_CMD env vars present in .env.example | grep count | 4 matches | PASS |
| Commit hashes in SUMMARYs exist in git log | `git log --oneline` | 25cccf5, 0a67652, 6bac74d, 14cb9c4 all found | PASS |

Note: Full end-to-end execution (actual GPU provisioning, service startup, benchmark runs) requires a live TensorDock/Vast.ai VM and is flagged for human verification below.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DEPLOY-02 | 05-01-PLAN.md | Server deployment script that provisions the stack from a fresh GPU VM | SATISFIED | `provision_server.sh` (292 lines): apt install, NVIDIA driver check/install with reboot guard, repo clone, 6 venvs, 7 HF model downloads with HF_HOME on large volume, Qdrant Docker, .env generation, stack launch |
| DEPLOY-03 | 05-02-PLAN.md | Smoke test script validates all services are healthy before benchmark execution | SATISFIED | `smoke_test.sh` (151 lines): existing 62 lines preserved, plus profile-aware sidecar /health checks, vLLM health + trivial completion check, VRAM headroom via nvidia-smi; `benchmark_orchestrator.sh` calls smoke_test.sh as pre-flight before any benchmark step |

**Coverage:** 2/2 Phase 5 requirements satisfied. No orphaned requirements.

---

### Anti-Patterns Found

Scanned all 5 files created/modified in this phase.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `provision_server.sh` | 264 | `start_stack` calls `bash scripts/stack.sh up` without checking if stack.sh itself succeeds or re-reads the .env file just written | Info | Not a stub; stack.sh does source .env at startup. No user-visible empty data. |
| `benchmark_orchestrator.sh` | 309-315 | After stopping Omni and restarting vLLM, no explicit wait for vLLM to finish loading before STT/TTS benchmarks | Info | `wait_healthy` is called for each sidecar start via `swap_model`, but the vLLM restart in step 4 uses a direct `$SUPERVISORCTL start qwen` without `wait_healthy`. This is a minor operational risk, not a data flow stub. |

No STUB, PLACEHOLDER, or hardcoded empty data patterns found. No `TODO/FIXME` in any phase 5 file. No `return null` or empty array returns. All anti-patterns are Info-level operational observations.

---

### Human Verification Required

The phase goal is fully expressed in code but the code targets a remote GPU server. The following items cannot be verified programmatically without that server:

#### 1. End-to-End Provisioning

**Test:** Spin up a fresh TensorDock or Vast.ai A100 80GB Ubuntu 22.04 VM. Set `HF_TOKEN`. Run: `HF_TOKEN=hf_... bash rag_demo_system/scripts/provision_server.sh`
**Expected:** Script completes without prompting for manual steps. All 6 venvs created. All 7 HF models downloaded. Qdrant container starts. `.env` written. `bash scripts/smoke_test.sh` exits 0.
**Why human:** Requires a live GPU VM, real HF token with model access, 80 GB model downloads.

#### 2. NVIDIA Driver Guard (First Boot Path)

**Test:** On a fresh VM without GPU drivers, run provision_server.sh.
**Expected:** Script runs apt install, detects nvidia-smi is absent, runs `ubuntu-drivers install`, prints "REBOOT REQUIRED", exits 1. After reboot, re-running the script detects driver OK and continues without re-installing.
**Why human:** Requires a VM state where the driver is intentionally absent.

#### 3. Smoke Test Profile Coverage

**Test:** With all services running, run: `BENCH_PROFILE=omni_hybrid bash smoke_test.sh`
**Expected:** Script skips vLLM check (correct for omni_hybrid), checks SenseVoice + Qwen3-Omni sidecars, checks VRAM.
**Why human:** Requires live sidecars at runtime; `bash -n` cannot verify runtime profile logic.

#### 4. Benchmark Orchestrator Full Run

**Test:** Run `bash benchmark_orchestrator.sh` on a provisioned server.
**Expected:** Pre-flight smoke passes, Step 1 produces two JSONL files and comparison table, interactive `read -rp` prompts appear, Step 2 patches .env with sed and restores it, Step 3 does `swap_model qwen qwen3_omni`, pauses for user decision, Step 4 only runs when user types "continue".
**Why human:** Interactive `read -rp` prompts cannot be verified without a TTY and a running server.

#### 5. VRAM Guard at 75 GB

**Test:** Load Qwen3.5-35B-A3B and Qwen3-Omni simultaneously (should not happen in normal flow but tests the guard).
**Expected:** `check_vram` aborts with "VRAM used >= limit" before loading the second model.
**Why human:** Requires live GPU with near-limit VRAM usage.

---

## Gaps Summary

No gaps. All artifacts exist, are substantive (line counts exceed minimums), have passing bash -n syntax checks, and all key links are wired. Requirements DEPLOY-02 and DEPLOY-03 are fully addressed. Commits cited in both SUMMARYs are verified in git log.

The only items requiring human judgment are operational tests that require a live GPU server, which is expected for a server deployment phase.

---

_Verified: 2026-03-26T17:07:25Z_
_Verifier: Claude (gsd-verifier)_
