---
phase: 05-server-deployment-and-benchmarks
plan: 01
subsystem: infra
tags: [bash, supervisord, huggingface, vllm, qdrant, docker, provisioning, tensordock]

# Dependency graph
requires:
  - phase: 02-voice-provider-adapters
    provides: "per-sidecar requirements files and venv names for qwen3_tts, qwen3_asr, voxtral"
  - phase: 04-qwen3-omni-hybrid
    provides: "qwen3_omni supervisord entry pattern and requirements-qwen3-omni.txt"

provides:
  - "provision_server.sh: single-command TensorDock A100 80GB VM provisioning"
  - "supervisord.conf: all 13 service entries including qwen3_tts, qwen3_asr, voxtral"
  - "results/.gitkeep: benchmark output directory in repo"
  - ".env.example: complete env template with all STACK_*_CMD and BASE_URL vars"

affects:
  - 05-02-benchmark-orchestrator
  - server deployment documentation

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Fail-fast HF_TOKEN check at top of provisioning script (before any network ops)"
    - "ensure_venv() helper for idempotent venv creation with pip install -r"
    - "install_all_venvs before download_models to ensure huggingface-cli is available"
    - "install_apt_packages before check_nvidia_driver to ensure ubuntu-drivers-common is present"
    - "--local-dir-use-symlinks False on all huggingface-cli download calls to avoid symlink caching"

key-files:
  created:
    - rag_demo_system/scripts/provision_server.sh
    - rag_demo_system/results/.gitkeep
  modified:
    - rag_demo_system/scripts/supervisord.conf
    - rag_demo_system/.gitignore
    - rag_demo_system/.env.example

key-decisions:
  - "install_all_venvs called before download_models: backend venv provides huggingface-cli binary"
  - "install_apt_packages called before check_nvidia_driver: ubuntu-drivers-common requires apt"
  - "exit 1 after ubuntu-drivers install: GPU device node requires reboot before first use"
  - "--local-dir-use-symlinks False on all HF downloads: prevents symlink caching that defeats HF_HOME volume placement"
  - "HF_HOME=$MODELS_DIR in download_models: all weights land on the large volume not the OS disk"
  - "autostart=false for qwen3_tts, qwen3_asr, voxtral: GPU budget prevents co-hosting all models; orchestrator starts only needed sidecar per profile"

requirements-completed: [DEPLOY-02]

# Metrics
duration: 10min
completed: 2026-03-26
---

# Phase 5 Plan 01: Server Provisioning Infrastructure Summary

**Single-command TensorDock A100 80GB provisioning via provision_server.sh: apt, NVIDIA driver check, repo clone, 6 isolated venvs, 7 HF model downloads, Qdrant Docker, .env generation, and stack launch**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-26T17:00:00Z
- **Completed:** 2026-03-26T17:10:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Created 292-line `provision_server.sh` that provisions a fresh A100 VM end to end with a single command
- Added 3 missing supervisord program entries (qwen3_tts, qwen3_asr, voxtral) bringing total to 13
- Updated `.env.example` with 4 new STACK_*_CMD vars and 4 new BASE_URL entries covering all sidecars
- Added `results/` to `.gitignore` and created `results/.gitkeep` so the benchmark output directory is tracked

## Task Commits

Each task was committed atomically:

1. **Task 1: Infrastructure prerequisites** - `25cccf5` (feat)
2. **Task 2: Full GPU server provisioning script** - `0a67652` (feat)

## Files Created/Modified

- `rag_demo_system/scripts/provision_server.sh` - Single-command A100 VM provisioner (292 lines, executable, bash -n passes)
- `rag_demo_system/results/.gitkeep` - Benchmark output directory placeholder
- `rag_demo_system/scripts/supervisord.conf` - Added qwen3_tts, qwen3_asr, voxtral program entries
- `rag_demo_system/.gitignore` - Added results/ exclusion
- `rag_demo_system/.env.example` - Added STACK_QWEN3_TTS_CMD, STACK_QWEN3_ASR_CMD, STACK_VOXTRAL_CMD, STACK_QWEN3_OMNI_CMD and four BASE_URL vars

## Decisions Made

- `install_all_venvs` is called before `download_models` in main: the backend venv provides the `huggingface-cli` binary used for all model downloads (Pitfall 2 from RESEARCH.md)
- `install_apt_packages` is called before `check_nvidia_driver`: `ubuntu-drivers-common` must be installed via apt before `ubuntu-drivers install` can run (Pitfall 1)
- Script exits with error message after `ubuntu-drivers install`: GPU device node does not exist until the VM is rebooted; safe to re-run after reboot
- `--local-dir-use-symlinks False` on every `huggingface-cli download` call: prevents symlink caching that bypasses HF_HOME volume placement (RESEARCH.md Pattern 3)
- `autostart=false` for all three new supervisor programs: GPU memory budget on A100 80GB prevents co-hosting all model sidecars; the benchmark orchestrator controls which sidecars are active per test run

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `results/.gitkeep` could not be staged with `git add` because the `.gitignore` entry was applied in the same commit. Used `git add -f` to force-add the tracked placeholder file despite the ignore rule. This is the correct pattern for tracked empty-directory placeholders that live inside a gitignored directory.

## User Setup Required

External services require configuration before running `provision_server.sh`:

| Service | Requirement |
|---------|-------------|
| HuggingFace | `HF_TOKEN` env var: Settings -> Access Tokens -> New token (read) |
| TensorDock | Create A100 80GB VM with Ubuntu 22.04 via TensorDock dashboard |

Run the script: `HF_TOKEN=hf_... bash rag_demo_system/scripts/provision_server.sh`

## Next Phase Readiness

- `provision_server.sh` is ready for use on a real TensorDock VM
- Plan 02 (benchmark orchestrator) can now reference all 13 supervisord programs including the new sidecars
- GPU OOM risk still applies: never load Qwen3.5-35B-A3B (~70 GB) and Qwen3-Omni-30B (~60 GB) simultaneously; swap via supervisorctl

---
*Phase: 05-server-deployment-and-benchmarks*
*Completed: 2026-03-26*
