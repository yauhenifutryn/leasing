# Phase 5: Server Deployment and Benchmarks - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Provision a GPU server from a fresh VM, validate all services with an expanded smoke test, then execute the full benchmark matrix in smart order (RAG first, brain second, Omni third, STT/TTS only if Omni fails). No new adapters, no new UI work, no code changes to the pipeline itself.

</domain>

<decisions>
## Implementation Decisions

### Server Provisioning
- **D-01:** Target is a TensorDock VM with A100 80GB. The provisioning script is written for this provider specifically.
- **D-02:** Full automation. Single script: apt packages, Docker, clone repo, create all venvs (backend + per-sidecar), download models from HuggingFace, write .env, start stack. SSH in, run one command, walk away.
- **D-03:** Model downloads use `huggingface-cli download` with HF_HOME set to a shared models directory. Handles resume on disconnect and caching. Requires HF_TOKEN env var for gated models.
- **D-04:** Driver handling: check `nvidia-smi` first. If it fails, trigger the ubuntu-drivers install procedure (`sudo apt install -y ubuntu-drivers-common && sudo ubuntu-drivers install`), reboot if needed, then verify again. If drivers are already present, skip the install entirely.

### Smoke Test Scope
- **D-05:** Expanded smoke test validates three additional checks beyond the existing HTTP endpoints: (1) sidecar health endpoints, (2) vLLM readiness via a trivial completion request, (3) VRAM headroom via nvidia-smi parsing.
- **D-06:** Smoke test is profile-aware. Reads the active .env.bench profile to determine which sidecars should be running. Only fails on sidecars the current profile actually needs. Avoids false failures from intentionally-stopped services.
- **D-07:** Existing smoke_test.sh checks (UI root, /api/health, /api/backends, /api/voice/status, KB indexing, chat stream) are preserved and extended, not replaced.

### Benchmark Execution Flow
- **D-08:** Single orchestrator script runs the full benchmark matrix sequentially. Steps: (1) RAG comparison, (2) brain comparison, (3) Omni hybrid vs best split, (4) STT/TTS matrix (conditional).
- **D-09:** Between benchmark steps, the orchestrator swaps models via `supervisorctl stop/start`. Stops old model service, updates .env with new profile, starts the new one, waits for /health 200, verifies VRAM with nvidia-smi before proceeding. Does NOT restart the full stack.
- **D-10:** After step 3 (Omni hybrid benchmark), the orchestrator always pauses. Prints Omni vs split pipeline comparison results on screen and waits for user input ('continue' to run step 4, 'skip' to finish). User decides whether STT/TTS matrix is needed based on Omni results.
- **D-11:** The orchestrator runs the comparison script automatically after each benchmark step. After step 1: our_rag vs dify_rag. After step 2: brain models. After step 3: Omni vs best split. Results are visible incrementally.

### Result Collection
- **D-12:** JSONL result files are written to `rag_demo_system/results/` directory within the repo checkout. Naming: `bench_{profile}_{timestamp}.jsonl`. Directory is gitignored.
- **D-13:** Results are transferred off the server via scp/rsync manually. The orchestrator prints file paths and a ready-to-paste scp command at the end of each step.

### Claude's Discretion
- Exact provisioning script structure (bash functions, error handling, logging)
- Which apt packages beyond the playbook list are needed
- Exact HuggingFace model repo IDs for download commands (researcher confirms these)
- How the orchestrator detects the "winning" configuration from each step to carry forward
- Smoke test VRAM threshold value
- Result file timestamp format

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Voice AI Playbook
- `docs/voice_ai_playbook_2026-03-25.md` -- Authoritative planning document. "Azure Server Recommendation" section (lines ~463-500) for VM sizing rationale. Steps 1-9 (lines ~540-648) for the full server setup procedure. "Phase B: First Azure Deployment" (lines ~682-688) and "Phase C: Controlled Upgrades" (lines ~690-696) for benchmark execution order.

### Existing Deployment Scripts
- `rag_demo_system/scripts/setup_vast_voice.sh` -- Existing provisioning script for legacy profiles. Pattern for venv creation, model download, and stack setup. Needs extending for new profiles.
- `rag_demo_system/scripts/stack.sh` -- Stack launcher; sources .env then delegates to stack_cli.py.
- `rag_demo_system/scripts/stack_cli.py` -- Stack CLI with supervisor management. `build_program_selection()` and `sync_supervisor_programs()` are the model swap primitives.
- `rag_demo_system/scripts/supervisord.conf` -- All supervisor program entries including qwen3_omni. Services use STACK_{NAME}_CMD env vars.

### Smoke Test
- `rag_demo_system/scripts/smoke_test.sh` -- Current smoke test. Needs extending with sidecar health, vLLM readiness, and VRAM checks.

### Benchmark Framework
- `rag_demo_system/scripts/benchmark_runner.py` -- Benchmark runner CLI. The orchestrator invokes this per benchmark step with appropriate --profile and --output flags.
- `rag_demo_system/scripts/benchmark_compare.py` -- Comparison script. Orchestrator calls this after each step with the two result files.

### Env Profiles
- `rag_demo_system/.env.bench.baseline` -- Baseline profile
- `rag_demo_system/.env.bench.qwen3_tts` -- Qwen3-TTS profile
- `rag_demo_system/.env.bench.qwen3_asr` -- Qwen3-ASR profile
- `rag_demo_system/.env.bench.voxtral` -- Voxtral profile
- `rag_demo_system/.env.bench.brain_upgrade` -- Brain upgrade profile
- `rag_demo_system/.env.bench.dify_rag` -- Dify RAG profile
- `rag_demo_system/.env.bench.omni_hybrid` -- Omni hybrid profile

### Sidecar Patterns
- `rag_demo_system/backend/qwen3_tts_sidecar.py` -- Sidecar pattern reference (standalone FastAPI, /health, per-service venv)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `setup_vast_voice.sh`: Venv creation pattern (ensure_venv, install_backend_env), model download pattern (ensure_vosk_model). Extend for new sidecars rather than rewriting.
- `stack_cli.py:sync_supervisor_programs()`: Starts/stops supervisor programs by selection. The orchestrator can call `supervisorctl` directly to swap models between benchmark steps.
- `smoke_test.sh`: HTTP health check pattern. Extend with sidecar and vLLM checks.
- `benchmark_runner.py`: Already accepts --fixture, --profile, --output, --ws-url, --backend-url, --timeout, --warmup flags. The orchestrator invokes it with different profiles.
- `benchmark_compare.py`: Already reads two JSONL files and outputs markdown comparison tables.

### Established Patterns
- **Supervisor management:** All services managed via supervisord.conf with STACK_{NAME}_CMD env vars. autostart=false for optional services.
- **Env profile overlay:** .env.bench.{name} loaded with load_dotenv(override=True) on top of base .env.
- **Sidecar convention:** Standalone FastAPI, /health endpoint, per-service venv, {NAME}_BASE_URL env var.

### Integration Points
- The provisioning script creates the same directory structure that stack.sh and supervisord.conf expect.
- The orchestrator loads env profiles the same way the benchmark runner does (base .env + overlay).
- The smoke test must be callable both standalone (smoke_test.sh) and from within the orchestrator.

</code_context>

<specifics>
## Specific Ideas

- The orchestrator should clearly log which step it's running, which profile is active, and which services it's starting/stopping. This is a remote server; you need to know what's happening from the terminal output.
- After step 3 pause, print the comparison table directly so you can make the step 4 decision without opening files.
- The scp command printed at the end should include the correct server hostname/IP (from an env var or argument).

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 05-server-deployment-and-benchmarks*
*Context gathered: 2026-03-26*
