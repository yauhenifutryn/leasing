# Phase 5: Server Deployment and Benchmarks - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 05-server-deployment-and-benchmarks
**Areas discussed:** Server provisioning, Smoke test scope, Benchmark execution flow, Result collection

---

## Server Provisioning

### GPU Target

| Option | Description | Selected |
|--------|-------------|----------|
| Vast.ai A100 80GB | Cheaper, more available for short benchmark runs. Pre-installed drivers. | |
| Azure H100 NVL 94GB | Playbook's preferred Azure pick. More headroom but higher cost. | |
| Both, with a flag | Script detects or accepts --target flag. | |
| TensorDock A100 80GB (Other) | User's actual target provider. | ✓ |

**User's choice:** TensorDock VM with A100 80GB
**Notes:** User specified TensorDock as the provider, not Vast.ai or Azure.

### Automation Level

| Option | Description | Selected |
|--------|-------------|----------|
| Full auto | Single script: apt, Docker, repo, venvs, models, .env, start. SSH + one command. | ✓ |
| Setup + manual model download | Script handles infra. Model downloads are separate manual step. | |
| Guided checklist | Print step-by-step instructions. Maximum control. | |

**User's choice:** Full auto
**Notes:** None

### Model Downloads

| Option | Description | Selected |
|--------|-------------|----------|
| huggingface-cli download | HF_HOME shared dir, resume, caching. Requires HF_TOKEN. | ✓ |
| Git LFS clone | Simple but no resume on failure. | |
| You decide | Claude picks. | |

**User's choice:** huggingface-cli download
**Notes:** None

### Driver Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Assume pre-installed | TensorDock pre-installs. Just verify nvidia-smi. | |
| Include driver install with --skip-drivers | Has install path but skips by default. | |
| Check and install if needed (Other) | Check nvidia-smi first; if missing, trigger installation. | ✓ |

**User's choice:** Check nvidia-smi first, install drivers if not present
**Notes:** User unsure whether TensorDock VM will have drivers pre-installed. Wants defensive check-then-install approach.

---

## Smoke Test Scope

### Validation Checks

| Option | Description | Selected |
|--------|-------------|----------|
| Sidecar health endpoints | Hit GET /health on every sidecar the profile needs. | ✓ |
| vLLM readiness check | Trivial completion request to vLLM endpoint. | ✓ |
| VRAM headroom check | Parse nvidia-smi, warn if free VRAM below threshold. | ✓ |
| One warmup voice turn | Full WebSocket pipeline end-to-end sanity check. | |

**User's choice:** Sidecar health, vLLM readiness, VRAM headroom (all three recommended)
**Notes:** Warmup voice turn was not selected.

### Profile Awareness

| Option | Description | Selected |
|--------|-------------|----------|
| Profile-aware | Only check sidecars the active profile needs. Less noise. | ✓ |
| Check everything | Hit all known endpoints. Full picture but expected failures. | |

**User's choice:** Profile-aware
**Notes:** None

---

## Benchmark Execution Flow

### Orchestration Style

| Option | Description | Selected |
|--------|-------------|----------|
| Single orchestrator | One script runs all steps. Pauses after step 3 for decision. | ✓ |
| Per-step scripts | Separate scripts, run manually per step. | |
| You decide | Claude picks. | |

**User's choice:** Single orchestrator
**Notes:** None

### Model Swap Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| supervisorctl stop/start | Targeted service swap. Wait for /health. Verify VRAM. | ✓ |
| Full stack restart | stack.sh down/up. Cleaner but slower. | |
| You decide | Claude picks. | |

**User's choice:** supervisorctl stop/start
**Notes:** None

### Step 4 Gate

| Option | Description | Selected |
|--------|-------------|----------|
| Always pause and ask | Print Omni results, wait for user input. | ✓ |
| Automatic threshold | Skip step 4 if Omni meets KPI thresholds. | |
| Always run step 4 | Maximum data regardless. | |

**User's choice:** Always pause and ask
**Notes:** None

---

## Result Collection

### Result File Location

| Option | Description | Selected |
|--------|-------------|----------|
| results/ in repo | rag_demo_system/results/. Gitignored. | ✓ |
| Separate output dir | Outside repo entirely. | |
| You decide | Claude picks. | |

**User's choice:** results/ directory in repo
**Notes:** None

### Comparison Timing

| Option | Description | Selected |
|--------|-------------|----------|
| After each step | Incremental comparisons. Informs step 4 gate decision. | ✓ |
| Only at the end | Manual comparisons after all steps complete. | |
| You decide | Claude picks. | |

**User's choice:** After each step
**Notes:** None

### Transfer Method

| Option | Description | Selected |
|--------|-------------|----------|
| scp/rsync manually | Orchestrator prints paths and scp command. | ✓ |
| Git commit to branch | Push results to branch, pull locally. | |
| You decide | Claude picks. | |

**User's choice:** scp/rsync manually
**Notes:** None

---

## Claude's Discretion

- Exact provisioning script structure (bash functions, error handling, logging)
- Which apt packages beyond the playbook list are needed
- Exact HuggingFace model repo IDs for download commands
- How the orchestrator detects the "winning" configuration from each step
- Smoke test VRAM threshold value
- Result file timestamp format

## Deferred Ideas

None -- discussion stayed within phase scope.
