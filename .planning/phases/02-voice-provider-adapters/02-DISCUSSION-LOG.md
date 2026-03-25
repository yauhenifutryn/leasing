# Phase 2: Voice Provider Adapters - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-25
**Phase:** 02-voice-provider-adapters
**Areas discussed:** Voxtral scope, Fallback policy, Sidecar design, Model versions

---

## Voxtral Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Research first | Let researcher investigate self-host availability. If yes, build sidecar. If no, build cloud API adapter as benchmark-only. | Yes |
| Cloud API only | Build Mistral API client right away. Useful for benchmarking but breaks privacy constraint. | |
| Defer Voxtral | Skip entirely in Phase 2. Focus on Qwen3-TTS and Qwen3-ASR only. | |

**User's choice:** Research first (Recommended)
**Notes:** None. User accepted recommended approach.

---

## Fallback Policy

| Option | Description | Selected |
|--------|-------------|----------|
| Hard fail | If selected provider is down, voice turn fails with clear error. No silent substitution. Best for benchmark integrity. | Yes |
| Fail with retry | Same hard fail but retries 2-3 times before giving up. Handles brief hiccups. | |
| Mode switch | Add benchmark mode toggle. Hard fail in benchmark mode, graceful fallback in normal mode. | |

**User's choice:** Hard fail (Recommended)
**Notes:** None. User accepted recommended approach.

---

## Sidecar Design

| Option | Description | Selected |
|--------|-------------|----------|
| One script per model | Each model gets standalone FastAPI script, own venv, own health endpoint. Matches CosyVoice/SenseVoice pattern. | Yes |
| Shared sidecar framework | Reusable template all sidecars share. Less duplication but more abstraction. | |
| You decide | Let Claude pick based on codebase patterns. | |

**User's choice:** One script per model (Recommended)
**Notes:** None. User accepted recommended approach.

---

## Model Versions

| Option | Description | Selected |
|--------|-------------|----------|
| Researcher confirms | Verify exact HuggingFace repo IDs, sample rates, and loading libraries before planning. Safest. | Yes |
| Pin playbook versions | Use playbook model names as-is. Fix during execution if wrong. Faster but riskier. | |
| I have specific versions | User provides exact model repos. | |

**User's choice:** Researcher confirms (Recommended)
**Notes:** None. User accepted recommended approach.

---

## Claude's Discretion

- Sidecar script internal structure (model loading, warmup, request handling organization)
- HTTP timeout values and request payload formats
- Error message wording for hard-fail scenarios

## Deferred Ideas

None -- discussion stayed within phase scope.
