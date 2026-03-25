# Micro Leasing Voice AI

## What This Is

An on-premises voice assistant for Micro Leasing that provides an ElevenLabs-like conversational experience over a company knowledge base, with Russian language support. Built as a split pipeline (STT -> RAG -> LLM brain -> TTS) with a browser-based UI, switchable voice providers, and a controlled benchmark framework for comparing component upgrades and an experimental native realtime path.

## Core Value

Accurate, low-latency Russian voice answers grounded in the company knowledge base, with full on-premises control over every pipeline component.

## Current Milestone: v1.0 Voice Stack Upgrade and Benchmark

**Goal:** Upgrade the split voice pipeline with better components, add a Qwen3-Omni experimental path, instrument latency, and benchmark all combinations on a GPU server.

**Target features:**
- End-to-end timing instrumentation (6 latency milestones)
- Benchmark logging and runner with fixed Russian test question set
- Qwen3-TTS, Qwen3-ASR, Voxtral voice provider adapters
- Brain upgrade path (Qwen3-30B-A3B to Qwen3.5-35B-A3B)
- Qwen3-Omni hybrid adapter with external retrieval injection
- UI selectors for all providers and brain models
- Server deployment scripts and env profiles
- Controlled benchmark execution across the full test matrix

## Requirements

### Validated

- Split pipeline architecture (STT -> RAG -> LLM -> TTS) with FastAPI + WebSocket transport
- Backend switch between `our_rag` and `dify_rag`
- Voice provider selector in UI with 4 providers: `local`, `yandex_speechkit`, `oss_russian`, `yandex_realtime`
- Low-latency voice retrieval profile (`voice_fast`: vector_top_k=3, bm25_top_k=1, final_top_n=2, reranker disabled)
- One-command stack launcher (`stack.sh`) and supervisor-managed services
- Qdrant-based vector retrieval with BM25 hybrid search
- End-to-end timing instrumentation: 6 latency milestones per voice turn with primary KPI (Validated in Phase 1: Instrumentation and UI Switching)
- Brain model selector (Qwen3-30B-A3B vs Qwen3.5-35B-A3B) with live switching (Validated in Phase 1)
- UI selectors for STT/TTS providers with localStorage persistence and session.update wiring (Validated in Phase 1)
- Automatic stack_id tagging derived from active selector combination (Validated in Phase 1)

### Active

- [ ] Benchmark logging output format and runner
- [ ] Fixed Russian benchmark test question set (80+ questions across 5 categories)
- [ ] Qwen3-TTS voice provider adapter
- [ ] Qwen3-ASR voice provider adapter
- [ ] Voxtral STT adapter (optional)
- [ ] Qwen3-Omni hybrid adapter (prompt-injected retrieved context)
- [ ] Server deployment automation and env profiles per benchmark stack
- [ ] Controlled benchmark execution and results comparison

### Out of Scope

- LiveKit/Pipecat migration — not before baseline measurements are done
- Telephony/SIP integration — browser-first for this milestone
- Qwen3-Omni pure native realtime mode — only after hybrid mode proves viable
- Replacing `our_rag` with heavier orchestration — benchmark first, then decide
- New repo or separate project — all work stays in this repo on a feature branch
- Merging to main — stays on feature branch until server benchmarks confirm value

## Context

- Repository: `leasing` at `/Users/jenyafutrin/workspace/coding_projects/leasing`
- Branch: `claude/qwen-voice-next` (based on `codex/split-voice-providers`)
- Playbook: `docs/voice_ai_playbook_2026-03-25.md` is the authoritative planning document
- Prior work done with OpenAI Codex through March 2026 (Yandex experiment, split-brain providers, stack scripts)
- Target server: A100 80GB (Vast.ai) or H100 NVL 94GB (Azure NC40ads_H100_v5)
- Russian language is the primary target for all speech and text
- Current brain default: `Qwen/Qwen3-30B-A3B` via vLLM
- Existing test suite in `rag_demo_system/tests/` validates provider contracts

## Constraints

- **GPU budget**: A100 80GB is sufficient for staged benchmarking but not for co-hosting all models simultaneously. Swap models between tests on single GPU.
- **Privacy**: Proprietary license; on-premises deployment required. No external API dependencies in the production path.
- **Branch isolation**: All work on `claude/qwen-voice-next`. Never merge to main until benchmarks confirm value.
- **Benchmark discipline**: Change one component at a time. Same UI, same question set, same timers across all tests.
- **Russian quality**: All STT/TTS models must have explicit Russian language support. NVIDIA Canary-Qwen-2.5B is English-only; do not use.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Split pipeline as production track | Strongest RAG control, easiest debugging, easiest component swapping | -- Pending |
| Qwen3-Omni as experimental only | Weaker grounding, harder debugging; must prove itself first | -- Pending |
| Upgrade TTS first, brain second, STT third | TTS is highest-ROI single swap; brain upgrade is natural path; STT needs comparative benchmark | -- Pending |
| Hybrid Omni mode before pure realtime | Prompt-injected context is minimum viable RAG for Omni; pure realtime weakens control too much | -- Pending |
| A100 80GB or H100 94GB as server | A100 80GB enough for staged testing; Azure H100 is cleanest single-GPU option on Azure | -- Pending |
| Stay in existing repo, new branch | UI, transport, toggles, deployment scripts already exist; rebuilding wastes time | -- Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? Move to Out of Scope with reason
2. Requirements validated? Move to Validated with phase reference
3. New requirements emerged? Add to Active
4. Decisions to log? Add to Key Decisions
5. "What This Is" still accurate? Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check: still the right priority?
3. Audit Out of Scope: reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-25 after milestone v1.0 initialization*
