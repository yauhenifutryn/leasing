# Phase 4: Qwen3-Omni Hybrid - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 04-qwen3-omni-hybrid
**Areas discussed:** Audio pipeline path, RAG injection strategy, Timing instrumentation, Serving infrastructure

---

## Audio Pipeline Path

| Option | Description | Selected |
|--------|-------------|----------|
| Text-in, audio-out | Keep existing STT to transcribe first. Feed text + RAG chunks to Omni as text prompt. Omni generates audio response directly. | |
| Audio-in, audio-out | Send raw audio to Omni. STT runs only for RAG text query. Omni hears the question natively. | |
| Text-in, text-out + separate TTS | Use Omni as brain replacement only. Keep existing STT and TTS. | |

**Initial clarification:** User asked why we would use STT before Omni when the whole point is native audio. Explained that STT is only needed as a helper to give RAG a text search query, not as input to Omni itself. Omni still receives the original raw audio.

**User's choice:** Audio-in, audio-out + STT for RAG query
**Notes:** User initially questioned the text-in approach. After explanation that STT serves only the RAG retrieval step (not Omni input), user agreed this is the right approach. Also noted that if Omni struggles with audio input, that IS a valid benchmark result.

---

## RAG Injection Strategy

### Grounding Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Strict grounding | Answer ONLY from provided context. Refuse if not covered. | ✓ |
| Soft grounding | Prefer context but may supplement with general knowledge. | |
| You decide | Claude picks based on Omni's instruction-following capability. | |

**User's choice:** Strict grounding
**Notes:** None

### Chunk Count

| Option | Description | Selected |
|--------|-------------|----------|
| Same as voice_fast profile | vector_top_k=3, bm25_top_k=1, final_top_n=2, reranker disabled. Fair comparison. | ✓ |
| Fewer chunks (final_top_n=1) | Smaller context, less chance of Omni ignoring grounding. | |
| You decide | Claude picks based on Omni's context window. | |

**User's choice:** Same as voice_fast profile
**Notes:** None

### Prompt Language

| Option | Description | Selected |
|--------|-------------|----------|
| Russian prompt | System prompt and grounding rules in Russian. Avoids code-switching. | ✓ |
| English prompt + Russian chunks | Instructions in English, chunks in Russian. Some models follow English better. | |
| You decide | Claude picks based on Omni's instruction-following. | |

**User's choice:** Russian prompt
**Notes:** None

---

## Timing Instrumentation

### Field Mapping

| Option | Description | Selected |
|--------|-------------|----------|
| Emit all 6 fields, collapse where needed | Keep same JSONL schema. llm_first_token and tts_first_chunk collapse to same timestamp. | ✓ |
| Emit all 6 fields + omni-specific extras | Same + bonus fields for Omni-specific analysis. | |
| You decide | Claude picks compatible mapping. | |

**User's choice:** Emit all 6 fields, collapse where needed
**Notes:** None

### STT Timing

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, real STT timing | stt_done reflects real STT run (used for RAG). Only llm_first_token and tts_first_chunk are collapsed. | ✓ |
| You decide | Claude picks honest timing representation. | |

**User's choice:** Yes, real STT timing
**Notes:** None

---

## Serving Infrastructure

### Serving Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone sidecar | FastAPI with own venv, transformers-based. Matches Phase 2 pattern. | ✓ |
| vLLM if supported, sidecar fallback | Research vLLM audio support first. Potentially faster. | |
| You decide | Claude picks based on research findings. | |

**User's choice:** Standalone sidecar
**Notes:** Avoids dependency on unconfirmed vLLM audio input support.

### API Design

| Option | Description | Selected |
|--------|-------------|----------|
| Single /chat endpoint | POST /chat: audio + chunks in, audio + text out. | ✓ |
| Separate /transcribe + /generate | Two endpoints, more granular but defeats Omni's purpose. | |
| You decide | Claude designs based on Omni's inference API. | |

**User's choice:** Single /chat endpoint
**Notes:** None

---

## Claude's Discretion

- Sidecar internal structure (model loading, warmup, audio preprocessing)
- Exact Russian system prompt wording
- Audio format conversion details
- Backend dispatch logic for Omni vs split pipeline
- Error message wording

## Deferred Ideas

None -- discussion stayed within phase scope.
