# Phase 2: Voice Provider Adapters - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Build Qwen3-TTS sidecar + adapter, Qwen3-ASR sidecar + adapter, and Voxtral adapter; all pass contract tests and appear in the frontend selector. No new pipeline architecture, no benchmark runner, no brain changes.

</domain>

<decisions>
## Implementation Decisions

### Voxtral Adapter Approach
- **D-01:** Researcher must investigate whether Voxtral can be self-hosted before planning the adapter. If self-hostable: build a local sidecar (matching other adapters). If not: build a thin Mistral cloud API client marked as benchmark-only (not production, violates privacy constraint). Outcome depends on research findings.

### Fallback Policy
- **D-02:** Hard fail when a selected provider is unavailable. If the user selects Qwen3-ASR and the sidecar is down, the voice turn fails with a clear error message (e.g., "Qwen3-ASR service unavailable"). No silent substitution to another provider. This protects benchmark integrity (VPROV success criteria #5).

### Sidecar Service Design
- **D-03:** Each new model gets its own standalone FastAPI sidecar script with its own Python venv. Matches the existing CosyVoice/SenseVoice pattern. No shared sidecar framework or multi-model service. Each sidecar exposes `/health`, `/transcribe` (STT) or `/speak` (TTS) endpoints following the existing contract.
- **D-04:** Per-service venvs are mandatory. The shared `rag_demo_system` venv must not have its `transformers==4.37.2` pin modified. Each sidecar installs its own model-specific dependencies in isolation.

### Model Version Pinning
- **D-05:** Researcher must confirm exact HuggingFace repo IDs, native sample rates, and recommended loading libraries (e.g., faster-whisper vs. raw transformers for Qwen3-ASR) before planning begins. Do not hardcode playbook model names; they are LOW confidence.

### Claude's Discretion
- Sidecar script internal structure (how model loading, warmup, and request handling are organized within each script)
- HTTP timeout values and request payload formats (follow existing patterns)
- Specific error message wording for hard-fail scenarios

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Voice AI Playbook
- `docs/voice_ai_playbook_2026-03-25.md` -- Authoritative planning document. Sections on TTS candidates (Qwen3-TTS-12Hz-1.7B), STT candidates (Qwen3-ASR-1.7B, Voxtral Realtime), Phase A step 4-6, and Source Notes with official repo URLs.

### Existing Voice Adapter Code
- `rag_demo_system/backend/voice_adapters.py` -- Current adapter dispatch logic. `transcribe_audio()` fallback chain and `synthesize_audio_with_provider()` if/elif dispatch. New adapters must integrate into these functions.
- `rag_demo_system/backend/voice_session.py` -- VoiceSession dataclass with stt_provider, tts_provider fields and stack_id property.
- `rag_demo_system/backend/app.py` -- WebSocket handler with session.update flow, `_voice_pipeline()` mapping, and structured JSON log emission. Lines ~638-682 for session.update handling.

### Contract Tests
- `rag_demo_system/tests/test_voice_adapters_official.py` -- Existing contract test patterns. New adapters must have equivalent tests (VPROV-04 gate).

### Existing Sidecar Examples
- `rag_demo_system/backend/yandex_speechkit.py` -- Example of a provider module with transcribe/synthesize functions.
- `rag_demo_system/backend/yandex_realtime.py` -- Example of a WebSocket-based provider relay.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `voice_adapters.py:transcribe_audio()` -- Fallback chain dispatcher for STT; new STT providers (qwen3_asr, voxtral) need branches added here
- `voice_adapters.py:synthesize_audio_with_provider()` -- TTS dispatcher; new TTS provider (qwen3_tts) needs a branch added here
- `voice_adapters.py:_service_status()` -- Health check helper; new sidecars get health entries via `build_voice_statuses()`
- `voice_adapters.py:_pcm16_b64_to_wav_bytes()` -- PCM16-to-WAV converter; may be reusable for audio format conversion
- `test_voice_adapters_official.py:FakeResponse` -- Test helper class for mocking HTTP responses

### Established Patterns
- **Env var convention:** `{NAME}_BASE_URL` for each sidecar (e.g., `COSYVOICE_BASE_URL`, `SENSEVOICE_BASE_URL`)
- **API style flag:** `{NAME}_API_STYLE` env var for providers with multiple API formats (e.g., `SENSEVOICE_API_STYLE=official`)
- **Response contract:** STT returns `{"text": "...", "provider": "..."}`, TTS returns `{"audio_b64": "...", "sample_rate_hz": N, "provider": "...", "session_id": "..."}`
- **Health endpoint:** Each sidecar serves `GET /health` returning 200 when ready

### Integration Points
- `app.py:_voice_pipeline()` -- Maps high-level `voice_provider` to `(stt, tts)` pair; may need updating if new providers are added as voice_provider options
- `app.py` WebSocket handler -- `session.update` already handles arbitrary `stt_provider` and `tts_provider` strings; no changes needed there
- Frontend `buildSessionUpdate()` in `app.js` -- Sends current dropdown values; dropdown `<option>` elements need adding for new providers
- `build_voice_statuses()` -- Needs new entries for qwen3_asr, qwen3_tts, and voxtral health checks

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond what's in the playbook and decisions above. Open to standard approaches for sidecar implementation.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 02-voice-provider-adapters*
*Context gathered: 2026-03-25*
