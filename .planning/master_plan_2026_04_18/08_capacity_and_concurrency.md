# Section 8 — Capacity, concurrency, multi-product platform

**Status**: pending. Authored 2026-05-01.
**Prereqs**: voice baseline stable (currently `f14ffdc`, ANALYSIS.md Batch 1 applied at `9ba36d5`). No code prereqs from earlier sections; planning-only until measurement work begins.
**Estimated effort**: discovery 1-2 days; full multi-product rollout depends on client decisions.
**Base commit**: stable head on `feature/voice-pipeline`.

## Goal

Answer four open client / platform questions with measured numbers and concrete production patterns:

1. How many concurrent voice calls does the current H100 setup handle before latency or quality degrades?
2. How does an H200 upgrade change that ceiling?
3. How does the chat widget (planned) coexist with voice on the same GPU?
4. How do future products (internal RAG agent, outbound debt-collector agent) share or split compute?

Plus: what is the production overflow strategy when concurrency is hit?

## Why now

- Client is making a hardware decision (H100 vs H200) and needs a defensible number.
- Chat widget is a near-term ask; concurrency interaction with voice must be designed before, not after, build.
- Future workloads (internal RAG, outbound) will land on the same GPU; an "every product gets its own model instance" trajectory wastes the H100.
- Current "5-12 concurrent calls" is an estimate, not a measurement. The first thing this section ships is the measurement.

## Required memories

- `project_capacity_planning_2026_05_01.md` — anchor for this section; H100/H200 ceilings, KV-cache safe-space sizing, chat coexistence math
- `project_load_test_harness_planned.md` — SIPp + Prometheus harness scope
- `project_overflow_callback_strategy.md` — chosen overflow pattern (callback over queue)
- `project_chat_widget_strategy.md` — voice-to-chat derivation and concurrency model
- `project_multi_product_vllm_tiered.md` — shared big + small model serving plan
- `project_git_cleanup_decision_pending.md` — open decision on voice-pipeline branch lifecycle (not capacity-related but logged here as the parallel cleanup item)
- `reference_live_server_38_128_232_83.md` — live VM endpoints for harness wiring
- `feedback_check_existing_scripts_first.md` — `ls scripts/` before writing new ones
- `feedback_curl_test_vllm_params_before_shipping.md` — measure before shipping

## Primary skills

- Phase A (measurement): `superpowers:systematic-debugging`, `superpowers:verification-before-completion`
- Phase B (overflow + chat): `superpowers:writing-plans`, `superpowers:test-driven-development`
- Phase C (multi-product): `superpowers:brainstorming` then `superpowers:writing-plans`

## Phase A — Measure the real ceiling

**Effort**: 1-2 days. **Risk**: zero in production (test runs against staging window).

### A.1 Audio corpus from client

Ask client for 15-20 short Russian audio recordings of representative call patterns:

- price / loan calculation (`посчитайте лизинг на ...`)
- term-and-down-payment changes mid-call
- KB questions (offices, accounting handoff, early termination)
- mind-changes (`а если по-другому`)
- multi-step parameter updates
- one or two confused / hesitant callers

Each 30-90 seconds, 8 kHz mono WAV. Mixed speakers, mixed pace, mild background noise OK. Save under `tests/load/corpus/<scenario>.wav` (gitignored or split repo — decide at commit time).

### A.2 SIPp scenario per recording

`tests/load/scenarios/<scenario>.xml`. Each scenario REGISTERs a test SIP user, INVITEs the prod number, plays the WAV, hangs up after the bot stops responding.

Use one of the six existing test SIP credentials (memory `reference_jambonz_sip_credentials_38_128_233_130.md`). Reserve a separate credential pool for load testing so live client SIP is never blocked.

### A.3 Metrics endpoint

Add `/metrics` (Prometheus format) to the orchestrator. Exposed gauges + histograms:

- `voice_active_calls` (gauge)
- `voice_stage_latency_ms{stage="stt|classifier|rag|llm_ttft|tts_first|end_to_end"}` (histogram)
- `vllm_kv_cache_used_pct` (gauge, scraped from vLLM `/metrics`)
- `vllm_pending_requests` (gauge)
- `gpu_utilization_pct`, `gpu_mem_used_gb` (gauge, via `nvidia-smi` exporter or pynvml)
- `call_quality_failures_total{type="repeat_request|silence|wrong_answer"}` (counter — populated by post-call session_analyzer pass)

Grafana dashboard: one row per stage, x-axis = active calls. Visually obvious where the curve breaks.

### A.4 Ramp protocol

Run from a load box (laptop or small VM, not the GPU server):

```
1 call  for 60s
2 calls for 60s
5 calls for 60s
8 calls for 60s
10 calls for 60s
15 calls for 60s
20 calls for 60s
```

For each step record p50 / p95 / p99 of `voice_stage_latency_ms{stage="end_to_end"}`. Note the first step where p95 > 1500 ms or p99 > 3000 ms. That is the **measured concurrency ceiling**.

### A.5 Quality spot-check

At the discovered ceiling, dispatch 6-8 humans to call simultaneously. Listen to recordings. Confirm bot answers are coherent, not just fast. Recording-only tests miss real barge-in chaos.

### A.6 KV-cache safe-space sizing

Long calls hold growing KV cache. The ceiling discovered in A.4 assumes average call length. Re-run A.4 with **prolonged scenarios (5-10 min calls)** to find the conservative ceiling. The production limit is the conservative number.

Implementation defenses (apply regardless of test results):

- vLLM `max_model_len` per request capped at ~6K tokens.
- History pruning: when conversation exceeds ~5 min or ~15 turns, auto-summarize older turns; keep profile + summary + last 3 turns + current RAG context.
- Admission control: refuse new call if `vllm_kv_cache_used_pct > 70`. Below the hard cap, leaving 30% headroom.
- Hard call timeout: 15 min. Beyond that, escalate or hang up gracefully.

### A.7 H200 extrapolation

Once H100 ceiling is known, project H200 capacity:

- H200 has 1.76× memory (141 GB vs 80 GB) and 1.43× HBM bandwidth (4.8 TB/s vs 3.35 TB/s) over H100.
- For batched LLM serving (memory-bandwidth-bound), expect roughly 1.7-2× concurrent ceiling.
- Document the math in the diagnostic report; do not buy hardware on extrapolation alone if budget is tight — book an H200 trial day on Sesterce or similar.

**Phase A exit criterion**: measured H100 ceiling number documented; KV-cache safe-space defenses shipped; H200 projection written; client decision on hardware unblocked.

## Phase B — Overflow and chat coexistence

**Effort**: 2-3 days after Phase A. **Risk**: low (additive features behind feature flags).

### B.1 Overflow: callback queue

Chosen pattern: **callback request**, not hold-music queue.

When `voice_active_calls >= measured_ceiling`:

- Bot answers immediately with: "Все консультанты заняты, перезвоню в течение 3-5 минут на ваш номер. Верно ___?"
- Read number back digit-by-digit; reuse existing classifier confirmation pattern.
- On confirmation: enqueue callback in `callback_queue` (Redis or sqlite — decide based on infra simplicity).
- On hangup: send SMS confirmation via existing SMS channel: "Перезвоним на +375... в течение 5 минут".
- Worker process polls `callback_queue` when `voice_active_calls < ceiling - 1` (always leave one slot for incoming calls). Initiates outbound via Jambonz.
- Voicemail-aware: if no audio frames after 10s of "answer", drop SMS retry instead of redialing.

Files affected:
- `voice_pipeline/admission_control.py` (new) — concurrency gate.
- `voice_pipeline/callback_queue.py` (new) — enqueue + poll logic.
- `voice_pipeline/outbound_dialer.py` (new) — Jambonz outbound trigger.
- `prompts/system_prompt_ru_v2.txt` — new playbook entry for "all-busy" scenario.
- `tests/test_overflow_callback.py` (new) — TDD per CLAUDE.md.

### B.2 Chat widget — derived from voice system

~60-70% of voice code reusable for chat. Specifically:

**Reused as-is:**
- Classifier (SessionAgent 4B)
- RAG retriever
- Knowledge base
- Calculator tool
- SMS tool
- Profile/state management
- System prompt (with chat-mode toggle for medium-formality)

**Removed:**
- Whisper / SenseVoice (STT)
- Silero (TTS)
- Jambonz / SIP integration
- Barge-in handling
- VAD / speaker mode

**New:**
- WebSocket endpoint (`/chat/ws`) — streaming token output.
- Thin React widget (embeddable `<script>` tag) — message list, input, optional file attach.
- Session ID propagated via cookie or query param.
- Rate limiter per IP / session.

Estimated effort: 3-5 days for the backend integration + a basic widget; up to 2 weeks for a polished widget with branding.

### B.3 Chat-voice coexistence model

Same vLLM serves both. Voice is priority.

- vLLM request priority: voice = high, chat = normal.
- When `vllm_kv_cache_used_pct > 70`: chat returns "сервис временно занят, попробуйте через минуту"; voice still admits up to ceiling.
- Chat per-IP rate limit (e.g. 10 messages / min) to prevent a single chatter from monopolizing.
- Hard chat session cap (e.g. 100 concurrent sessions) as backstop. Realistically the H100 handles 50-200 chat sessions easily; 100 is conservative.

The realistic failure scenario: 50+ active chatters during peak voice load. In that case, chat degrades first (priority), voice continues. If chat degradation becomes routine, that is the trigger for tier-2 scaling (see Phase C).

### B.4 Tests

- `tests/test_admission_control.py` — voice rejected at ceiling.
- `tests/test_callback_queue.py` — enqueue, poll, outbound trigger, voicemail SMS.
- `tests/test_chat_voice_priority.py` — under simulated load, voice TTFT stays under threshold while chat slows.
- `tests/test_chat_rate_limit.py` — IP-level limit enforced.

**Phase B exit criterion**: callback flow tested end-to-end; chat widget MVP deployed to staging; combined-load test shows voice unaffected by chat surge up to design limits.

## Phase C — Multi-product model strategy

**Effort**: scoped per product. **Risk**: depends on product (internal RAG = low; outbound dialer = medium).

### C.1 Tiered model serving

Two vLLM instances on the same GPU, not one per product:

- **Big (Qwen3.5-30B-FP8)** — voice agent, outbound debt collector, complex chat.
- **Small (Qwen3-4B)** — internal docs RAG, simple chat, existing classifier (already on this).

Approximate footprint on H100 80 GB:
- Big weights: ~30 GB
- Small weights: ~6 GB
- STT (Whisper or SenseVoice): ~3-5 GB
- KV cache pool: ~30-35 GB free
- Headroom: ~5 GB

Comfortable on H100. Very comfortable on H200 (would allow ~70 GB KV cache).

### C.2 Per-product configuration

Each product is a thin orchestration layer over the shared vLLM:

- **Internal RAG agent**: separate Qdrant collection (`internal_docs`), small model, simple chat-only frontend, no voice. Effort: ~3-5 days assuming docs already structured.
- **Outbound debt collector**: big model, voice via Jambonz outbound, scripted opening + RAG-backed Q&A, integration with CRM for call-list source. Lower complexity than inbound (script-driven), but compliance-sensitive. Effort: ~1-2 weeks plus client compliance review.

### C.3 Scaling trigger

Single H100 (or H200) handles all four products until measured load exceeds:

- Voice ceiling discovered in Phase A; or
- KV cache headroom < 20% during normal hours; or
- Chat degradation becomes routine (per B.3).

At that point, options in order of operational simplicity:
1. H100 → H200 upgrade (if not already done).
2. Add a second GPU node, route products by dedicated assignment (e.g. voice on node 1, chat + internal RAG on node 2).
3. Horizontal scale the bottleneck product (multiple voice nodes behind a SIP load balancer).

Do not pre-build (2) or (3). Wait for the trigger.

**Phase C exit criterion**: per-product capacity model documented; scaling trigger thresholds wired into Grafana alerts; client briefed on when next hardware investment is needed.

## Files affected (summary)

### New
- `voice_pipeline/admission_control.py`
- `voice_pipeline/callback_queue.py`
- `voice_pipeline/outbound_dialer.py`
- `chat_widget/backend/ws_endpoint.py`
- `chat_widget/frontend/` (React)
- `tests/load/scenarios/*.xml` (SIPp)
- `tests/load/corpus/*.wav` (client-provided audio; consider git-lfs or external bucket)
- `tests/test_admission_control.py`
- `tests/test_callback_queue.py`
- `tests/test_chat_voice_priority.py`
- `tests/test_chat_rate_limit.py`
- `tests/test_overflow_callback.py`
- `monitoring/grafana_dashboards/voice_capacity.json`
- `monitoring/prometheus_rules/admission_alerts.yml`
- `docs/superpowers/capacity-measurement-<date>.md` (Phase A diagnostic)

### Modified
- `voice_pipeline/orchestrator.py` (admission gate, priority on vLLM requests)
- `voice_pipeline/conversation_state.py` (history pruning at length thresholds)
- `prompts/system_prompt_ru_v2.txt` (all-busy playbook entry)
- vLLM startup args: `--max-model-len 6000`, request-priority enabled
- `.env` template + provisioning: `MAX_CONCURRENT_CALLS`, `KV_CACHE_HEADROOM_PCT`, `CHAT_RATE_LIMIT_PER_IP`

### Reference
- `reference_live_server_38_128_232_83.md`
- `reference_jambonz_sip_credentials_38_128_233_130.md`
- existing `scripts/` helpers per `feedback_check_existing_scripts_first.md`

## Checkpoints

- [ ] CP-8.0 — Phase A audio corpus received from client; SIPp scenarios + metrics endpoint shipped.
- [ ] CP-8.1 — Phase A ramp test executed; H100 ceiling number documented; KV-cache defenses applied; H200 projection written.
- [ ] CP-8.2 — Phase B callback flow shipped; tests green; SMS confirmation working.
- [ ] CP-8.3 — Phase B chat widget MVP on staging; combined-load test passes.
- [ ] CP-8.4 — Phase C tiered model plan approved by client; first additional product (internal RAG) shipped or scoped.
- [ ] CP-8.5 — `project_capacity_v1_complete_<date>.md` memory written; this section closed.

## Rollback

- **Phase A**: read-only measurement; no production state changes. KV-cache defenses are additive; revert via env-var (`MAX_MODEL_LEN`, `KV_CACHE_HEADROOM_PCT`).
- **Phase B**: callback flow behind feature flag (`OVERFLOW_MODE=reject|queue|callback`); chat widget on separate endpoint, can be disabled by removing route. No coupling to existing voice path.
- **Phase C**: each product is an independent vLLM client; rollback = stop that client.

## What NOT to do in this section

- No buying H200 before Phase A measurement is complete.
- No queue-with-hold-music as primary overflow strategy; callback is the chosen pattern.
- No per-product dedicated vLLM instance unless tier (big vs small) actually differs.
- No horizontal scale build-out before single-node ceiling is hit and measured.
- No chat queue UX; chat scales by capacity or rate-limits, never by hold-music equivalent.

## Done criterion

All checkpoints green. Measured H100 ceiling documented and shared with client. Callback overflow live in production. Chat widget on staging or production. Tiered model serving plan approved. `MEMORY.md` updated with `project_capacity_v1_complete_<date>.md`.
