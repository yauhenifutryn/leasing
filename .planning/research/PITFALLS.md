# Pitfalls Research

**Domain:** On-premises Russian voice assistant: adding new speech model providers, GPU memory management, benchmark infrastructure, and Qwen3-Omni experiment to an existing split-pipeline voice system.
**Researched:** 2026-03-25
**Confidence:** MEDIUM — Based on direct code analysis of the existing system (`voice_adapters.py`, `app.py`, `voice_session.py`, supervisord config, playbook) plus training-data knowledge of vLLM, CUDA memory management, WebSocket timing, and voice benchmarking patterns. Qwen3-TTS, Qwen3-ASR, Voxtral, and Qwen3-Omni are relatively new models and some implementation details below are flagged where training-data confidence is low.

---

## Critical Pitfalls

### Pitfall 1: New adapter returns different audio format than the existing `audio_b64 + sample_rate_hz` contract

**What goes wrong:**
`synthesize_audio_with_provider` in `voice_adapters.py` always returns a dict with `audio_b64` (PCM16, base64-encoded) and `sample_rate_hz`. The frontend assumes this exact shape to schedule Web Audio playback. Qwen3-TTS and Voxtral may return opus frames, float32 PCM, or chunked streaming responses instead of a single base64 blob. If the adapter does not normalize to the contract, the frontend receives an empty or corrupt audio blob and plays silence — with no visible error.

**Why it happens:**
Developers add a new adapter, test that the HTTP call succeeds, and ship. They do not check what the frontend does with the `sample_rate_hz` field. The current code in `app.py` line 759 passes `audio_response.get("sample_rate_hz")` directly to the WebSocket JSON; if it is wrong or absent, the browser will decode at the wrong rate and produce distorted audio.

**How to avoid:**
Write a unit test that calls each new adapter and asserts the shape: dict has `audio_b64` (non-empty string), `sample_rate_hz` (integer, matches what the frontend's Web Audio decoder expects — currently 24000 Hz for the existing path), and `provider` key. Run this test in CI before integrating into `voice_adapters.py`. For streaming TTS (Qwen3-TTS produces chunks), buffer all chunks inside the adapter and return the assembled blob rather than exposing the streaming interface to `app.py` — unless you also update `app.py` to handle chunked sends.

**Warning signs:**
- Browser plays a very short burst of noise then stops
- `sample_rate_hz` in `response.done` WebSocket message is `null` or `0`
- Frontend console shows `AudioBuffer` decode errors
- `tts_failed` error in WebSocket `warning` type message even though TTS HTTP call returned 200

**Phase to address:**
Phase: TTS Upgrade (Qwen3-TTS adapter). Address before any end-to-end testing of the new provider.

---

### Pitfall 2: Streaming TTS first-chunk delay negates the latency advantage

**What goes wrong:**
Qwen3-TTS is marketed as a streaming, low-latency TTS. The existing pipeline synthesizes a full response blob and then sends it. If the adapter calls the Qwen3-TTS streaming endpoint but buffers the entire response before returning, the adapter is serial: `LLM finishes -> TTS starts -> TTS finishes -> audio sent`. This eliminates the streaming benefit and can add 400-900 ms compared to a non-streaming call because the HTTP streaming connection has overhead.

**Why it happens:**
The simplest adapter implementation buffers all chunks inside `synthesize_audio_with_provider` before returning. This matches the existing interface (which was designed for single-call, single-blob TTS like CosyVoice) but discards the streaming benefit.

**How to avoid:**
Either (a) pipeline TTS streaming with WebSocket sends so that the first audio chunk is sent to the browser before TTS finishes, or (b) measure the single-blob approach first and only invest in pipelined streaming if benchmarks show the wait is unacceptable. The playbook's sequencing (TTS first, then benchmark) is correct; do not assume streaming will be faster without measuring. If you choose pipelined streaming, the `app.py` WebSocket handler for `input_audio_buffer.commit` must be restructured to use `async for` over TTS chunks instead of awaiting a full blob.

**Warning signs:**
- `tts_first_chunk_at - llm_first_token_at` timing (from the playbook's secondary metrics) is close to `tts_total_ms`, meaning the client receives audio only after TTS completes
- Measured `tts_total_ms` is 800 ms+ for short (2-3 sentence) responses
- GPU utilization for TTS sidecar spikes and stays high throughout synthesis rather than falling off after the first chunk

**Phase to address:**
Phase: TTS Upgrade. Instrument first before deciding to restructure the streaming path.

---

### Pitfall 3: New STT adapter falls into the silent fallback chain and hides failures

**What goes wrong:**
`transcribe_audio` in `voice_adapters.py` has a fallback chain: preferred -> sensevoice -> whisper. If the new adapter (Qwen3-ASR or Voxtral) returns `{"text": ""}` (empty transcription) rather than raising an exception, the code falls through to the next provider silently. During benchmarks, the recorded provider in the log will say `qwen3_asr` but the actual transcription was done by sensevoice. Latency comparisons will be meaningless.

**Why it happens:**
The current code at line 150 checks `if data.get("text")` to decide whether to fall back. An adapter that returns HTTP 200 with an empty result (which both sensevoice and whisper can do for short or silent utterances) triggers the fallback. If a new adapter's server is misconfigured and returns 200 with no text, the benchmark will log the wrong provider.

**How to avoid:**
When benchmarking, disable fallback: call `transcribe_audio` with `preferred=<new_provider>` and remove `sensevoice` and `whisper` from the environment (`unset SENSEVOICE_BASE_URL`, `unset WHISPER_BASE_URL`). This forces the benchmark to use only the intended provider. Log `transcript.get("provider")` in every benchmark row and assert it matches the intended provider before including the row in analysis.

**Warning signs:**
- Provider key in `conversation.item.input_audio_transcription.completed` WebSocket message alternates unexpectedly between runs
- STT latency for the new provider is suspiciously similar to sensevoice
- New adapter's access log shows no requests during benchmark but transcriptions still complete

**Phase to address:**
Phase: STT Benchmark. Required before comparing Qwen3-ASR, Voxtral, and the baselines.

---

### Pitfall 4: GPU OOM when loading second model alongside the running brain on A100/H100

**What goes wrong:**
The A100 80GB (or H100 NVL 94GB) has enough VRAM for one large model but not for two simultaneously loaded in FP16. Qwen3.5-35B-A3B in FP16 uses approximately 70 GB VRAM. Adding Qwen3-TTS-1.7B (approximately 3-4 GB) and Qwen3-ASR-1.7B (approximately 3-4 GB) while keeping the brain loaded approaches the VRAM limit. Qwen3-Omni-30B-A3B is approximately 60 GB. Loading Omni alongside the split pipeline brain causes an OOM on an 80GB device and a near-OOM (requiring KV-cache tuning) on 94GB.

**Why it happens:**
VRAM estimates from model parameter counts use FP16 rule of thumb (2 bytes per parameter). But vLLM allocates KV-cache on top of model weights: a default `gpu_memory_utilization=0.9` means vLLM reserves 90% of VRAM, including KV-cache. Two vLLM processes with default settings will fight over VRAM if total weights approach 75% of VRAM. PyTorch CUDA allocator does not coordinate between processes.

**How to avoid:**
- Run only one large model at a time. The playbook's phased approach (TTS first, brain upgrade second, Omni as final experiment) is the correct sequencing. Do not load Omni and the split pipeline brain simultaneously.
- For the split pipeline, use a single vLLM instance for the brain and separate lightweight model servers (not vLLM) for TTS and STT. Qwen3-TTS and Qwen3-ASR at 1.7B can be served with HuggingFace `transformers` directly or a dedicated TTS/ASR inference server without vLLM's KV-cache overhead.
- Set explicit `gpu_memory_utilization` in vLLM. For the brain model, `0.85` is safer than the default `0.90` when cohosting small sidecar models.
- When swapping to Omni for experiments: stop the brain vLLM process via supervisord (`supervisorctl stop qwen`), confirm `nvidia-smi` shows expected VRAM freed before launching Omni. The current supervisord config (`qwen` program) supports this pattern.

**Warning signs:**
- `nvidia-smi` shows > 90% VRAM before Omni launch
- `CUDA out of memory` in `qwen.log` or `sensevoice.log` (sidecar OOM due to fragmentation)
- vLLM startup log: "KV cache blocks: 0" — means no cache at all was allocated, model will barely serve one request
- Process restarts in supervisord immediately after new model starts

**Phase to address:**
Phase: Brain Upgrade (switching to Qwen3.5-35B-A3B) and Phase: Omni Experiment. OOM prevention must be in the launch checklist for both phases.

---

### Pitfall 5: Timing instrumentation adds async blocking that changes the latency being measured

**What goes wrong:**
Adding `time.perf_counter()` calls and log writes inside the async WebSocket handler in `app.py` is safe as long as logging is non-blocking. But if timing instrumentation writes to disk synchronously (e.g., `state.log(...)` inside the hot path), or if it triggers a serialization step on large payloads (e.g., recording full `used_knowledge` chunks per turn in the benchmark log), the instrumentation adds measurable latency to the voice turn — particularly on low-throughput SSD or networked storage. The benchmark then measures the instrumented system, not the production system.

**Why it happens:**
The `state.log` call is already in the hot path (line 486 in `app.py`). Benchmark logging adds more I/O to the same path. Developers add benchmark fields (`question_id`, `stack_id`, `retrieved_chunks`, `timing_breakdown`) without checking whether `state.log` is synchronous or whether the storage layer is fast enough.

**How to avoid:**
- Inspect `StateStore.log` implementation to confirm whether it is synchronous file I/O. If it is (likely for the current simple implementation), add a lightweight in-memory ring buffer for timing events and flush to disk after the WebSocket response is sent, not before.
- For benchmark runs specifically, write timing data to a separate append-only JSONL file in a RAM-backed directory (`/dev/shm` on Linux) and sync to disk after each benchmark session, not each turn.
- Time the overhead: run 10 turns with and without benchmark instrumentation and verify the delta is under 5 ms before treating the instrumented measurements as valid.

**Warning signs:**
- `state.log` or file write appears in flame graph between `tts_first_chunk_at` and `playback_started_at`
- Benchmark runs show consistently 20-50 ms higher latency than non-benchmark runs with the same stack
- Disk I/O spikes in `iostat` during voice turns

**Phase to address:**
Phase: Timing and Benchmark Instrumentation. Must be validated before Phase 0 (Baseline Control) is recorded.

---

### Pitfall 6: Qwen3-Omni hybrid mode ignores retrieved context and hallucinates from pretraining

**What goes wrong:**
In hybrid Omni mode, retrieved chunks are injected into the prompt context. The model is instructed to "answer only from the following fragments." But Qwen3-Omni in audio-in/audio-out mode processes both the audio stream and the text context simultaneously. The attention mechanism may not weight text-injected context as strongly as the audio input, especially when the audio question is phrased in a way that maps to a pretraining fact. The model generates a confident-sounding Russian answer from memory that does not match the KB — with no hallucination signal.

**Why it happens:**
Native audio-in models are not fine-tuned on the same strict RAG instruction-following pattern as text-only models. The `"answer only from context"` instruction is well-established for text-only LLMs but is undertested for multimodal audio input models. The grounding behavior is also sensitive to how chunks are formatted in the prompt (plain text vs. XML tags vs. numbered lists) and whether the audio question is explicitly repeated as text in the context.

**How to avoid:**
- Use the split pipeline as the control. Always run the same question set through both the Omni hybrid path and the split pipeline path in parallel and compare answers against a reference answer set derived from the actual KB.
- Add a strict refusal test: include 5-10 questions where the correct answer is explicitly not in the KB. If Omni hybrid mode answers these from pretraining, hybrid mode grounding is failing.
- Keep chunk count small (the playbook says 2 final chunks at voice_fast settings). Larger context injections can paradoxically degrade grounding in multimodal models by diluting the instruction signal.
- Log retrieved chunk IDs and compare against Omni's answer. If the answer mentions facts not present in retrieved chunks, it is a grounding failure.

**Warning signs:**
- Omni hybrid answers out-of-scope questions with confident financial product details not in the KB
- Omni answers differ from split pipeline for the same question and KB query
- Russian answer contains loanword or phrasing not in the KB text but typical of LLM pretraining on financial Russian text
- Retrieved chunks shown in logs do not support the claim in the generated answer

**Phase to address:**
Phase: Omni Experiment. Do not proceed to pure native realtime mode unless hybrid mode passes grounding evaluation.

---

### Pitfall 7: Russian-specific TTS/STT quality regressions that only appear in production phrases

**What goes wrong:**
Qwen3-TTS and Qwen3-ASR may produce acceptable quality on benchmark sentences (short, clean Russian text/audio) but degrade on production-specific content: leasing terminology (`лизингополучатель`, `аванс`, `выкупная стоимость`), Russian numeral phrases (`до 84 месяцев`, `от 15% первоначального взноса`), mixed Russian/number constructions. This is not caught by WER metrics on standard Russian test sets.

Voxtral is Mistral's speech model and its Russian quality is not confirmed at MEDIUM confidence based on available training data. The claim that it handles Russian well requires direct testing.

**Why it happens:**
Open-weight TTS models are primarily trained on English and Mandarin. Russian support may be bolted on via a smaller fine-tuning set that does not include domain-specific vocabulary. STT models may similarly have lower coverage of financial Russian. Benchmark datasets built from Wikipedia or news do not capture leasing domain vocabulary.

**How to avoid:**
- Build the benchmark question set (as described in the playbook) specifically around the actual Micro Leasing KB vocabulary. Include exact phrases from the KB as both audio input (recorded by a native speaker) and expected TTS output text.
- For TTS evaluation: run Qwen3-TTS on a list of 20 KB-derived sentences and have a native Russian listener score naturalness on a 1-5 scale. Include at least 5 sentences with numerals and 5 with domain-specific terms.
- For STT evaluation: record the same 20 sentences and measure WER against them. Compare Qwen3-ASR, Voxtral, and the Yandex SpeechKit baseline.
- Do not accept "supports Russian" from documentation as confirmation. Test the specific vocabulary.

**Warning signs:**
- TTS output has hesitations or incorrect stress on compound words (`лизинг-получатель` pronounced as two words)
- STT transcribes leasing terminology incorrectly (e.g., `аванс` -> `ав анс`, `лизинг` -> `лизин`)
- WER on KB-derived sentences is significantly higher than on standard Russian test sentences

**Phase to address:**
Phase: TTS Upgrade and Phase: STT Benchmark. Add domain-specific test items to the benchmark dataset before starting either phase.

---

### Pitfall 8: Brain model upgrade changes answer length and style, breaking voice latency targets

**What goes wrong:**
Qwen3.5-35B-A3B may produce longer answers than Qwen3-30B-A3B for the same prompts, or it may have different TTFB characteristics under vLLM. Longer answers increase TTS synthesis time, which increases end-to-end latency. The `concise_sentences_min`/`concise_sentences_max` settings in `settings.llm` constrain length, but prompt sensitivity varies between model versions. If the new brain answers in 4-5 sentences where the old one answered in 2-3, every TTS call gets longer.

**Why it happens:**
Model upgrades are tested for answer quality, not for token length distribution. The length constraint is a soft instruction, not a hard token limit enforced by the model. Different model versions have different instruction-following sensitivity.

**How to avoid:**
- After switching to Qwen3.5-35B-A3B, run the full benchmark question set and record the distribution of answer word counts alongside latency. If mean word count increases by more than 20%, tune the length instruction in the system prompt or reduce `fast_max_tokens` for the voice path.
- Keep Qwen3-30B-A3B running concurrently (in its vLLM instance) during the initial brain comparison phase so you can A/B the same question without model teardown.
- Measure TTFB separately from total generation time. A slower TTFB from the new brain hits perceived latency more than a longer total generation time.

**Warning signs:**
- Benchmark answers for the new brain average more than 50 words where the old brain averaged 30-35
- `llm_ttfb_ms` increases by more than 100 ms on the new brain versus the old brain
- End-to-end latency regresses past the 1.5s target despite no changes to STT or TTS

**Phase to address:**
Phase: Brain Upgrade. Record length distribution and TTFB before declaring the upgrade successful.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Buffer full TTS blob instead of streaming chunks | Simpler adapter code, matches existing interface | Adds 300-800 ms to perceived latency for Qwen3-TTS | Acceptable for baseline benchmarking; revisit if latency target not met |
| Hardcode sample rate in new adapter (e.g., 22050) | One less config variable | Frontend plays distorted audio if model changes default output rate | Never — always read from model response or make it a config constant |
| Run Omni and split-pipeline brain on same GPU simultaneously | No model teardown between tests | OOM crash or severe throughput degradation; corrupts benchmark data | Never during benchmarking |
| Use `time.time()` instead of `time.perf_counter()` for latency | Minimal | OS clock resolution too coarse for sub-millisecond measurements; results look identical | Never for timing instrumentation |
| Log benchmark results to same `state.log` as production logs | No new logging code | Hard to extract benchmark rows later; production log grows unboundedly | Acceptable for MVP benchmark; extract to a separate file before running full matrix |
| Skip warm-up turns in benchmark | Faster to run | First 1-3 requests show inflated latency due to model KV-cache cold start; averages are skewed | Never — always warm up with 3 discarded turns per stack configuration |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Qwen3-TTS via HuggingFace transformers | Load model in the same Python process as the FastAPI backend | Run as a separate sidecar process (existing supervisord pattern); FastAPI backend calls it via HTTP like cosyvoice |
| Qwen3-ASR | Pass raw PCM bytes instead of WAV-wrapped audio | Always wrap in WAV container with correct headers before sending; the existing `_pcm16_b64_to_wav_bytes` helper does this — reuse it |
| Voxtral | Use the Mistral API endpoint instead of local model | For on-prem benchmark, you must host the weights locally; verify the license allows on-premises deployment before committing resources |
| vLLM for brain model | Start vLLM without `--gpu-memory-utilization` tuning | Set `--gpu-memory-utilization 0.85` when cohosting sidecar models; set `--max-model-len` to limit KV-cache allocation |
| Supervisord for new model sidecars | Add new programs without `autostart=false` | New programs should default to `autostart=false`; the stack selector script (`stack.sh` / `stack_cli.py`) should explicitly start only what the current stack needs |
| Qwen3-Omni vLLM serving | Use same vLLM config as text-only brain | Omni requires audio input tensor support; verify which vLLM version supports Omni's audio modality and whether tensor-parallel config changes are needed |
| WebSocket timing instrumentation | Record timestamps with wall-clock `datetime.now()` | Use `time.perf_counter()` for all intra-request timings; use wall clock only for the session-level log entry timestamp |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| No warm-up before benchmark run | First 3 rows show 2-3x higher latency than subsequent rows; averages are inflated | Always send 3 discarded turns to each stack before recording benchmark data | Every benchmark run |
| Benchmarking over the local network with WiFi | Round-trip jitter of 5-30 ms per turn; latency variance swamps model differences | Use wired connection or localhost for all benchmark runs; if server is remote, use SSH port-forward to localhost | Always — any WiFi in the path |
| Qdrant reindex during benchmark | Retrieval latency spikes during reindex; benchmark rows become non-comparable | Lock KB index before starting benchmark run; do not trigger reindex between configurations | Any time KB is modified near a benchmark run |
| CosyVoice model cold cache on first TTS call | First synthesis takes 2-4x longer than subsequent calls | Warm up TTS with a silent or throwaway synthesis before recording | Every server restart |
| Python GIL in synchronous STT/TTS HTTP calls inside async handler | Other WebSocket sessions stall during a long STT or TTS call | Wrap blocking HTTP calls in `asyncio.get_event_loop().run_in_executor(None, ...)` if concurrency becomes a requirement; for single-user benchmark, this is low priority | At 2+ concurrent voice sessions |
| Recording timings across timezone-mismatched machines | Log timestamps do not align between client-side and server-side events | Use only server-side `perf_counter` deltas; never mix client JS timestamps with server Python timestamps |  Always if client clock is not NTP-synchronized |

---

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Exposing new model sidecar ports (TTS, STT, Omni) on the public interface | Unauthenticated inference calls from the internet burn GPU quota and expose voice data | Bind all sidecar ports to `127.0.0.1` in the sidecar launch command; access only through FastAPI backend |
| Logging full audio base64 blobs to state log | Conversation audio stored permanently in plaintext `.state/` directory | Never log `audio_b64` in state logs; log only timing, provider, and session IDs |
| Using Yandex SpeechKit with real user audio during benchmark | Real user audio sent to Yandex cloud; GDPR and Russian personal data law implications | Benchmark with synthetic or pre-recorded test audio, not live user sessions |
| Storing benchmark JSONL with transcripts on a public-facing path | User questions and assistant answers become accessible via the HTTP static server | Store benchmark output outside the `frontend/` directory; the current `frontend/` is served by `http.server` |

---

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Switching voice provider mid-session without resetting audio buffer state | User hears audio from previous provider overlapping new provider response | Clear `audio_chunks` and reset `VoiceSession` state on `session.update` event; currently `audio_chunks` in `app.py` is not cleared on provider switch |
| TTS silence gap between last LLM token and first audio chunk | User perceives 500-1000 ms dead time after they stop speaking | Pipeline LLM token streaming to TTS chunk generation; or reduce TTS minimum chunk size |
| No user-facing indicator of which voice stack is active | Developer testing with multiple providers loses track of which result belongs to which stack | Show `voice_provider` value from `session.updated` WebSocket message in the frontend provider label; the selector already exists but may not update the display on switch |
| Russian numerals read as digit strings by TTS | User hears "8 4 months" instead of "восемьдесят четыре месяца" | Pre-process LLM answer text with a Russian number-to-word normalizer before passing to TTS |

---

## "Looks Done But Isn't" Checklist

- [ ] **Qwen3-TTS adapter:** Returns non-empty `audio_b64` for a 2-sentence Russian text, `sample_rate_hz` is correct and non-null, `provider` key is present — verify with a unit test, not just a manual curl.
- [ ] **Qwen3-ASR adapter:** `provider` key in the return dict matches `qwen3_asr`, not a fallback; verify by running the adapter with `SENSEVOICE_BASE_URL` and `WHISPER_BASE_URL` unset.
- [ ] **Voxtral adapter:** Weights are actually hosted locally and the adapter hits the local endpoint — check adapter's base URL env var points to `127.0.0.1`, not a Mistral cloud URL.
- [ ] **Brain upgrade timing:** `llm_ttfb_ms` measured and recorded for both Qwen3-30B-A3B and Qwen3.5-35B-A3B before claiming the upgrade is complete.
- [ ] **Omni grounding test:** Out-of-scope questions return the strict refusal text, not a hallucinated answer — test with at least 5 out-of-scope questions before any demo.
- [ ] **Benchmark warm-up:** First 3 turns are discarded in benchmark JSONL — verify row numbering starts at turn 4 or that the first 3 rows are flagged `"warmup": true`.
- [ ] **VRAM headroom:** `nvidia-smi` shows at least 8 GB free after all planned models for a given stack configuration are loaded — record this as part of stack launch checklist.
- [ ] **Supervisord new programs:** New model sidecars (qwen3_tts, qwen3_asr, voxtral) added to `supervisord.conf` with `autostart=false` — never autostart on server reboot without explicit stack selection.
- [ ] **Audio buffer cleared on provider switch:** When `session.update` changes `voice_provider`, `audio_chunks` list in `app.py` is cleared — currently this does not happen (code review confirms `audio_chunks` persists across provider switches within the same WebSocket session).
- [ ] **Benchmark results not served by static frontend:** Benchmark JSONL output path is outside `frontend/` and not accessible via `http.server` on port 8081.

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| New adapter ships wrong audio format | MEDIUM | Roll back adapter to previous provider; fix format conversion in adapter and re-test with unit test before merging |
| GPU OOM during model swap | LOW | `supervisorctl stop <program>`; confirm VRAM freed with `nvidia-smi`; restart target model with lower `--gpu-memory-utilization` |
| Benchmark data contaminated by fallback chain | LOW | Identify affected rows by checking `provider` field; re-run those turns with sidecar ENV vars restricted to target provider only |
| Benchmark data contaminated by no warm-up | LOW | Discard first 3 rows per stack; re-run or adjust the analysis script to skip them |
| Omni hybrid mode hallucination confirmed | LOW (experimental track) | Stop Omni experiment; keep split pipeline as production baseline; add grounding test to Omni go/no-go checklist |
| Brain upgrade causes latency regression | MEDIUM | Revert to Qwen3-30B-A3B; investigate `llm_ttfb_ms` difference; consider `--max-model-len` reduction or `fast_max_tokens` reduction before retrying |
| TTS installs into the vLLM process space and causes OOM | MEDIUM | Move TTS to a standalone process with its own Python environment; do not share vLLM process with any sidecar model |
| Timing instrumentation adds > 5 ms overhead | LOW | Move `state.log` to async background task (use `asyncio.create_task`); log to in-memory buffer and flush after WebSocket response is sent |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Wrong audio format from new adapter | Phase: TTS Upgrade — before integration | Unit test asserting `audio_b64`, `sample_rate_hz`, `provider` shape |
| Streaming TTS buffered, no latency gain | Phase: TTS Upgrade — before benchmark | Measure `tts_total_ms` vs first-chunk-to-browser delta |
| Fallback chain hides STT provider identity | Phase: STT Benchmark — before any comparison run | Assert `provider` field in every benchmark row matches intended provider |
| GPU OOM on model swap | Phase: Brain Upgrade and Phase: Omni Experiment | `nvidia-smi` free VRAM check in stack launch runbook |
| Timing instrumentation adds latency | Phase: Timing/Benchmark Setup — before Phase 0 | Compare 10 turns with/without instrumentation; delta < 5 ms |
| Qwen3-Omni hybrid mode grounding failure | Phase: Omni Experiment | Out-of-scope question test set; compare answers to KB |
| Russian domain vocabulary TTS/STT errors | Phase: TTS Upgrade and Phase: STT Benchmark | Domain-specific sentence list scored by native speaker |
| Brain upgrade changes answer length/latency | Phase: Brain Upgrade | Answer word count distribution; `llm_ttfb_ms` comparison |
| Audio buffer not cleared on provider switch | Phase: TTS Upgrade or Provider Integration | Integration test: switch provider mid-session, verify no audio bleed |
| Benchmark warm-up not enforced | Phase: Benchmark Setup — before any phase | First 3 rows per stack flagged or discarded in analysis |

---

## Sources

- Direct code analysis: `rag_demo_system/backend/app.py`, `voice_adapters.py`, `voice_session.py`, `scripts/supervisord.conf` (leasing repo, branch `codex/split-voice-providers`, commit `9ef1b3d`)
- Playbook: `docs/voice_ai_playbook_2026-03-25.md` (internal, this repo)
- vLLM GPU memory allocation behavior: training data, MEDIUM confidence — verify against vLLM docs for the specific version deployed
- Qwen3-TTS streaming API behavior: training data, LOW confidence — verify against official Qwen3-TTS documentation and HuggingFace model card when integrating
- Voxtral on-premises licensing and Russian quality: LOW confidence — requires direct model card review before committing to implementation
- Qwen3-Omni hybrid mode grounding behavior: LOW confidence — no published evaluation on Russian RAG tasks found; treat as unknown until tested
- WebSocket async blocking patterns in FastAPI: training data, HIGH confidence

---
*Pitfalls research for: on-premises Russian voice assistant (leasing/rag_demo_system)*
*Researched: 2026-03-25*
