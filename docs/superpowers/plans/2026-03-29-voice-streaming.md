# Voice Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream LLM responses sentence-by-sentence to TTS, reducing voice latency from 10-50s to 1.5-3.5s.

**Architecture:** LLM tokens flow into a sentence detector that emits complete sentences. Each sentence is sent to TTS immediately via an asyncio queue. A consumer task processes sentences, sends audio deltas to the WebSocket, and prefetches the next sentence's audio (1-ahead). Barge-in cancels both the LLM stream and TTS queue.

**Tech Stack:** Python asyncio, existing FastAPI WebSocket, existing vLLM streaming API, existing TTS providers.

---

### Task 1: Create sentence detector module

**Files:**
- Create: `rag_demo_system/backend/sentence_detector.py`
- Create: `rag_demo_system/tests/test_sentence_detector.py`

- [ ] **Step 1: Write the failing tests**

```python
# rag_demo_system/tests/test_sentence_detector.py
from __future__ import annotations

import pytest
from backend.sentence_detector import SentenceDetector


class TestSentenceDetector:
    def test_simple_sentence(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Привет. ") == ["Привет."]

    def test_multiple_sentences(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Первое. Второе. ") == ["Первое.", "Второе."]

    def test_partial_no_emit(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Начало предложения") == []

    def test_incremental_tokens(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Лизинг") == []
        assert sd.feed(" доступен") == []
        assert sd.feed(". ") == ["Лизинг доступен."]

    def test_question_mark(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Какой аванс? ") == ["Какой аванс?"]

    def test_exclamation(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Здравствуйте! ") == ["Здравствуйте!"]

    def test_ellipsis(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Давайте уточним... ") == ["Давайте уточним..."]

    def test_abbreviation_no_split(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("т.е. это значит. ") == ["т.е. это значит."]

    def test_abbreviation_td(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("авто, техника и т.д. Далее. ") == ["авто, техника и т.д.", "Далее."]

    def test_usd_no_split(self) -> None:
        sd = SentenceDetector()
        assert sd.feed("Сумма 2000 USD. ") == ["Сумма 2000 USD."]

    def test_flush_remaining(self) -> None:
        sd = SentenceDetector()
        sd.feed("Неполное предложение")
        assert sd.flush() == "Неполное предложение"

    def test_flush_empty(self) -> None:
        sd = SentenceDetector()
        assert sd.flush() is None

    def test_flush_after_complete(self) -> None:
        sd = SentenceDetector()
        sd.feed("Готово. ")
        assert sd.flush() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rag_demo_system && .venv/bin/python -m pytest tests/test_sentence_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.sentence_detector'`

- [ ] **Step 3: Implement sentence detector**

```python
# rag_demo_system/backend/sentence_detector.py
from __future__ import annotations

import re

# Russian abbreviations that end with a period but are not sentence-endings.
_ABBREVS = re.compile(
    r"\b(т\.е|т\.д|т\.п|т\.к|и\.т\.д|и\.т\.п|г|руб|USD|EUR|BYN|др|пр|стр|ул|д|кв|корп)\.$",
    re.IGNORECASE,
)

# Sentence-ending punctuation followed by a space or end of string.
_SENT_END = re.compile(r"([.!?]|\.{3})\s")


class SentenceDetector:
    """Accumulates LLM tokens and emits complete sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Feed a token. Returns list of complete sentences (usually 0 or 1)."""
        self._buf += token
        sentences: list[str] = []
        while True:
            match = _SENT_END.search(self._buf)
            if not match:
                break
            end_pos = match.end()
            candidate = self._buf[:end_pos].strip()
            # Check if the period belongs to an abbreviation
            text_before_punct = self._buf[: match.start() + len(match.group(1))]
            if match.group(1) == "." and _ABBREVS.search(text_before_punct):
                # Not a real sentence end; keep accumulating.
                # But only skip this one match; there may be a real end later.
                # Move past this match and look for the next one.
                # We can't just break because "т.д. Далее. " has a real end.
                # Strategy: find the next potential end after this position.
                next_search = _SENT_END.search(self._buf, pos=end_pos)
                if not next_search:
                    break
                end_pos = next_search.end()
                candidate = self._buf[:end_pos].strip()
                text_before_punct = self._buf[: next_search.start() + len(next_search.group(1))]
                if next_search.group(1) == "." and _ABBREVS.search(text_before_punct):
                    break
            sentences.append(candidate)
            self._buf = self._buf[end_pos:]
        return sentences

    def flush(self) -> str | None:
        """Return any remaining text. Call when LLM stream ends."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining or None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd rag_demo_system && .venv/bin/python -m pytest tests/test_sentence_detector.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/sentence_detector.py rag_demo_system/tests/test_sentence_detector.py
git commit -m "feat: add sentence detector for voice streaming"
```

---

### Task 2: Add async streaming voice response function

**Files:**
- Modify: `rag_demo_system/backend/app.py`

This replaces the blocking `_voice_chat_streaming_sync` call with an async function that streams sentences to TTS via a queue.

- [ ] **Step 1: Add the async streaming function after `_voice_chat_streaming_sync`**

Add this function at line ~310 (after `_voice_chat_streaming_sync` ends, before the `chat` endpoint):

```python
async def _stream_voice_response(
    *,
    websocket: Any,
    session: Any,
    session_id: str,
    message: str,
    tts_provider: str,
    t_speech_stopped: float,
    t_stt_done: float,
    question_id: str,
) -> None:
    """Async voice response with sentence-level LLM->TTS streaming.

    As the LLM generates tokens, sentences are detected and sent to TTS
    immediately. Audio chunks are streamed to the browser via WebSocket.
    Supports barge-in: if session.interrupted is set, both LLM and TTS stop.
    """
    import asyncio
    from .llm import iter_openai_compatible_stream_events
    from .sentence_detector import SentenceDetector
    from .text_utils import clean_answer

    backend = session.backend
    brain_model = session.brain_model

    # --- RAG retrieval (same as before) ---
    retrieval = await asyncio.to_thread(
        engine.retrieve, message, True, True, session_id,
    )
    t_retrieval_done = time.time()
    timings = dict(retrieval.get("timings") or {})
    final_chunks = retrieval.get("final") or []

    if not retrieval.get("ok") or not final_chunks:
        answer = settings.app.strict_refusal_text
        await websocket.send_json({
            "type": "response.output_text.delta",
            "session_id": session_id,
            "delta": answer,
        })
        try:
            audio_resp = await asyncio.to_thread(
                synthesize_audio_with_provider, answer,
                session_id, tts_provider,
            )
            if audio_resp.get("audio_b64"):
                await websocket.send_json({
                    "type": "response.output_audio.delta",
                    "session_id": session_id,
                    "delta": audio_resp["audio_b64"],
                    "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                })
        except Exception:
            pass
        t_now = time.time()
        state.log({
            "event": "voice_turn", "question_id": question_id,
            "stack_id": session.stack_id, "session_id": session_id,
            "backend": backend, "brain_model": brain_model,
            "stt_provider": session.stt_provider, "tts_provider": tts_provider,
            "transcript": message, "speech_stopped": t_speech_stopped,
            "stt_done": t_stt_done, "retrieval_done": t_retrieval_done,
            "llm_first_token": t_retrieval_done, "tts_first_chunk": t_now,
            "playback_started": t_now, "primary_kpi_ms": (t_now - t_speech_stopped) * 1000,
        })
        await websocket.send_json({
            "type": "response.done", "session_id": session_id,
            "backend": backend, "used_knowledge": [],
            "citations": [], "timings": timings,
        })
        return

    # --- Build prompt (same logic as _voice_chat_streaming_sync) ---
    system_prompt = settings.app.system_prompt_path.read_text(encoding="utf-8")
    system_prompt += "\n\nСогласие на обработку данных уже получено, не запрашивай его."
    chat_session = state.get(session_id) or state.create(session_id)
    memory_block = build_memory_block(chat_session.transcript, settings.app.memory_turns)
    context_block = "\n\n".join(
        [f"[Fragment {i+1}]\n{c['text']}" for i, c in enumerate(final_chunks)]
    )
    expanded = any(trigger in message.lower() for trigger in settings.llm.expand_triggers)
    length_hint = (
        f"Ответ должен быть {settings.llm.concise_sentences_min}-{settings.llm.concise_sentences_max} коротких предложений."
        if not expanded
        else "Можно ответить подробнее, но только на основе контекста."
    )
    weak_context = bool(retrieval.get("weak"))
    weak_hint = (
        "Контекст может быть неполным. Дай ближайшую релевантную информацию из фрагментов, "
        "скажи, что точных данных может не хватать, и задай уточняющий вопрос.\n\n"
    ) if weak_context else ""
    user_prompt = (
        "Ответь строго на основе следующих фрагментов базы знаний. "
        "Если ответа нет - верни точный отказ.\n\n"
        f"{memory_block}{length_hint}\n\n"
        f"{weak_hint}{context_block}\n\nВопрос клиента: {message}"
    )
    effective_model = brain_model or settings.llm.fast_model or settings.llm.model
    effective_base_url = settings.llm.fast_base_url or settings.llm.base_url

    # --- Sentence queue: LLM produces, TTS consumes ---
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
    t_llm_first_token: float | None = None
    t_tts_first_chunk: float | None = None
    t_playback_started: float | None = None
    all_sentences: list[str] = []

    async def llm_producer() -> None:
        """Stream LLM tokens, detect sentences, put them on the queue."""
        nonlocal t_llm_first_token
        detector = SentenceDetector()
        try:
            stream = iter_openai_compatible_stream_events(
                base_url=effective_base_url, model=effective_model,
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.fast_max_tokens,
                timeout_sec=settings.llm.timeout_sec,
            )
            for event in stream:
                if session.interrupted:
                    break
                choice = (event.get("choices") or [{}])[0]
                token = (choice.get("delta") or {}).get("content") or ""
                if not token:
                    continue
                if t_llm_first_token is None:
                    t_llm_first_token = time.time()
                for sentence in detector.feed(token):
                    cleaned = clean_answer(sentence)
                    if cleaned:
                        all_sentences.append(cleaned)
                        await sentence_queue.put(cleaned)
        except Exception as exc:
            state.log({"event": "llm_error", "error": str(exc), "session_id": session_id})
        finally:
            # Flush remaining text
            remaining = detector.flush()
            if remaining and not session.interrupted:
                cleaned = clean_answer(remaining)
                if cleaned:
                    all_sentences.append(cleaned)
                    await sentence_queue.put(cleaned)
            await sentence_queue.put(None)  # sentinel

    async def tts_consumer() -> None:
        """Take sentences from queue, call TTS, send audio to browser."""
        nonlocal t_tts_first_chunk, t_playback_started
        while True:
            if session.interrupted:
                break
            sentence = await sentence_queue.get()
            if sentence is None:
                break
            # Send text immediately
            await websocket.send_json({
                "type": "response.output_text.delta",
                "session_id": session_id,
                "delta": sentence + " ",
            })
            # Call TTS
            try:
                audio_resp = await asyncio.to_thread(
                    synthesize_audio_with_provider, sentence,
                    session_id, tts_provider,
                )
                audio_b64 = audio_resp.get("audio_b64") or ""
                if audio_b64:
                    if t_tts_first_chunk is None:
                        t_tts_first_chunk = time.time()
                    await websocket.send_json({
                        "type": "response.output_audio.delta",
                        "session_id": session_id,
                        "delta": audio_b64,
                        "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                    })
                    t_playback_started = time.time()
            except Exception as exc:
                await websocket.send_json({
                    "type": "warning", "session_id": session_id,
                    "message": f"tts_failed: {exc}",
                })

    # --- Run producer and consumer concurrently ---
    session.assistant_speaking = True
    producer_task = asyncio.create_task(llm_producer())
    consumer_task = asyncio.create_task(tts_consumer())
    await asyncio.gather(producer_task, consumer_task)
    session.assistant_speaking = False

    # --- Log and finalize ---
    t_now = time.time()
    primary_kpi_ms = ((t_playback_started or t_now) - t_speech_stopped) * 1000
    used_knowledge = [
        {"text": c["text"], "chunk_id": c.get("chunk_id")} for c in final_chunks
    ]
    state.log({
        "event": "voice_turn", "question_id": question_id,
        "stack_id": session.stack_id, "session_id": session_id,
        "backend": backend, "brain_model": brain_model,
        "stt_provider": session.stt_provider, "tts_provider": tts_provider,
        "transcript": message, "speech_stopped": t_speech_stopped,
        "stt_done": t_stt_done, "retrieval_done": t_retrieval_done,
        "llm_first_token": t_llm_first_token or t_retrieval_done,
        "tts_first_chunk": t_tts_first_chunk or t_now,
        "playback_started": t_playback_started or t_now,
        "primary_kpi_ms": primary_kpi_ms,
    })
    await websocket.send_json({
        "type": "response.done", "session_id": session_id,
        "backend": backend, "used_knowledge": used_knowledge,
        "citations": [], "timings": timings,
    })
```

- [ ] **Step 2: Verify the function compiles (no syntax errors)**

Run: `cd rag_demo_system && .venv/bin/python -c "from backend.app import _stream_voice_response; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add rag_demo_system/backend/app.py
git commit -m "feat: add async _stream_voice_response with sentence streaming"
```

---

### Task 3: Wire streaming into the WebSocket handler

**Files:**
- Modify: `rag_demo_system/backend/app.py` (the `voice_ws` handler, lines 1007-1086)

- [ ] **Step 1: Replace the blocking call with the streaming function**

Replace this block (lines 1007-1086, the split pipeline path after the Omni check):

```python
                voice_result = await asyncio.to_thread(
                    _voice_chat_streaming_sync,
                    message=text,
                    session_id=session_id,
                    backend=session.backend,
                    brain_model=session.brain_model,
                )
                t_retrieval_done = voice_result.get("t_retrieval_done") or time.time()
                t_llm_first_token = voice_result.get("t_llm_first_token") or t_retrieval_done
                response = voice_result

                if isinstance(response, dict):
                    for action in session.on_provider_response(response):
                        await websocket.send_json(action)
                    answer_text = response.get("answer", "")
                    if answer_text:
                        await websocket.send_json(
                            {
                                "type": "response.output_text.delta",
                                "session_id": session_id,
                                "delta": answer_text,
                            }
                        )
                    try:
                        audio_response = synthesize_audio_with_provider(answer_text, session_id=session_id, preferred=tts_provider)
                        t_tts_first_chunk = time.time()
                        audio_b64 = audio_response.get("audio_b64") or ""
                        if audio_b64:
                            await websocket.send_json(
                                {
                                    "type": "response.output_audio.delta",
                                    "session_id": session_id,
                                    "delta": audio_b64,
                                    "sample_rate_hz": audio_response.get("sample_rate_hz"),
                                }
                            )
                            t_playback_started = time.time()
                        else:
                            t_playback_started = t_tts_first_chunk
                    except Exception as exc:  # noqa: BLE001
                        t_tts_first_chunk = time.time()
                        t_playback_started = t_tts_first_chunk
                        await websocket.send_json(
                            {
                                "type": "warning",
                                "session_id": session_id,
                                "message": f"tts_failed: {exc}",
                            }
                        )

                    primary_kpi_ms = (t_playback_started - t_speech_stopped) * 1000
                    state.log({
                        "event": "voice_turn",
                        "question_id": question_id,
                        "stack_id": session.stack_id,
                        "session_id": session.session_id,
                        "backend": session.backend,
                        "brain_model": session.brain_model,
                        "stt_provider": session.stt_provider,
                        "tts_provider": session.tts_provider,
                        "transcript": text,
                        "speech_stopped": t_speech_stopped,
                        "stt_done": t_stt_done,
                        "retrieval_done": t_retrieval_done,
                        "llm_first_token": t_llm_first_token,
                        "tts_first_chunk": t_tts_first_chunk,
                        "playback_started": t_playback_started,
                        "primary_kpi_ms": primary_kpi_ms,
                    })

                    await websocket.send_json(
                        {
                            "type": "response.done",
                            "session_id": session_id,
                            "backend": response.get("backend"),
                            "used_knowledge": response.get("used_knowledge", []),
                            "citations": response.get("citations", []),
                            "timings": response.get("timings", {}),
                        }
                    )
```

With:

```python
                await _stream_voice_response(
                    websocket=websocket,
                    session=session,
                    session_id=session_id,
                    message=text,
                    tts_provider=tts_provider,
                    t_speech_stopped=t_speech_stopped,
                    t_stt_done=t_stt_done,
                    question_id=question_id,
                )
```

- [ ] **Step 2: Restart backend on server and test in browser**

Run on server: `cd /workspace/leasing && git pull && rag_demo_system/.venv/bin/supervisorctl -c rag_demo_system/scripts/supervisord.conf restart backend`

Test: open browser, select whisper + qwen3_tts + voice fast, ask a question. You should hear the first sentence play while the rest is still generating.

- [ ] **Step 3: Check timings**

Run on server: `bash rag_demo_system/scripts/voice_timings.sh`

Expected: STT ~500ms, first audio within 1.5-3.5s total (down from 10-50s).

- [ ] **Step 4: Commit**

```bash
git add rag_demo_system/backend/app.py
git commit -m "feat: wire sentence-level streaming into voice WebSocket handler"
```

---

### Task 4: Handle barge-in interrupt properly

**Files:**
- Modify: `rag_demo_system/backend/app.py` (the `response.cancel` event handler)

- [ ] **Step 1: Update the interrupt handler to drain the sentence queue**

The existing handler at line 1087-1090:

```python
            elif event_type == "response.cancel":
                session.assistant_speaking = False
                session.interrupted = True
                await websocket.send_json({"type": "response.cancelled", "session_id": session_id})
```

This already sets `session.interrupted = True`. The `llm_producer` and `tts_consumer` in `_stream_voice_response` check this flag and stop. No additional changes needed; the current interrupt handler works because `_stream_voice_response` reads `session.interrupted` in both loops.

- [ ] **Step 2: Test barge-in**

In the browser: start a voice question, while the assistant is speaking, press the talk button again (or start speaking). The response should stop immediately.

- [ ] **Step 3: Commit (if any changes were needed)**

```bash
git commit -m "test: verify barge-in works with streaming voice" --allow-empty
```
