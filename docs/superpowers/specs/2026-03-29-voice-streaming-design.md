# Voice Streaming Design: Sentence-Level LLM -> TTS Pipeline

**Date:** 2026-03-29
**Status:** Approved
**Goal:** Reduce voice response latency from 10-50s to 1.5-3.5s by streaming sentences to TTS as they're generated.

## Problem

The current voice pipeline is fully sequential:
1. Wait for ALL LLM tokens (3-8s)
2. Send entire text to TTS (5-38s depending on length)
3. Send entire audio to browser

Total time-to-first-audio: 10-50s. Unacceptable for a customer service voice assistant.

## Solution

Sentence-level streaming pipeline with 1-ahead TTS prefetch:

```
LLM tokens -> Sentence Detector -> TTS Queue (1-ahead) -> Browser audio chunks
```

Three concurrent stages run in parallel:
- **LLM** keeps generating tokens
- **Sentence Detector** emits complete sentences
- **TTS Consumer** processes sentences and streams audio to browser

## Architecture

### Sentence Detector

Accumulates LLM tokens into a buffer. Emits a sentence when the buffer ends with sentence-terminal punctuation followed by a space or end-of-stream.

**Emit triggers:** `. ! ? ...` followed by space or end-of-stream.

**False positive protection:** Skip splits after common Russian abbreviations: `т.е.`, `т.д.`, `т.п.`, `г.`, `руб.`, `USD.`, `EUR.`, `BYN.`, `др.`, `пр.`, `стр.`, `ул.`

**Flush on stream end:** When LLM stream finishes, emit whatever remains in the buffer as the final sentence.

### TTS Queue

An `asyncio.Queue` connects the sentence detector to the TTS consumer.

**Producer (LLM loop):** Detects sentences, puts them on the queue. Puts a sentinel `None` when LLM stream ends.

**Consumer (TTS loop):** Takes sentences from the queue, calls `synthesize_audio_with_provider` for each, sends `response.output_audio.delta` to the WebSocket immediately. Starts the next TTS call as soon as the previous one returns (1-ahead: while browser plays sentence N, TTS processes sentence N+1).

**Text display:** Each sentence is also sent as `response.output_text.delta` immediately when detected, before TTS.

### Interrupt (Barge-in)

When the user starts talking while the assistant is responding:
1. Set `session.interrupted = True`
2. The LLM loop checks this flag and stops generating
3. The TTS consumer checks this flag and stops processing
4. Drain the queue
5. Send `interrupt` event to browser
6. Browser stops audio playback

### Error Handling

- TTS fails on one sentence: skip audio, send text-only delta, continue with next sentence
- LLM stream errors mid-generation: flush accumulated buffer as final sentence
- WebSocket disconnect: cancel all tasks immediately

## Files to Modify

### `backend/app.py`

**Remove:** `_voice_chat_streaming_sync` (synchronous, blocks until all tokens collected)

**Add:** `_stream_voice_response` async function that:
1. Runs RAG retrieval
2. Opens LLM stream
3. Runs sentence detection loop (producer)
4. Runs TTS consumer loop (concurrent asyncio task)
5. Handles interrupt flag

**Modify:** `voice_ws` handler to call `_stream_voice_response` instead of `asyncio.to_thread(_voice_chat_streaming_sync)`.

### `backend/sentence_detector.py` (new file)

Simple class with `feed(token: str) -> list[str]` method.
- Accumulates tokens in buffer
- Returns list of complete sentences (usually 0 or 1)
- `flush() -> str | None` returns remaining buffer

### `backend/voice_adapters.py`

No changes. `synthesize_audio_with_provider` already handles single-sentence calls.

### `frontend/app.js`

No changes. `playPcm` with `nextPlayTime` scheduling already handles multiple `response.output_audio.delta` events with gapless playback.

## Expected Latency

| Stage | Current | With streaming |
|-------|---------|---------------|
| STT | 500ms | 500ms |
| RAG | 300ms | 300ms |
| LLM to first sentence | 3-8s (all tokens) | 0.5-1.5s |
| TTS for first sentence | 5-38s (full text) | 0.3-1s |
| **Time to first audio** | **10-50s** | **1.5-3.5s** |

## Implementation Sequence

1. Create `sentence_detector.py` with unit tests
2. Refactor `voice_ws` handler: replace sync blocking call with async streaming
3. Add TTS queue consumer as concurrent asyncio task
4. Add interrupt handling to both producer and consumer
5. Test end-to-end in browser
6. Measure latency improvement with `voice_timings.sh`
