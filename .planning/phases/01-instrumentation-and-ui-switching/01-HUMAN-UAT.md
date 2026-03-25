---
status: partial
phase: 01-instrumentation-and-ui-switching
source: [01-VERIFICATION.md]
started: 2026-03-25T00:00:00Z
updated: 2026-03-25T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. Visual selector placement
expected: Three labeled dropdown rows appear below the Voice Provider selector; Brain model shows "Qwen3-30B-A3B" and "Qwen3.5-35B-A3B"; STT shows sensevoice, whisper, vosk, yandex_speechkit; TTS shows cosyvoice, vosk_tts, yandex_speechkit.
result: [pending]

### 2. End-to-end session.update round-trip
expected: Log line's stack_id contains the newly selected model name; primary_kpi_ms is a positive millisecond value.
result: [pending]

### 3. localStorage persistence across reload
expected: After reload, Brain model, STT, and TTS selectors show the values that were saved, not the hardcoded defaults.
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
