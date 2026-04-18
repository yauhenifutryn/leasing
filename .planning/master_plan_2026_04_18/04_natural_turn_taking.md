# Section 4 — Natural turn-taking (ElevenLabs-style)

**Status**: pending
**Prereqs**: Section 3 complete (cleaner state machine for hooking into)
**Estimated effort**: ~2 days
**Base commit**: end of Section 3

## Goal

Production-grade conversational turn-taking. Goes beyond the Section 1 VAD bump to implement adaptive endpointing, filler tolerance, and prosody-aware bot-speaking delay.

This section is explicitly **iterate-and-revert-tolerant** — ship incrementally in its own branch, A/B on the server, roll back quickly if perception worsens.

## Required memories

- `project_barge_in_plan.md` — original barge-in design
- `project_barge_in_remaining.md` — where the current (500ms) tuning landed and why
- `project_mvp_complete_2026_04_18.md` — baseline; Section 1 Fix 1.4 may have already bumped to 900ms
- `feedback_production_quality.md` — no temporary fixes
- `feedback_universal_fixes.md` — systemic, not special-case

## Primary skill

`superpowers:brainstorming` — this is exploratory/design-heavy. ElevenLabs-style turn-taking has many tradeoffs to discuss before committing. Then `superpowers:writing-plans` for the chosen approach.

Research skills: use `WebFetch` on ElevenLabs Conversational AI docs + papers on end-of-turn detection. Use `Agent` with `general-purpose` for a research sub-task.

## What "natural turn-taking" means here

Four behaviors to layer:

### 4.1 Adaptive silence threshold

Base threshold bumped to 900ms in Section 1. Now make it adaptive:
- After a completed utterance (profile capture succeeded → COLLECTING, state moved): 700ms
- After a barge-in while bot was speaking: 1200-1500ms (user formulating thought)
- After an incomplete utterance fragment ("Я хочу..."): 1500-1800ms
- During a readback/confirm response window: 800ms (user expected to be brief)

Implementation: expose a per-turn `expected_silence_ms` from the state machine (extended from Section 3's apply_turn), passed to VAD as a hint.

### 4.2 Filler tolerance

"Эм", "ну", "так", "короче" at start of utterance should NOT reset the silence timer. They're thinking-sounds, not content.

Implementation options:
- Whisper partial transcripts: if the first decoded word is in a filler set, don't count the gap as silence start
- OR: a small server-side "filler detector" that checks the first 300ms of transcribed audio and suppresses if it's all fillers

### 4.3 Prosody-aware endpoint detection

ElevenLabs does this with a light model. Open-source options:
- SileroVAD's "speech_end" confidence + intonation heuristic
- A small trained model on Russian prosody (probably overkill for MVP)

Start simpler: if the last transcribed word has a trailing rising pitch (question) or a strong fall (statement end), adjust silence threshold. Needs a lightweight pitch-tracker on the last 500ms of audio. Defer if research shows it's hard.

### 4.4 Bot-side pre-speech delay

After user stops, bot waits ~300-500ms before starting TTS (feels natural, not robotic). Currently, bot starts as fast as possible (latency-optimized). Add a small "humanizing" delay option.

## Research phase (do this first, before writing code)

1. Read ElevenLabs Conversational AI public docs on turn-taking.
2. Look at Silero VAD's advanced API — does it expose speech-end confidence?
3. Evaluate pitch-tracker libs: `librosa`, `parselmouth` (Praat wrapper). Latency and dep weight?
4. Benchmark current perceived-latency baseline: record 5 calls with Section 1 bump (900ms), user rates naturalness 1-10. Use as baseline.

Write findings to this file in a new **## Research findings** section before implementing.

## Scope (phased execution)

### Phase 4.A — Adaptive silence (state machine plug-in)

Smallest win, least risk. Requires Section 3's apply_turn cleanup (state determines expected silence).

### Phase 4.B — Filler tolerance

Medium risk. Adds a transcript pre-check. Revertible.

### Phase 4.C — Prosody endpoint

High risk. Implement only if research shows a lightweight path. Otherwise defer to a future section.

### Phase 4.D — Pre-speech delay

Trivial. Add a 300-500ms sleep before the first TTS emission in responses. A/B on server.

## Checkpoints

- CP-4.0: Research findings documented in this file.
- CP-4.1: Baseline perceived-naturalness score captured (5 calls, user rates 1-10).
- CP-4.2: Phase 4.A — adaptive silence per state. Live A/B, naturalness score improves by +1.
- CP-4.3: Phase 4.B — filler tolerance. Naturalness +0.5.
- CP-4.4: Phase 4.C — prosody endpoint (if feasible). Naturalness +1.
- CP-4.5: Phase 4.D — pre-speech delay. Naturalness +0.5.
- CP-4.6: Section closed — commit tagged `turn-taking-v1`.

## Rollback

Each phase is independently revertible via its own commit. If naturalness worsens after a phase, revert that phase only:

```bash
git revert <phase-commit-sha>
git push
```

## Handoff

After Section 4, master plan is effectively done. Sections 5 and 6 are deferred and only touched on explicit user request.
