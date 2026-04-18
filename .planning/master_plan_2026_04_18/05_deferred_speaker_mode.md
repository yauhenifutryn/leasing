# Section 5 — Speaker mode adaptive VAD (DEFERRED)

**Status**: deferred
**Prereqs**: Section 4 complete (shares VAD tuning machinery)
**Estimated effort**: ~1 day

## Why deferred

Client complaint is about "system kicks in too fast" (Sections 1.4 + 4), not about speakerphone calls specifically. Speaker mode adaptive VAD is a separate enhancement for calls made with speaker on loudspeaker (more ambient noise). Useful but not blocking.

## Scope summary (when picked up)

- Adaptive RMS threshold based on measured ambient noise in first 2 seconds of the call
- Separate VAD tuning profile for "speakerphone" vs "handset" (inferred from audio characteristics)

## Required memories (when picked up)

- `project_handoff_2026_04_18.md` — deferred task 7 (original note)
- Section 4 memories

## Do not start without explicit user instruction.
