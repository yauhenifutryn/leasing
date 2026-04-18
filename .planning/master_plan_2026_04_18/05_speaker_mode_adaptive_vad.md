# Section 5 — Speaker mode adaptive VAD

**Status**: pending (experimental; isolated session recommended)
**Prereqs**: Section 4 complete (shares VAD tuning path); ideally run on a branch off `turn-taking-v1`
**Estimated effort**: ~1 day including live A/B

## Goal

Detect and adapt to speakerphone calls. Users on speaker have lower SNR and different RMS profile; the current static RMS floor misses their speech or catches their echo. Measure ambient RMS per session and scale the barge-in threshold accordingly. Default handset behavior must not regress.

**Must not break handset mode.** Prior attempts to tune global thresholds regressed handset — that's the forbidden failure mode. This section only adds per-session scaling.

## Required memories

- `project_handoff_2026_04_18.md` — original Fix 39 task note + prior attempts context
- `project_barge_in_plan.md` — original barge-in design
- `project_barge_in_remaining.md` — where handset tuning landed (RMS floor 300, 500ms pre-roll, 0.5s warmup)
- `project_mvp_complete_2026_04_18.md` — baseline commit
- `feedback_production_quality.md` — no temporary fixes
- `feedback_universal_fixes.md` — systemic, not special-case

## Primary skill

`superpowers:systematic-debugging` — this is an empirical tuning task with high regression risk; follow evidence-gathering protocol. `superpowers:test-driven-development` for config and threshold tests. `superpowers:verification-before-completion` before declaring A/B won. `superpowers:using-git-worktrees` to isolate from main feature branch during experiment.

## Chosen approach (locked in prior discussion)

- Measure ambient RMS in the first 2 seconds of each call, after consent TTS and before user speaks.
- If ambient RMS exceeds a threshold `N` (to tune live), scale the barge-in RMS floor for this session only.
- Default handset mode (low ambient RMS) path unchanged.
- Add ONE new config knob: `SPEAKER_RMS_HEURISTIC_ENABLED` in `backend/settings.py` (default `true`). Disable to hard-revert without deploy.

## Scope

### 5.1 Ambient RMS measurement

Implementation point: `backend/app.py` jambonz audio loop around line 3503 (where `_frame_rms` is computed). Capture the first ~100 frames (~2 sec at 16kHz with 320-sample frames) after consent, average, store on `session` as `session.ambient_rms`.

Edge cases:
- User starts speaking immediately → sample window gets user speech. Mitigate by limiting measurement to frames where SileroVAD says `speech=False`.
- Call drops mid-measurement → fall back to default floor.
- Some Jambonz transport quirk with silence bytes → confirm amplitude scale matches existing `_frame_rms` calc.

### 5.2 Per-session threshold derivation

```python
# backend/audio_utils.py or similar
def derive_rms_floor(ambient_rms: float, default_floor: int = 300) -> int:
    if ambient_rms < 50:   # handset / quiet
        return default_floor
    if ambient_rms < 150:  # borderline speaker
        return int(default_floor * 1.5)
    return int(default_floor * 2.5)  # loud speaker
```

Exact multipliers are placeholders — **tune live** (see Phase 5.A).

Log the computed floor on session start: `[VAD] session=<id> ambient_rms=<float> rms_floor=<int>`. Critical for diagnosing failures.

### 5.3 Config knob

`backend/settings.py`:
```python
SPEAKER_RMS_HEURISTIC_ENABLED: bool = Field(
    default=True,
    description="Per-session RMS floor adaptation. Set false to revert to static RMS_FLOOR.",
)
```

If false, always return `default_floor`.

### 5.4 Tests

- Unit: `derive_rms_floor(10)` returns 300; `derive_rms_floor(100)` returns 450; `derive_rms_floor(300)` returns 750.
- Unit: `SPEAKER_RMS_HEURISTIC_ENABLED=False` forces 300 regardless.
- Integration (mocked audio frames): session-level flow measures ambient and sets floor correctly.

## Live A/B protocol (MUST RUN BOTH)

### Test A — Handset regression guard

Replay the 8-turn scenario from session e46dae0b (MVP baseline), speakerphone off. Expected: ≥ 10 BARGE-IN markers across the call. NO ignored user speech.

**Fail condition**: any user utterance not barge-ing in when bot is speaking. If fail → revert immediately, investigate why ambient RMS misclassified the handset.

### Test B — Speaker mode reliability

Start call on handset, switch to speaker after consent. Give the same 8-turn scenario. Expected: barge-in still fires on every user attempt. NO echo triggering false barge-in.

**Fail condition**: bot's TTS triggers barge-in on itself (echo), OR user speech not detected. Debug via the `[VAD] session=<id> ambient_rms=<float> rms_floor=<int>` log line — correlate floor with observed behavior.

## Checkpoints

- [ ] CP-5.0 — Chosen multipliers validated on one pilot call (each of handset + speaker). Log confirms correct ambient_rms classification.
- [ ] CP-5.1 — `derive_rms_floor` + config knob implemented with unit tests.
- [ ] CP-5.2 — Ambient RMS measurement wired into jambonz audio loop; log line emitted once per session.
- [ ] CP-5.3 — Test A (handset) passes with ≥10 BARGE-IN markers, zero ignored speech.
- [ ] CP-5.4 — Test B (speaker) passes with reliable barge-in and no echo false positives.
- [ ] CP-5.5 — Section closed — commit tagged `speaker-mode-v1`.

## Rollback

Single-env-var rollback:
```bash
# On server:
export SPEAKER_RMS_HEURISTIC_ENABLED=false
.venv/bin/supervisorctl -c scripts/supervisord.conf restart backend
```

Full code revert:
```bash
git revert <commits-in-this-section> --no-edit
git push
```

## Constraints (do NOT touch)

- Fix 36 unified `_speak_tts` — leave alone
- Orchestrator state gates — leave alone
- VAD silence threshold from Section 1.4 and adaptive logic from Section 4 — separate concern
- Default RMS floor constant (300) — do not change the constant; only SCALE via the new function

## Handoff to Section 6

Section 5 completing adds the `SPEAKER_RMS_HEURISTIC_ENABLED` knob to document in README + PROJECT_LOG entry. Section 6's verification gate should include "speaker-mode call test passes" as a row if Section 5 shipped.
