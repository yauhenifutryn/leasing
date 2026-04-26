# Section 5c — Classifier model upgrade eval (optional sideline)

**Status**: pending (optional, not blocking close-out)
**Prereqs**: Section 2 complete (ClassifierOutput schema exists) + Section 3 complete (apply_turn transaction model in place)
**Estimated effort**: 3-5 hours (scratch branch, measure first, swap only if regression-free and non-trivially better)

## Goal

Evaluate whether to upgrade the SessionAgent classifier from `Qwen/Qwen3-4B-Instruct-2507-FP8` to a larger small model — candidates `Qwen/Qwen3-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct` — to close the three known 4B weak spots flagged in the post-MVP critique:

1. **Silent numeric clamping** (e.g. `term_months=5` → sanitized to 60) — instruction-tuned small models over-helpfully "fix" user input.
2. **Discourse-marker confusion** (e.g. "Давай" mid-sentence triggering `is_confirmation=true` when it's actually a filler).
3. **`change_value=0` ghost** — 4B emits `0` as "I don't know" instead of `null`.

**Must not regress latency or parallel-call capacity.** SessionAgent is on a dedicated vLLM instance (port 8788); bumping to 8B increases VRAM ~2× and may cost throughput.

## Why this is NOT automatically worth doing

Section 2 already closed most of the blast radius from these bugs at the schema level:

- `change_value=0 ghost`: `ClassifierOutput` schema's `Literal[0, 1]` on `condition_new` + guard-on-value-zero pattern (E-Codex-2 path in CP-2 acceptance) now denies ungrounded `0` at the boundary. The model can still emit it, but it no longer corrupts state.
- `"Давай" confirmation`: utterance-grounding layer in 2.3b (Option A, shipped) null-out is_confirmation when the utterance lacks a confirmation cue. Structural defense, not model-dependent.
- **Silent numeric clamping** is the one remaining genuine model-side weakness — Pydantic accepts the clamped value because `60` is in-range. Grounding regexes can catch "пять" → 5 explicitly, but the 4B is reportedly still one bad turn away.

So before swapping: **prove the 4B is still the bottleneck.** Run the eval *after* Section 3 ships and the orchestrator is clean — on a clean apply_turn, the residual failures are attributable to the classifier, not to gate-ordering chaos masking as model bugs.

## Required memories

- `reference_env_config.md` — dedicated SessionAgent vLLM command on port 8788
- `project_section_2_complete_2026_04_20.md` — baseline `structured-classifier-v1`
- `project_orchestrator_refactor_pending.md` — context on why schema + apply_turn were prioritized first
- `feedback_universal_fixes.md` — systemic, no special-case patches
- `feedback_benchmark_order.md` — measure before swapping

## Primary skill

`superpowers:systematic-debugging` — evidence-first. Do not swap on vibes.
`superpowers:verification-before-completion` — side-by-side analyzer deltas required.
`superpowers:using-git-worktrees` — scratch branch only, env-var rollback.

## Approach

### Step 1 — Build the residual-failure corpus (blocker gate)

After Section 3 ships, run `session_analyzer` on the last 30+ calls and **extract only the turns where the 4B classifier was the proximate cause of a failure** (not orchestrator bugs, not hygiene bugs). Target buckets:

- `silent_numeric_clamp` (N turns)
- `discourse_marker_false_confirmation` (N turns)
- `change_value_zero_ghost` (N turns)
- `ungrounded_enum_extraction` (N turns — though these should now all be schema-null'd)

If total residual-failure rate < 2% of turns, **abort eval**. The 4B is fine; gains are marginal; the 8B upgrade isn't worth the VRAM. Write memory `project_qwen4b_sufficient_<date>.md` and close section.

If ≥ 2%, proceed.

### Step 2 — Candidate shortlist

| Candidate | Size | Why consider | Why not |
|---|---|---|---|
| `Qwen/Qwen3-8B-Instruct-FP8` | 8B dense | Same model family, prompt transfer likely clean | +4GB VRAM vs 4B; no FP8 variant may exist — check HF |
| `Qwen/Qwen2.5-7B-Instruct` | 7B dense | Strong small-model baseline, AWQ available for <8GB | Different family; prompt may need tuning |
| `Qwen/Qwen3-4B-Thinking-2507` | 4B reasoning | Same VRAM, may fix clamping via CoT | Latency hit from thinking tokens; incompatible with `enable_thinking: False` discipline |
| Stay on `Qwen3-4B-Instruct-2507-FP8` | 4B MoE | Current, known-good on latency | Known weak on edge cases |

Do not evaluate > 8B — dedicated SessionAgent runs alongside brain on the same H100; VRAM budget is tight per `project_shared_infra_plan.md`.

### Step 3 — Scratch branch swap (single candidate at a time)

```bash
git worktree add /tmp/classifier-8b-eval feature/voice-pipeline
cd /tmp/classifier-8b-eval
git checkout -b scratch/classifier-8b-eval

# On H100 server:
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  Qwen/Qwen3-8B-Instruct-FP8 \
  --local-dir /workspace/models/Qwen3-8B-Instruct-FP8

# Swap SessionAgent model only — keep brain on 3.5 / 3.6 as-is
# rag_demo_system/.env:
#   SESSIONAGENT_MODEL=Qwen/Qwen3-8B-Instruct-FP8
#   SESSIONAGENT_GPU_UTIL=0.18   # up from 0.12; verify total stays < 1.0
```

VRAM sanity check before boot: brain (0.60) + new SessionAgent (~0.18) + Whisper (0.04) + embed/rerank (~0.07) ≈ 0.89. Leave 10% for KV cache growth.

### Step 4 — Checkpoints

| # | Check | How | Pass criteria |
|---|---|---|---|
| CP-5c.1 | SessionAgent boots on dedicated port 8788 | `curl :8788/v1/models` | 200 + correct model ID |
| CP-5c.2 | No brain VRAM starvation | `nvidia-smi` during smoke test + parallel-call test | Brain generation latency unchanged (±10%) |
| CP-5c.3 | ClassifierOutput schema validation | Replay residual-failure corpus (Step 1) through new classifier | Validation failure rate < 1% |
| CP-5c.4 | Silent clamp fixed | Targeted replay: "пять месяцев", "на пять лет аванс 30" etc. | `term_months=5` emitted unmodified (not 60) on ≥ 80% of clamp-prone inputs |
| CP-5c.5 | Discourse marker fixed | Targeted replay: "давай, рассчитай" vs "давай, хорошо" | `is_confirmation` differentiated correctly on ≥ 90% |
| CP-5c.6 | No new regressions | Full session_analyzer replay on last 30 calls, side-by-side diff | No degradation on any metric that was green on 4B |
| CP-5c.7 | Classifier latency | p50 / p95 on 100 turns | SessionAgent p95 ≤ 2× current 4B p95 (hard ceiling: no perceivable turn-taking delay) |
| CP-5c.8 | Concurrent-call capacity | Stress-test 5 parallel SIP calls | No queueing degradation vs 4B baseline |

### Step 5 — Decision

- **All CP green AND residual-failure rate drops by ≥ 50%**: swap prod, tag `classifier-8b-v1`, update `SESSIONAGENT_MODEL` in prod `.env`. Keep 4B weights on disk for fast rollback.
- **Any CP red** or gains marginal (< 50% residual reduction): stay on 4B, memoize findings, revisit after Section 6.

## Known risks

1. **Prompt transfer drift.** Current SessionAgent prompt is tuned against 4B-Instruct-2507. Both 8B and 7B may interpret the same prompt slightly differently. Budget time to retune if zero-shot transfer is poor.
2. **VRAM starvation.** If the new classifier pushes total GPU util past ~0.95, KV cache thrashing causes brain latency spikes under parallel load — fatal for voice SLA. Monitor during CP-5c.2 and CP-5c.8.
3. **Latency ceiling.** SessionAgent runs in parallel with RAG at the top of every turn (per `project_perf_parallel_classifier_rag.md`). If the new classifier is slower than RAG, the parallelism benefit evaporates. CP-5c.7 enforces this.
4. **No Qwen3-8B-Instruct-FP8 may exist.** Check HF availability first. Fallback: BF16 8B (uses ~16GB) — may not fit budget. If BF16 8B doesn't fit, try Qwen2.5-7B-Instruct-AWQ (~4GB, different quantization).
5. **Overkill for voice turns.** 8B classifier on <20-token utterances is arguably wasteful. If Section 3's apply_turn + Section 2's schema kill 90%+ of the failures, the 8B buys nothing.

## Rollback

```bash
# Edit rag_demo_system/.env back to:
#   SESSIONAGENT_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
#   SESSIONAGENT_GPU_UTIL=0.12
./rag_demo_system/scripts/regenerate_env_and_restart.sh

# Or at tag level:
git reset --hard structured-classifier-v1   # or current prod tag
```

No schema changes, no prompt-breaking changes, no client-facing exposure until swap is committed.

## Done criterion

Either:
- Swap committed, new tag `classifier-8b-v1` (or `classifier-qwen25-7b-v1`) pushed, prod on upgraded classifier — OR
- Decision memo `project_classifier_4b_sufficient_<date>.md` saved explaining why 4B stays.

## Not in scope

- Upgrading brain model (that's Section 5b)
- Multi-classifier ensemble (complexity > benefit for this use case)
- Fine-tuning the 4B (post-MVP consideration; not in current roadmap)

## Relationship to Section 5b

5b (brain swap 3.5 → 3.6) and 5c (classifier swap 4B → 8B/7B) are independent and can run in either order or in parallel on separate scratch branches. They touch different vLLM instances (8787 vs 8788). Avoid running both at once on the same H100 — too many variables to attribute regression to the right component.
