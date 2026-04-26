# Section 5b — Qwen3.6-35B-A3B-FP8 evaluation (optional sideline)

**Status**: pending (optional, not blocking close-out)
**Prereqs**: none strictly; safest to run after Section 3 ships so the refactored apply_turn is the baseline under test
**Estimated effort**: 2-4 hours (scratch branch, no prod change unless eval passes)

## Goal

Evaluate `Qwen/Qwen3.6-35B-A3B-FP8` as a drop-in replacement for the current production brain `Qwen/Qwen3.5-35B-A3B-FP8`. Qwen3.6 was released 2026-04-22 with identical architecture (`model_type: qwen3_5_moe`, same MoE topology E=256 N=512, same 40 layers / 2048 hidden / 16-head gated attention). This means the swap is infrastructurally one-line, but production behavior (tool calling, Russian readback, RAG+tool interaction) must be re-validated before any swap.

**Must not regress Russian leasing voice quality.** If any eval dimension regresses vs `structured-classifier-v1` baseline on the Section 2 call log, stay on 3.5.

## Why we're doing this

- 3.6 headline benchmarks: SWE-Bench Verified 70.0 → 73.4, SWE-Bench Multilingual 60.3 → 67.2, Terminal-Bench 2.0 40.5 → 51.5. Most gains are agentic-coding — **not our use case**.
- 3.6 knowledge/reasoning benchmarks: MMLU-Pro flat (85.3 → 85.2), MMLU-Redux flat (93.3 → 93.3), GPQA 84.2 → 86.0. Marginal for voice.
- TAU3-Bench and HLE slightly regressed (68.9 → 67.2, 22.4 → 21.4). Agent task regression is a yellow flag for our orchestrator.
- **No Russian benchmarks published by Qwen for 3.6.** The 3.5 card had MMMLU, MMLU-ProX, NOVA-63, WMT24++; all removed from 3.6's card. Russian quality delta is unknown.
- Upside: potentially cleaner tool-call behavior, new `preserve_thinking` option (not useful for us — we run `enable_thinking: False`).

## Required memories

- `reference_env_config.md` — current prod vLLM command
- `project_section_2_complete_2026_04_20.md` — baseline tag `structured-classifier-v1` + CP-2.5 SIP call cc7fc318 for comparison
- `project_voice_stack_direction.md` — brain decision history (why 3.5-35B-A3B-FP8 was picked over 3-30B-A3B)
- `feedback_production_quality.md` — no temporary fixes; production-ready only
- `feedback_benchmark_order.md` — benchmark-driven decisions

## Primary skill

`superpowers:systematic-debugging` — this is an empirical regression test. Evidence before assertions.
`superpowers:verification-before-completion` — do not claim "3.6 is better" without side-by-side analyzer output.
`superpowers:using-git-worktrees` — isolate from `feature/voice-pipeline` during the experiment.

## Approach

Run on a scratch branch off `feature/voice-pipeline` (or whatever is prod at the time). Do **not** touch prod env until eval passes.

### Step 1 — Pull and boot

```bash
git worktree add /tmp/qwen36-eval feature/voice-pipeline
cd /tmp/qwen36-eval
git checkout -b scratch/qwen36-eval

# On the H100 server:
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  Qwen/Qwen3.6-35B-A3B-FP8 \
  --local-dir /workspace/models/Qwen3.6-35B-A3B-FP8

# Swap model name only — keep all other vLLM flags as-is
# rag_demo_system/.env:
#   RAG_LLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8
#   RAG_LLM_FAST_MODEL=Qwen/Qwen3.6-35B-A3B-FP8

./rag_demo_system/scripts/regenerate_env_and_restart.sh
```

### Step 2 — Smoke tests (blocker gate)

```bash
./rag_demo_system/scripts/smoke_test.sh
```

Must pass. If any smoke test fails, abort — do not proceed to analyzer eval.

### Step 3 — Checkpoints to validate

| # | Check | How | Pass criteria |
|---|---|---|---|
| CP-5b.1 | Model boots on vLLM 0.19.0 | `curl :8787/v1/models` | 200 with correct model ID |
| CP-5b.2 | Basic Russian response | Smoke test Russian prompt | No language mixing, coherent output |
| CP-5b.3 | Tool-call parser works | Smoke test calculator call (line 326 of smoke_test.sh) | Tool call parsed, args extracted |
| CP-5b.4 | RAG+tool-call workaround still needed? | Send prompt with both KB context AND tool definition | If tool still suppressed: workaround at `backend/app.py:997` stays. If tool fires: workaround can be removed |
| CP-5b.5 | Readback compliance | Run session_analyzer on last 10 calls replayed through 3.6 | Readback rate ≥ 3.5 baseline |
| CP-5b.6 | Classifier accuracy (unchanged) | SessionAgent stays on Qwen3-4B-Instruct-2507-FP8; verify no cross-model regression | Classifier passes its own test suite |
| CP-5b.7 | Grounding pass rate | Compare kb_gap_report output | No regression on grounded answers |
| CP-5b.8 | Latency profile | p50 / p95 TTFT + tokens/sec on 20 calls | ≤ 10% regression tolerable; >10% = abort |

### Step 4 — Decision

- **All CP green, no regression**: open PR to swap prod. Keep `structured-classifier-v1` tag as rollback anchor. Tag new prod as `qwen3.6-eval-passed-<date>`.
- **Any CP red**: close scratch branch, stay on 3.5, write a follow-up memory noting what failed. Revisit in 2-4 weeks when 3.6.x community fixes land.

## Known risks

1. **Tool-call parser drift.** Prod uses `--tool-call-parser qwen3_xml`. The 3.6 HF card recommends `qwen3_coder`. If the parser produces different XML tags, the tool-call extraction in `backend/app.py` silently breaks. First thing to check.
2. **RAG-vs-tool workaround.** `backend/app.py:997` comments "Qwen3.5 suppresses tool calling when ANY reference/KB text is present." 3.6 may behave differently — better or worse. Explicit test at CP-5b.4.
3. **MoE kernel tuning.** `tune_vllm_kernels.sh:88` was tuned for 3.5's fp8_w8a8 weight distribution. Topology is identical so the config file shape works, but optimal kernel config might shift. Re-run the tuner (5-10 min) as part of boot.
4. **Russian regression undetected.** Benchmarks don't exist for 3.6. Only the session_analyzer replay at CP-5b.5 can catch this. **Do not skip CP-5b.5.**
5. **No Qwen3.6-4B exists** at time of writing. SessionAgent classifier stays on `Qwen3-4B-Instruct-2507-FP8`. Mixed-version stack is fine but one more thing to verify.

## Rollback

Trivially safe:

```bash
# Edit rag_demo_system/.env back to:
#   RAG_LLM_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
#   RAG_LLM_FAST_MODEL=Qwen/Qwen3.5-35B-A3B-FP8
./rag_demo_system/scripts/regenerate_env_and_restart.sh

# Or at tag level:
git reset --hard structured-classifier-v1   # or whatever prod tag is current
```

Nothing else touched — no schema changes, no prompt changes, no client-facing changes until swap is committed.

## Done criterion

Either:
- Swap committed, new tag pushed, scratch branch deleted, prod on 3.6 — OR
- Scratch branch deleted, decision memo saved as memory `project_qwen36_eval_<date>.md` describing what regressed and when to retry.

## Not in scope for this section

- Swapping SessionAgent 4B model — no Qwen3.6-4B exists yet
- Trying non-FP8 variants (GPTQ-Int4, BF16) — FP8 is the right quantization for H100
- Evaluating Qwen3.6-27B dense — 35B-A3B MoE is already faster and smarter on H100 (see memory `project_voice_stack_direction.md`)
