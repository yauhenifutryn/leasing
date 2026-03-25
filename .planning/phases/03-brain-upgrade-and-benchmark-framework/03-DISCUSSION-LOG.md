# Phase 3: Brain Upgrade and Benchmark Framework - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-25
**Phase:** 03-brain-upgrade-and-benchmark-framework
**Areas discussed:** Question fixture design, Benchmark runner behavior, Comparison script output, Env profile structure

---

## Question Fixture Design

### Who writes the questions?

| Option | Description | Selected |
|--------|-------------|----------|
| You write them | User provides real questions, Claude formats. Best quality. | |
| Claude generates from KB | Claude reads KB files, generates per category. User reviews after. | ✓ |
| Mix of both | User provides 20-30, Claude fills rest from KB. | |

**User's choice:** Claude generates from KB
**Notes:** None

### File format

| Option | Description | Selected |
|--------|-------------|----------|
| JSONL | One JSON object per line. Same format as results output. | ✓ |
| YAML | Human-friendly, needs parser step. | |
| CSV | Simplest, limited structured fields. | |

**User's choice:** JSONL
**Notes:** None

### Quality scoring fields

| Option | Description | Selected |
|--------|-------------|----------|
| Expected keywords only | 2-5 key terms per question. Rough quality signal. | ✓ |
| Full expected answers | Complete reference answer. Deeper comparison but high effort. | |
| No quality fields | Questions only, latency focus. Manual quality eval later. | |
| You decide | Claude picks. | |

**User's choice:** Expected keywords only
**Notes:** None

---

## Benchmark Runner Behavior

### How the runner sends questions

| Option | Description | Selected |
|--------|-------------|----------|
| WebSocket | Full pipeline through real voice path. Most realistic. | ✓ |
| HTTP API shortcut | New /benchmark endpoint, skips STT. Faster but incomplete. | |
| You decide | Claude picks. | |

**User's choice:** WebSocket
**Notes:** None

### Warmup turns

| Option | Description | Selected |
|--------|-------------|----------|
| 3 turns | Industry standard. Enough for GPU caches. | ✓ |
| 5 turns | More conservative. | |
| You decide | Claude picks. | |

**User's choice:** 3 turns
**Notes:** None

### Error handling mid-run

| Option | Description | Selected |
|--------|-------------|----------|
| Log error and continue | Write error to JSONL, move to next question. | ✓ |
| Retry once then continue | Wait 5s, retry, then log and continue. | |
| Stop the run | Any failure stops entire benchmark. | |

**User's choice:** Log error and continue
**Notes:** None

---

## Comparison Script Output

### Output format

| Option | Description | Selected |
|--------|-------------|----------|
| Markdown table | Paste-ready table with metrics, deltas. | ✓ |
| Markdown + per-question CSV | Summary table plus detailed CSV. | |
| You decide | Claude picks. | |

**User's choice:** Markdown table
**Notes:** None

### Winner highlighting

| Option | Description | Selected |
|--------|-------------|----------|
| Highlight winners | Arrow/marker per metric row showing better stack. | ✓ |
| Neutral numbers only | Raw numbers, user interprets. | |
| You decide | Claude picks. | |

**User's choice:** Highlight winners
**Notes:** None

---

## Env Profile Structure

### Naming and organization

| Option | Description | Selected |
|--------|-------------|----------|
| Flat .env.bench.{name} | In rag_demo_system/. Follows .env.voice.{name} pattern. | ✓ |
| Subdirectory envs/ | Create envs/ folder. Cleaner but breaks existing pattern. | |
| You decide | Claude picks. | |

**User's choice:** Flat .env.bench.{name}
**Notes:** None

### Complete vs incremental

| Option | Description | Selected |
|--------|-------------|----------|
| Incremental overrides | Only variables that differ. Runner loads base .env then overlay. | ✓ |
| Complete standalone files | Full .env per profile. No inheritance. | |
| You decide | Claude picks. | |

**User's choice:** Incremental overrides
**Notes:** None

### How many profiles

| Option | Description | Selected |
|--------|-------------|----------|
| All 7 now | Create all including omni_hybrid placeholder. | ✓ |
| Phase 3 relevant only | 6 profiles, skip omni_hybrid. | |
| You decide | Claude picks. | |

**User's choice:** All 7 now
**Notes:** None

---

## Claude's Discretion

- Benchmark runner CLI argument design
- JSONL result schema field ordering
- Comparison script percentile computation internals
- How runner synthesizes question text into audio
- Profile variable names and values
- LLM first-token extraction implementation details

## Deferred Ideas

None -- discussion stayed within phase scope.
