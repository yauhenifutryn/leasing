# Section 7 — KB Refinement: canonical-question → topical knowledge base

**Status**: pending. Authored 2026-04-29; revised 2026-05-03 to drop external prereqs and simplify validation.
**Prereqs**:
- **Phase A + B + C build**: NONE. Standalone branch (`feature/kb-refinement`) off current `feature/voice-pipeline` HEAD. Runs in parallel with ANALYSIS.md fix sweep.
- **Phase C production cutover** (env-var flip to `KB_LAYOUT=topical`): ANALYSIS.md scope stable on prod (don't ship two big changes in the same window).
- `eval_rag.py` (Section 6 CP-6.6a) is **NOT a prereq** — Phase C validation uses the existing `voice_lab.py` spot-check instead.

**Estimated effort**: ~2-3 days (A: half-day, B: half-day, C: 1-2 days).
**Base commit**: current `feature/voice-pipeline` HEAD at Section 7 kickoff.

## Goal

Replace the current canonical-question KB layout with a topical knowledge-base layout. Strip bot-operational metadata into a separate playbook. Build a synonym/canonical-query lookup index from real transcripts. Validate via spot-check on 20-30 real-transcript queries + 1-2 day live observation post-swap. Client review + KB-viz feedback loop is deferred to Section 7.1 (post-stabilization).

## Why now

Evidence from the current file (`kb_faq_ru.yaml` 350 entries / `kb_faq_ru_v2.md` 375 sections, 17.6k lines):

1. **Title-level duplication.** `выбор офиса в IVR` 3-4×; `связь с бухгалтерией / перевод в бухгалтерию / контакт с бухгалтерией` 3×; `досрочный выкуп / досрочное погашение / досрочное закрытие` 3×; `доверенность лизинг / после лизинга / после закрытия` 3×; `автолизинг при работе в РФ / за границей` 3×. Retriever's 0.75 dedup masks the symptom.
2. **Canonical questions are LLM-synthesized**, not real-user phrasings. Documented production failure: `"как с вами связаться в Бресте"` returns 404 while `"адрес офиса в Бресте"` works (master_plan.txt line 385-390). Embedding similarity is mediocre against real client utterances.
3. **14 metadata fields per entry, ~9 bot-operational** (`eligibility_rules`, `required_fields`, `compliance_notes`, `handoff_when`, `empathy_patterns`, `followups`, `references`). They pollute RAG context — the chunker doesn't strip them cleanly.
4. **Stale numbers across copies.** `kb-audit-report-2026-04-16.md` counted: 25 hits for `"10%"`, 14 for `"30%"` advance. Bug 14 in ANALYSIS.md (ПДН wrong definition) is the same disease.
5. **Hallucination class.** `2026-04-17-kb-grounding-design.md` documents Vadim director hallucination + Минск ул. Немига 24 hallucination — both partially driven by canonical-question retrieval bringing irrelevant entries into the LLM's context window.

## Risks (verified against codebase 2026-05-03)

Honest cost/benefit before approving the rewrite. Items below are **verified by grep**, not asserted from generic knowledge.

### Risks worth treating seriously

1. **More LLM hallucination surface inside topical sections.** Today each entry's `best_answer` is a tight rule statement. After: a section is several paragraphs with sub-headers — the LLM is more tempted to interpolate across them. Mitigation: keep explicit sub-headers (`Кому доступно`, `Финансовые параметры`, etc.); the existing grounding validator (`backend/utterance_grounding.py`) carries the weight.
2. **`bot_playbook.yaml` is NEW wiring, not migration.** Verified: nothing in `rag_demo_system/backend/` reads `eligibility_rules`, `empathy_patterns`, `handoff_when`, `compliance_notes`, `followups`, or `required_fields` today — they reach the LLM only via being chunked into retrieval text. After refactor we have to actively inject them via `turn_dispatcher.py` + the system-prompt builder. **If that wiring is incomplete, the bot loses guardrails it has today.** This is the single biggest engineering risk in Phase C.
3. **Synonym matching needs tuning.** Exact match misses real phrasings; fuzzy match routes wrong topic. Has to be empirically tuned on the transcript corpus.
4. **Editor cognitive load goes up.** Today: one new fact = add one YAML entry. After: decide topical section + edit `kb_topics_ru.md` + maybe add synonyms + maybe touch playbook. Three files instead of one. Real cost for long-term maintenance.
5. **`dedup_similarity_threshold=0.75` (prod, `rag_demo_system/config/app.yaml:36`) was masking source duplication.** After we remove source dupes, two distinct topical sub-chunks within one section may now over-dedup. Likely needs to come up to 0.82–0.85 post-swap. Add to Phase C tuning checklist.
6. **Section boundary mistakes.** "Автолизинг физлиц" vs "автолизинг ИП" — one section or two? Wrong boundary → wrong topic retrieved. Phase A.5 must derive boundaries from cluster output, not impose them, and explicitly flag uncertain calls.
7. **Existing tests will need fixture updates.** `test_chunking.py`, `test_dedup_chunks.py`, `test_retrieval_threshold.py` reference current chunk shapes.

### Verified-safe items (not actually risks)

1. **No code references KB `intent:` field by name.** Verified in `rag_demo_system/backend/`: `intent` in code is the classifier's intent (`TOOL`/`RAG`/`CONVERSATION`), not the KB's `intent:` metadata field. Restructuring KB intents is safe — zero code coupling.
2. **Total fact coverage.** Phase A.3 (coverage_check), the "no silent deletion" rule (Section "What NOT to do"), and the immutable archive at `knowledge_base/_archive/baseline_2026_05_03/` (committed `8830459` with SHA-256 manifest) together protect against losing facts.
3. **Rollback.** `KB_LAYOUT=legacy|topical` env var → ~5-min revert. File-level archive is the deeper net.

## Required memories (retrieve before starting)

- `project_kb_refinement_planned_2026_04_29.md` — this section's anchor memory
- `project_kb_viz_feature.md` + `project_kb_viz_dedup_plan.md` — KB-viz client feedback loop
- `project_kb_viz_session_2026_04_22.md` — current KB-viz state
- `project_rag_eval_deferred_2026_04_23.md` — RAG eval harness scope
- `project_qdrant_rebuild_bug.md` — `/api/index rebuild=true` does NOT drop collection; use REST DELETE directly
- `feedback_pin_all_versions.md` — pin every dependency version
- `feedback_universal_fixes.md` — no special-cases
- `feedback_no_postprocessing_hacks.md` — fix at source, not via regex post-processing

## Primary skills

- Phase A: `superpowers:systematic-debugging` (forensic data analysis), `superpowers:verification-before-completion`
- Phase B: `superpowers:test-driven-development` (every fix has a regression test)
- Phase C: `superpowers:writing-plans` for topical scaffolding, `superpowers:subagent-driven-development` for parallel cluster-by-cluster section authoring, `superpowers:requesting-code-review` before final ship

## Phase A — Diagnostic (no KB writes)

**Effort**: 3-4 hours. **Risk**: zero (read-only).

### A.1 Cluster all 350 entries by embedding similarity

`rag_demo_system/scripts/kb_cluster.py` (new):
- Embed each entry's `canonical_question + best_answer` using the production embedding model (same one Qdrant uses, for fidelity).
- Cluster by cosine similarity at threshold 0.85 (tunable).
- Emit `docs/superpowers/kb-clusters-<date>.md`: one section per cluster, listing contributing intents + 1-line content delta per entry. Highlight clusters with ≥3 members for surgical-pass priority.

### A.2 Stale-number scan

`rag_demo_system/scripts/kb_stale_scan.py` (new — spec already exists at `docs/superpowers/specs/2026-04-16-kb-audit-dedup-design.md`):
- Regex patterns: `(\d+)\s*мес` near `срок|максим`, `(\d+)\s*процент` near `минимальн|аванс`, `только \d+`, age limits.
- Cross-check against the calculator's rule matrix (Excel) and current `app.yaml` constraints.
- Emit `docs/superpowers/kb-stale-scan-<date>.md` with chunk IDs + suggested review.

### A.3 Coverage check on real-user queries

`rag_demo_system/scripts/kb_coverage_check.py` (new):
- Source corpus: `.state/kb_viz_queries.jsonl` (raw query log, on server) + session_analyzer transcripts + the 9 transcripts referenced in ANALYSIS.md.
- For each query, run BM25+embedding retrieval against current Qdrant index. Log top-K with similarity scores.
- Flag: `MISS` (top hit < 0.65), `PARTIAL` (0.65-0.80), `COVERED` (≥ 0.80).
- Emit `docs/superpowers/kb-coverage-<date>.md` with miss/partial lists. These are exactly the queries that the Phase C topical sections must explicitly cover via examples + synonym entries.

### A.4 Field-classification audit

For each of the 14 metadata fields, decide retrieval-facing vs bot-operational:
- **Retrieval-facing**: `intent`, `category`, `subtopic`, `keywords`, `tags`, `best_answer` (rules + numbers + facts). Goes into `kb_topics_ru.md`.
- **Bot-operational**: `eligibility_rules`, `required_fields`, `compliance_notes`, `handoff_when`, `empathy_patterns`, `followups`. Moves to `bot_playbook.yaml`, consumed by the dispatcher and the system prompt builder, NOT by the retriever.
- **Bridge**: `canonical_question` is replaced by `kb_synonyms.yaml` entries sourced from real transcripts.
- Document the split in the diagnostic report; surface for user approval before Phase B.

### A.5 Diagnostic synthesis

Combine A.1-A.4 into `docs/superpowers/kb-refinement-diagnostic-<date>.md`. User-facing summary: how bad the duplication is, where boundaries should land, target topical taxonomy, retrieval-payload shrinkage estimate.

**Phase A exit criterion**: user reviews diagnostic + approves Phase B scope.

## Phase B — Surgical pass

**Effort**: half-day. **Risk**: low (atomic commit, easy revert).

### B.1 Apply stale-number fixes

Direct edits to `kb_faq_ru.yaml` (source of truth) for the lines flagged in A.2. Re-render `kb_faq_ru_v2.md` via existing build pipeline (Makefile / `scripts/55_export_kb_markdown.py`). Re-index Qdrant: `python scripts/index_kb.py` on server.

### B.2 Strip bot-operational fields from retrieval-facing markdown

The chunker (`rag_demo_system/backend/ingest.py`) currently consumes the rendered markdown. Either:
- Update the markdown export script (`55_export_kb_markdown.py`) to omit operational-only sub-headings; or
- Update the chunker to skip those sub-headings when building chunks.

Whichever path is chosen, `bot_playbook.yaml` is NOT yet wired up — that's Phase C. Phase B just removes the operational-noise from retrieval.

### B.3 Drop the 3-4× duplicates

For each cluster ≥3 members from A.1: pick the best entry, redirect the rest. Implementation options:
- Mark redundant entries as `_archived: true` in YAML; export script skips them.
- Or physical removal with the deleted-entries archived to `kb_faq_ru.legacy_canonical.yaml`.

### B.4 Validation

- Spot-check retrieval recall on 10-20 known queries before/after via existing `voice_lab.py`.
- Run a smoke test against the existing simulator scenarios.
- Pin a regression test under `tests/test_kb_smoke.py` so future drifts get caught.

**Phase B exit criterion**: retrieval recall ≥ baseline on the 10-20 known queries, smoke test green, atomic commit pushed.

## Phase C — Full topical rewrite + production swap

**Effort**: 1-2 days. **Risk**: medium (large content change). Mitigated by env-var swap (`KB_LAYOUT=legacy|topical`) — instant revert.

**Validation philosophy:** current KB was never client-validated. The bar to swap is "not worse than current on accuracy + latency", not "client-approved". Client review is a separate later effort (Section 7.1).

### C.1 Duplicate canonical KB

```bash
cp knowledge_base/kb_faq_ru.yaml knowledge_base/kb_faq_ru.legacy_canonical.yaml
```

The repo already has a `kb_faq_ru.legacy.md` precedent so this naming convention exists. The legacy file is committed once, never edited again.

### C.2 Topical KB authoring

For each cluster from A.1, write a topical section in `knowledge_base/kb_topics_ru.md`:

```markdown
## section_id: auto-physlico-conditions
### parent_topic: автолизинг
### title: Условия автолизинга для физлиц

### Кому доступно
- возраст 21-75 лет (детали в matrix)
- гражданство РБ или ВНЖ
- (etc.)

### Финансовые параметры
- стоимость от X до Y BYN (per current Excel matrix; not hardcoded numbers)
- аванс 0-40% (calculator-driven, not KB-fixed)
- срок 12-84 месяца (calculator-driven)
- валюта BYN; USD/EUR с конверсией по НБРБ

### Примеры реальных формулировок клиента
- "хочу машину в лизинг"
- "посчитайте лизинг на тойоту"
- "беру в лизинг автомобиль за 100 тысяч"

### Когда передать специалисту
- (handoff conditions)

### last_verified: 2026-04-29
```

Use `superpowers:subagent-driven-development` to parallelize: dispatch a subagent per cluster, each writes one section, main session reviews and integrates.

### C.3 Synonym / canonical-query lookup

`knowledge_base/kb_synonyms.yaml`:

```yaml
- query: "как с вами связаться в Бресте"
  source: kb_viz_queries.jsonl
  matches:
    - section_id: offices-brest
      confidence: high
- query: "адрес офиса в Бресте"
  source: kb_viz_queries.jsonl
  matches:
    - section_id: offices-brest
      confidence: high
- query: "что такое нагрузка"
  source: client_test_2026_04_16
  matches:
    - section_id: glossary-nagruzka
```

Sourced from `.state/kb_viz_queries.jsonl` + session_analyzer transcripts. This replaces the synthetic `canonical_question` field. Wired into the retriever as a pre-search lookup: if the user query exact-matches or near-matches a synonym entry, jump directly to the section.

### C.4 Bot playbook YAML

`knowledge_base/bot_playbook.yaml`:

```yaml
- section_id: auto-physlico-conditions
  empathy_patterns: [...]
  followups: [...]
  handoff_when: [...]
  compliance_notes: [...]
  required_fields: [...]
```

Consumed by `turn_dispatcher.py` and `prompts/system_prompt_ru_v2.txt` builder. NOT indexed by Qdrant.

### C.5 Ingestion pipeline update

Modify `rag_demo_system/backend/ingest.py`:
- Read `kb_topics_ru.md` instead of `kb_faq_ru_v2.md` when `KB_LAYOUT=topical`.
- Chunk per section OR per sub-heading (decision in Phase A based on average section length).
- Embed examples-from-transcripts as part of the chunk text so embedding similarity catches real-user phrasings.

### C.6 Quick offline spot-check

Take ~20-30 representative queries from real transcripts (`.state/kb_viz_queries.jsonl` + session_analyzer dumps). Fire each through both KBs via existing `voice_lab.py` (or direct curl against `/api/retrieve`). Eyeball-compare top-K results.

Pass criterion: for each query, the new KB returns a chunk that is at least as relevant as the legacy KB. Borderline cases get logged to a `kb-spotcheck-<date>.md` for awareness — don't gate the swap on them.

**Effort**: 1-2 hours. No new tooling required.

### C.7 Production swap

Coordinate with ANALYSIS.md branch: do NOT swap if ANALYSIS.md is mid-deploy or fixes are still landing. Wait for a stable window.

Steps:
1. Merge `feature/kb-refinement` to `feature/voice-pipeline`. Default `KB_LAYOUT=legacy` so the merge itself is a no-op for production.
2. Pull on server. Set `KB_LAYOUT=topical` in `.env`. Run `python scripts/index_kb.py` to re-index the existing `kb` Qdrant collection in place (no parallel collection needed — revert is env-var flip + re-index of legacy YAML).
3. Restart backend.

### C.8 Live observation (1-2 days)

Monitor for regression:
- `session_analyzer` accuracy metrics (LLM-fallback rate, "обратитесь к специалисту" rate, ungrounded-replacement rate).
- Retrieval latency (`scripts/analyze_latency.sh` — RAG stage).
- Spot-listen to a few live calls.

If any regression appears:
- Flip `KB_LAYOUT=legacy` in `.env`, re-run `python scripts/index_kb.py`, restart. Production back on legacy KB in ~5 minutes.
- Investigate, patch, re-attempt swap. The topical KB stays in the repo; only the env var changes.

### C.9 Archive legacy KB

After 1-2 days stable on topical KB:
- Move `kb_faq_ru.yaml` and `kb_faq_ru_v2.md` to `knowledge_base/_archive/<date>/`.
- Drop the `KB_LAYOUT` env var (topical becomes the only path).
- Update README + PROJECT_LOG with the layout change.

**Phase C exit criterion**: 1-2 days stable on topical KB; no regression in session_analyzer metrics or retrieval latency; legacy archived.

## Section 7.1 — Client validation + KB-viz feedback loop (deferred)

**Status**: deferred. **Trigger**: Section 7 production swap stable for ~2 weeks AND (client requests deeper validation OR ramp-up of post-v1 KB-viz sweep per `project_kb_viz_dedup_plan.md`).

Two complementary client-facing artifacts:

1. **`KB-CHANGES.md` diff report** (~100-200 lines): per-section summary of what was consolidated, newly added, stale-numbers fixed, dropped as duplicate. Client reads this first, drills into topical KB sections of interest. Lower review burden than reading 5-8k lines of `kb_topics_ru.md` cold.
2. **KB-viz on the new collection**: client interacts with retrieval, sees what the bot would actually fetch, leaves ✗+comment feedback. Same loop as today's KB-viz, just pointed at the topical KB.

Resolves KB-viz branch open thread:
- KB-viz code lives on `feature/kb-viz` (HEAD `735eaaf`, 2026-04-22). Open main-branch merge thread per `project_kb_viz_session_2026_04_22.md`. Section 7.1 either (a) resolves the merge first (preferred) or (b) cherry-picks the KB-viz overlay onto a scratch branch.

**Effort**: ~2-3 days, depends on client availability for the feedback window.

## Files affected

### New
- `rag_demo_system/scripts/kb_cluster.py`
- `rag_demo_system/scripts/kb_stale_scan.py`
- `rag_demo_system/scripts/kb_coverage_check.py`
- `knowledge_base/kb_topics_ru.md`
- `knowledge_base/kb_synonyms.yaml`
- `knowledge_base/bot_playbook.yaml`
- `knowledge_base/kb_faq_ru.legacy_canonical.yaml` (Phase C.1 archive)
- `docs/superpowers/kb-refinement-diagnostic-<date>.md`
- `docs/superpowers/kb-clusters-<date>.md`
- `docs/superpowers/kb-stale-scan-<date>.md`
- `docs/superpowers/kb-coverage-<date>.md`
- `tests/test_kb_smoke.py` (regression pin from Phase B)

### Modified
- `rag_demo_system/backend/ingest.py` (chunker handles new layout, env-var-gated)
- `rag_demo_system/scripts/55_export_kb_markdown.py` (omit operational sub-headings)
- `rag_demo_system/scripts/eval_rag.py` (validation gate, may already exist from CP-6.6a)
- `rag_demo_system/backend/turn_dispatcher.py` (pull operational metadata from `bot_playbook.yaml`)
- `prompts/system_prompt_ru_v2.txt` builder (consume playbook entries)
- `.env` template + provisioning scripts (`KB_LAYOUT` env var)

### Reference (consult, don't modify)
- `docs/superpowers/specs/2026-04-16-kb-audit-dedup-design.md`
- `docs/superpowers/specs/2026-04-17-kb-grounding-design.md`
- `docs/plans/2026-01-29-kb-structured.md`
- `docs/kb-audit-report-2026-04-16.md`

## Checkpoints

- [ ] CP-7.0 — Phase A diagnostic report committed; `kb_cluster.py` + `kb_stale_scan.py` + `kb_coverage_check.py` shipped; field-classification audit user-approved.
- [ ] CP-7.1 — Phase B surgical commit pushed; retrieval recall ≥ baseline on 10-20 known queries; smoke test green.
- [ ] CP-7.2 — Phase C topical KB + synonyms YAML + bot_playbook YAML drafted; spot-check on 20-30 real-transcript queries shows new KB ≥ legacy.
- [ ] CP-7.3 — `feature/kb-refinement` merged to `feature/voice-pipeline` (default `KB_LAYOUT=legacy`, no-op for prod). ANALYSIS.md window stable.
- [ ] CP-7.4 — `KB_LAYOUT=topical` flipped on prod; re-index complete; backend restarted.
- [ ] CP-7.5 — 1-2 day live observation passed (no session_analyzer / latency regression); legacy KB archived; `KB_LAYOUT` env var dropped.
- [ ] CP-7.6 — `project_kb_refinement_complete_<date>.md` memory written; `project_kb_refinement_planned_2026_04_29.md` archived.

## Rollback

- **Phase A**: read-only; revert is `git revert` of report files.
- **Phase B**: single atomic commit; revert + `python scripts/index_kb.py` on server.
- **Phase C**: env-var flag `KB_LAYOUT=legacy|topical`. Worst case: flip to `legacy`, re-index legacy YAML, restart. Production back on legacy KB in ~5 minutes. No git revert needed. Topical KB stays in repo for next attempt.

## Expected file-size impact

| Layer | Current | After C.9 |
|---|---|---|
| Retrieval-facing (`kb_faq_ru_v2.md` → `kb_topics_ru.md`) | 17.6k lines | ~5k-8k lines |
| Bot-operational metadata (bundled → `bot_playbook.yaml`) | ~50% of yaml weight | 2k-4k separate lines |
| Synonym lookup (canonical_question → `kb_synonyms.yaml`) | bundled | 0.5k-1.5k lines |
| Total content | ~36k yaml+md | ~8k-13k |

Retrieval payload shrinks ~50-70%. RAG context size per turn drops correspondingly. Section 6 prompt-trim gains stack on top.

## What NOT to do in this section

- No new factual content introduced without evidence from real client interactions or client-confirmed source documents.
- No taxonomy imposed from scratch — derive it from cluster output in A.1.
- No silent deletion of any entry whose facts are not yet covered by a topical section. Phase B drops only confirmed duplicates.
- No production cutover during an active ANALYSIS.md deploy window — wait for stability.
- No modification of the canonical-question YAML once `kb_faq_ru.legacy_canonical.yaml` is created — it's a frozen reference.
- No client-facing review (file or KB-viz) inside Section 7 — deferred to Section 7.1.

## Done criterion

All checkpoints green. Production traffic on topical KB for 1-2 days stable. Session_analyzer shows no regression in retrieval-related metrics. Legacy KB archived. `MEMORY.md` updated. Section 7.1 (client validation) tracked separately as a deferred follow-up.
