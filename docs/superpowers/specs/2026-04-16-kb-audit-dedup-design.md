# Spec 6: KB Audit and Deduplication

**Cluster:** F — Knowledge base
**Depends on:** user deliverable (list of client questions from transcript)
**Blocks:** —
**Status:** deferred until user provides transcript questions list

## Context

Two concerns:

1. **Coverage gaps.** User will provide a list of questions the client asked
   during recent calls. Each needs to be mapped to an existing KB entry or
   flagged as missing.
2. **Stale tool-related entries.** KB (`knowledge_base/kb_faq_ru_v2.md`)
   contains language from the pre-tool era (e.g., "срок до 36 месяцев" as a
   fixed fact). After Specs 1-5 land, term can go up to 84. The bot reads KB
   context during RAG calls and may regurgitate stale numbers even when
   calculator says otherwise.

## Problem

1. No audit list of what the KB covers vs. what real clients ask.
2. Numeric values in KB are duplicated across many chunks; if one needs
   updating, all copies drift.
3. The RAG retriever (dedup threshold 0.75) doesn't guard against stale vs.
   fresh; it just picks top-N similarity. Stale content can win.

## Goals (draft — to refine after user input)

- Produce an audit report: for each client question from user's list, map to
  nearest KB chunks (top 3) with similarity scores; flag gaps (no chunk ≥ 0.7).
- Scan KB for tool-conflicting entries: any line mentioning specific term
  limits, prepaid percentages, or calculator-related constraints. Flag for
  review.
- Scan for duplication: entries where near-identical text appears in 3+
  chunks; propose consolidation.
- Propose diff patches to `kb_faq_ru_v2.md` for approval.
- Re-index post-patch.

## Non-goals

- Auto-apply KB changes without human review.
- Add new content without evidence from real client interactions.
- Rewrite KB from scratch.

## Design (preliminary)

### Pipeline

```
user questions list → embed each → search KB → emit JSON report
                                              → add/modify/remove recommendations

KB full scan → flag tool-conflict entries → emit markdown review doc
```

### Script outline

`rag_demo_system/scripts/kb_audit.py` (new):

1. Read user questions list (`/path/to/questions.md` or inline list).
2. For each question, query the existing BM25 + embedding retriever.
3. If top hit score < 0.65, flag as GAP.
4. If top hit score 0.65-0.80, flag as PARTIAL (may need enrichment).
5. If ≥ 0.80, mark COVERED.
6. Emit `docs/superpowers/kb-audit-report-<date>.md`.

### Stale content scan

`rag_demo_system/scripts/kb_stale_scan.py` (new):

Regex patterns flagging potentially stale content:
- `(\d+)\s*мес` followed by context about max term.
- `аванс.*(\d+)\s*процент` followed by minimum language.
- Mentions of "только" + quantity ("только 36 месяцев").
- Any hardcoded number conflicting with the Excel rule matrix.

Output: `docs/superpowers/kb-stale-scan-<date>.md` with chunk IDs + suggested
review.

## Files to change

To be determined after user provides questions list. Initial expectation:

- `knowledge_base/kb_faq_ru_v2.md` (diff patches, human-reviewed)
- `rag_demo_system/scripts/kb_audit.py` (new)
- `rag_demo_system/scripts/kb_stale_scan.py` (new)
- Re-run existing `rag_demo_system/scripts/index_kb.py` after patches

## Testing

**Unit**
- KB audit script correctness on a synthetic KB with known gaps.
- Stale scan script regex coverage on synthetic examples.

**Integration**
- Full re-index; spot-check retrieval for 5 post-patch questions.

## Risks

| Risk | Mitigation |
|---|---|
| Over-aggressive removal deletes useful context | Every change is a proposed diff; no auto-apply |
| User's question list is informal; hard to match | Bot side can also do Whisper-style clean-up of user's list before matching |

## Rollback

KB changes land as a single commit to `kb_faq_ru_v2.md`. Revert = one git command + re-index.

## Trigger

User delivers: list of 10-30 client questions from recent call transcripts.
Until then, this spec is idle.
