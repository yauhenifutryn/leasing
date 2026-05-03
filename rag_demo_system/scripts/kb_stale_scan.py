#!/usr/bin/env python3
"""Scan KB entries for stale-number drift.

Section 7 Phase A.2 of the master plan
(`.planning/master_plan_2026_04_18/07_kb_refinement.md`). Read-only.

Loads `knowledge_base/kb_faq_ru.yaml`, runs a catalogued set of regex
patterns over each entry's free-text fields (`best_answer`,
`canonical_question`, `eligibility_rules`, `compliance_notes`,
`handoff_when`, plus any string list values), groups hits by
`(category, pattern, value)`, and emits a markdown report that surfaces
drift candidates: pattern-value combinations where multiple entries in
the same category report different numbers for what's likely the same
underlying constraint.

The report is the input to Phase B.1 (surgical stale-number fixes). It
does NOT propose fixes — disagreements are flagged for human review
because legitimately-different values exist (e.g., advance for физлица
vs ИП) that automation can't disambiguate.

Usage:
    python rag_demo_system/scripts/kb_stale_scan.py
    python rag_demo_system/scripts/kb_stale_scan.py --kb knowledge_base/kb_faq_ru.yaml --out /tmp/scan.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_KB_PATH = REPO_ROOT / "knowledge_base" / "kb_faq_ru.yaml"
DEFAULT_OUT_PATH = REPO_ROOT / "docs" / "superpowers" / f"kb-stale-scan-{date.today().isoformat()}.md"

# Free-text fields scanned for numbers. Operational lists are scanned too because
# stale numbers tend to live in eligibility_rules / compliance_notes (e.g.
# "возраст до 75 лет" repeated inconsistently across entries).
SCANNED_FIELDS: tuple[str, ...] = (
    "canonical_question",
    "best_answer",
    "eligibility_rules",
    "required_fields",
    "compliance_notes",
    "handoff_when",
    "empathy_patterns",
    "followups",
)


@dataclass(frozen=True)
class Pattern:
    """A regex pattern that captures a (label, raw_value) tuple per match.

    `value_group` is the regex group whose contents represent the canonical
    numeric value for grouping. `unit` is appended to the value for display.
    """

    name: str
    description: str
    regex: re.Pattern[str]
    value_group: int
    unit: str = ""


def _re(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)


# Pattern catalogue. Russian morphology handled via `\w*` suffix on roots.
PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        name="advance_percent",
        description="Advance percentage (аванс N% / N% аванс) with up to 3 intervening words",
        regex=_re(
            r"\bаванс\w*\s*(?:[—:\-,]\s*)?(?:[а-яёЁ]+\s+){0,3}(?:от\s+)?(\d{1,2})\s*%"
            r"|(\d{1,2})\s*%(?:\s+[а-яёЁ]+){0,3}\s+\bаванс\w*",
        ),
        value_group=1,
        unit="%",
    ),
    Pattern(
        name="term_months",
        description="Lease term in months (N мес / до N месяцев)",
        regex=_re(r"\b(\d{1,3})\s*месяц\w*|\bдо\s+(\d{1,3})\s*мес\b"),
        value_group=1,
        unit=" мес",
    ),
    Pattern(
        name="age_lower",
        description="Lower age limit (от N лет / от N до / с N лет)",
        regex=_re(r"\b(?:от|с)\s+(\d{2})\s+(?:до|лет|года|год)\b"),
        value_group=1,
        unit=" лет",
    ),
    Pattern(
        name="age_upper",
        description="Upper age limit (до N лет)",
        regex=_re(r"\bдо\s+(\d{2,3})\s+(?:лет|года|год)\b"),
        value_group=1,
        unit=" лет",
    ),
    Pattern(
        name="restrictive_only",
        description="Restrictive 'только N' / 'не более N' / 'не менее N'",
        regex=_re(r"\b(?:только|не\s+бол(?:ее|ьше)|не\s+мен(?:ее|ьше))\s+(\d{1,4})\b"),
        value_group=1,
        unit="",
    ),
    Pattern(
        name="amount_thousands",
        description="Currency amounts in thousands (от N тыс / N тысяч)",
        regex=_re(r"\b(?:от\s+)?(\d{1,4})\s*тыс\w*"),
        value_group=1,
        unit=" тыс",
    ),
    Pattern(
        name="pdn",
        description="ПДН / debt-to-income ratio (часто-перевранный термин)",
        regex=_re(r"\b(?:ПДН|пдн|нагрузк\w+)\b"),
        value_group=0,
        unit="",
    ),
)


@dataclass
class Hit:
    pattern_name: str
    value: str  # canonical value for grouping (e.g. "30")
    unit: str
    snippet: str  # ~80 chars of surrounding context
    field: str
    entry_index: int
    intent: str
    category: str
    subtopic: str


def load_entries(yaml_path: Path) -> list[dict]:
    import yaml  # local import keeps the module importable without pyyaml

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("entries", "items", "faq", "data"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        raise SystemExit(
            f"Unexpected YAML root: dict with keys {list(raw.keys())[:5]}; expected a list",
        )
    if not isinstance(raw, list):
        raise SystemExit(f"Unexpected YAML root type: {type(raw).__name__}")
    return raw


def _iter_strings(value) -> Iterable[str]:
    """Yield every string under a YAML value (handles lists of strings)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)


def _snippet(text: str, span: tuple[int, int], width: int = 80) -> str:
    """Return ~width chars of context around [span], collapsed whitespace."""
    start, end = span
    pad = max(0, (width - (end - start)) // 2)
    s = max(0, start - pad)
    e = min(len(text), end + pad + (width - (end - start)) - pad)
    snippet = text[s:e].replace("\n", " ").strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if s > 0:
        snippet = "…" + snippet
    if e < len(text):
        snippet = snippet + "…"
    return snippet


def scan_entry(entry: dict, entry_index: int, patterns: Iterable[Pattern]) -> list[Hit]:
    intent = str(entry.get("intent") or "(no intent)")
    category = str(entry.get("category") or "(no category)")
    subtopic = str(entry.get("subtopic") or "")
    hits: list[Hit] = []

    for field_name in SCANNED_FIELDS:
        value = entry.get(field_name)
        if value is None:
            continue
        for text in _iter_strings(value):
            if not text:
                continue
            for pat in patterns:
                for m in pat.regex.finditer(text):
                    raw = m.group(pat.value_group) if pat.value_group else m.group(0)
                    if raw is None:
                        # alternation can put the value in a different group; pick first non-None
                        groups = [g for g in m.groups() if g is not None]
                        raw = groups[0] if groups else m.group(0)
                    hits.append(
                        Hit(
                            pattern_name=pat.name,
                            value=str(raw).strip(),
                            unit=pat.unit,
                            snippet=_snippet(text, m.span()),
                            field=field_name,
                            entry_index=entry_index,
                            intent=intent,
                            category=category,
                            subtopic=subtopic,
                        ),
                    )
    return hits


def render_report(
    all_hits: list[Hit],
    patterns: Iterable[Pattern],
    entries: list[dict],
    yaml_path: Path,
) -> str:
    pattern_by_name = {p.name: p for p in patterns}
    by_pattern_category_value: dict[tuple[str, str, str], list[Hit]] = defaultdict(list)
    for h in all_hits:
        by_pattern_category_value[(h.pattern_name, h.category, h.value)].append(h)

    by_pattern_category: dict[tuple[str, str], dict[str, list[Hit]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for (pname, cat, val), hits in by_pattern_category_value.items():
        by_pattern_category[(pname, cat)][val] = hits

    lines: list[str] = []
    lines.append(f"# KB Stale-Number Scan — {date.today().isoformat()}")
    lines.append("")
    lines.append(f"Source: `{yaml_path.relative_to(REPO_ROOT)}` ({len(entries)} entries)")
    lines.append(f"Total numeric/term hits: **{len(all_hits)}**")
    lines.append("")
    lines.append(
        "Drift candidates: any (pattern, category) cell with **≥2 distinct values** is a candidate. "
        "Read the snippets — some splits are legitimate (физлица vs ИП); others are stale copies.",
    )
    lines.append("")

    lines.append("## Per-pattern summary")
    lines.append("")
    lines.append("| Pattern | Description | Total hits | Categories with ≥2 values |")
    lines.append("|---|---|---:|---:|")
    for pat in patterns:
        pname = pat.name
        pat_hits = [h for h in all_hits if h.pattern_name == pname]
        cats_with_multi = 0
        cats_seen: set[str] = set()
        for h in pat_hits:
            cats_seen.add(h.category)
        for cat in cats_seen:
            distinct = {h.value for h in pat_hits if h.category == cat}
            if len(distinct) >= 2:
                cats_with_multi += 1
        lines.append(f"| `{pname}` | {pat.description} | {len(pat_hits)} | {cats_with_multi} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Drift candidates (≥2 distinct values per category)")
    lines.append("")
    drift_seen = False
    for pname in (p.name for p in patterns):
        pat = pattern_by_name[pname]
        cats = sorted({c for (p_, c) in by_pattern_category if p_ == pname})
        for cat in cats:
            value_groups = by_pattern_category[(pname, cat)]
            if len(value_groups) < 2:
                continue
            drift_seen = True
            lines.append(f"### `{pname}` × *{cat}* — {len(value_groups)} distinct values")
            lines.append("")
            lines.append("| value | count | sample intents | sample snippet |")
            lines.append("|---|---:|---|---|")
            for value in sorted(value_groups, key=lambda v: (len(v), v)):
                hits = value_groups[value]
                intents = sorted({h.intent for h in hits})[:4]
                intents_str = ", ".join(f"`{i}`" for i in intents)
                if len({h.intent for h in hits}) > 4:
                    intents_str += f", … (+{len({h.intent for h in hits}) - 4} more)"
                sample_snip = hits[0].snippet
                if len(sample_snip) > 110:
                    sample_snip = sample_snip[:107] + "…"
                lines.append(
                    f"| **{value}{pat.unit}** | {len(hits)} | {intents_str} | {sample_snip} |",
                )
            lines.append("")

    if not drift_seen:
        lines.append("_(No drift candidates found — every category has a single value per pattern.)_")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Full hit catalog (per pattern, all values)")
    lines.append("")
    for pname in (p.name for p in patterns):
        pat = pattern_by_name[pname]
        cats = sorted({c for (p_, c) in by_pattern_category if p_ == pname})
        if not cats:
            continue
        lines.append(f"### Pattern `{pname}` — {pat.description}")
        lines.append("")
        for cat in cats:
            value_groups = by_pattern_category[(pname, cat)]
            total = sum(len(h) for h in value_groups.values())
            lines.append(f"#### *{cat}* — {len(value_groups)} distinct value(s), {total} hit(s)")
            lines.append("")
            for value in sorted(value_groups, key=lambda v: (len(v), v)):
                hits = value_groups[value]
                lines.append(f"- **{value}{pat.unit}** ({len(hits)} hit(s))")
                for h in hits[:6]:
                    lines.append(f"  - `{h.intent}` ({h.field}): {h.snippet}")
                if len(hits) > 6:
                    lines.append(f"  - … (+{len(hits) - 6} more)")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan KB entries for stale-number drift (Section 7 Phase A.2).",
    )
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    if not args.kb.exists():
        print(f"KB file not found: {args.kb}", file=sys.stderr)
        sys.exit(1)

    entries = load_entries(args.kb)
    all_hits: list[Hit] = []
    for i, entry in enumerate(entries):
        all_hits.extend(scan_entry(entry, i, PATTERNS))

    report = render_report(all_hits, PATTERNS, entries, args.kb)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")

    out_display = args.out.resolve()
    try:
        out_display = out_display.relative_to(REPO_ROOT)
    except ValueError:
        pass

    drift_count = 0
    by_cat_val: dict[tuple[str, str], set[str]] = defaultdict(set)
    for h in all_hits:
        by_cat_val[(h.pattern_name, h.category)].add(h.value)
    for vals in by_cat_val.values():
        if len(vals) >= 2:
            drift_count += 1

    print(f"Wrote {out_display}", file=sys.stderr)
    print(
        f"Hits: {len(all_hits)} total across {len(PATTERNS)} patterns. "
        f"Drift candidates (pattern×category with ≥2 distinct values): {drift_count}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
