#!/usr/bin/env python3
"""Aggregate KB gaps + operational signals across all session analysis reports.

Reads .state/analysis/session_reports.jsonl and produces:
  - Ranked list of KB topic gaps (clients asking, KB not answering)
  - Recurring issue ledger (severity-tagged)
  - New operational counters from the 2026-04-16 client-feedback round:
      readback compliance, change-confirmation compliance, USD->BYN
      conversion rate, EUR/RUB rejection quality, linear-graph success
      rate, stop-command respect rate, profile-hygiene repeat-ask counts,
      defaults-assumed incidents.

Usage:
    python scripts/kb_gap_report.py
    python scripts/kb_gap_report.py --min-count 2 --reports .state/analysis/session_reports.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default=".state/analysis/session_reports.jsonl")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum occurrences to show")
    args = parser.parse_args()

    reports_path = Path(args.reports)
    if not reports_path.exists():
        print(f"No reports file: {reports_path}")
        return

    gap_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    defaults_counter: Counter[str] = Counter()
    repeat_ask_counter: Counter[str] = Counter()
    stop_command_respected = 0
    stop_command_ignored = 0
    readback_done = 0
    readback_skipped = 0
    change_confirmed = 0
    change_skipped = 0
    usd_conversions = 0
    eur_rub_rejections = 0
    linear_requested_total = 0
    linear_successful_total = 0
    calc_calls = 0
    sms_calls = 0
    total_sessions = 0
    total_scores: list[float] = []

    for line in reports_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            continue

        if report.get("skipped") or report.get("error"):
            continue

        total_sessions += 1
        score = report.get("overall_score")
        if score is not None:
            total_scores.append(float(score))

        for gap in report.get("kb_gaps", []):
            gap_counter[gap.strip().lower()] += 1

        for issue in report.get("issues", []):
            severity = issue.get("severity", "minor")
            issue_type = issue.get("type", "unknown")
            issue_counter[f"[{severity}] {issue_type}"] += 1

        # New dimensions (2026-04-16 round)
        tc = report.get("tool_calls") or {}
        if tc.get("calculator_called"):
            calc_calls += 1
        if tc.get("sms_called"):
            sms_calls += 1
        if tc.get("readback_before_first_calc") is True:
            readback_done += 1
        elif tc.get("readback_before_first_calc") is False:
            readback_skipped += 1
        if tc.get("change_confirmed_before_recalc") is True:
            change_confirmed += 1
        elif tc.get("change_confirmed_before_recalc") is False:
            change_skipped += 1
        if tc.get("usd_to_byn_conversion_done"):
            usd_conversions += 1
        if tc.get("eur_rub_rejected_cleanly"):
            eur_rub_rejections += 1
        if isinstance(tc.get("linear_requests_count"), (int, float)):
            linear_requested_total += int(tc["linear_requests_count"])
        if isinstance(tc.get("linear_successes_count"), (int, float)):
            linear_successful_total += int(tc["linear_successes_count"])

        for ev in report.get("stop_command_events") or []:
            if ev.get("bot_respected"):
                stop_command_respected += 1
            else:
                stop_command_ignored += 1

        ph = report.get("profile_hygiene") or {}
        for repeat in ph.get("repeat_asks") or []:
            field = repeat.get("field", "unknown")
            count = int(repeat.get("count", 1) or 1)
            repeat_ask_counter[field] += count

        for case in report.get("defaults_assumed") or []:
            defaults_counter[str(case).strip().lower()] += 1

    if not total_sessions:
        print("No valid session reports found")
        return

    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0

    print(f"Sessions analyzed: {total_sessions}")
    print(f"Average quality score: {avg_score:.1f}/10")
    print()

    gaps = [(topic, count) for topic, count in gap_counter.items() if count >= args.min_count]
    gaps.sort(key=lambda x: x[1], reverse=True)

    if gaps:
        print(f"KB Gaps (topics clients asked about, no KB answer):")
        print(f"{'Topic':<50s} Count")
        print("-" * 60)
        for topic, count in gaps:
            print(f"  {topic:<48s} {count}")
    else:
        print("No KB gaps detected (all questions answered from KB)")

    print()

    issues = [(desc, count) for desc, count in issue_counter.items() if count >= args.min_count]
    issues.sort(key=lambda x: x[1], reverse=True)

    if issues:
        print(f"Recurring issues:")
        print(f"{'Issue':<50s} Count")
        print("-" * 60)
        for desc, count in issues:
            print(f"  {desc:<48s} {count}")
    else:
        print("No recurring issues")

    # ── Operational signals (2026-04-16 client-feedback round) ──
    print()
    print("Operational signals")
    print("-" * 60)

    def _rate(num: int, denom: int) -> str:
        if denom <= 0:
            return "n/a"
        return f"{num}/{denom} ({100*num/denom:.0f}%)"

    print(f"  Calculator calls: {calc_calls}  |  SMS sends: {sms_calls}")
    rb_total = readback_done + readback_skipped
    print(f"  Readback before first calc: {_rate(readback_done, rb_total)}")
    chg_total = change_confirmed + change_skipped
    print(f"  Change-confirm before recalc: {_rate(change_confirmed, chg_total)}")
    print(f"  USD->BYN conversions: {usd_conversions}")
    print(f"  EUR/RUB rejections (clean): {eur_rub_rejections}")
    print(f"  Linear-graph success rate: {_rate(linear_successful_total, linear_requested_total)}")
    stop_total = stop_command_respected + stop_command_ignored
    print(f"  Stop-command respected: {_rate(stop_command_respected, stop_total)}")

    if repeat_ask_counter:
        print()
        print("Profile hygiene (fields bot re-asked, summed across sessions):")
        for field, count in sorted(repeat_ask_counter.items(), key=lambda x: -x[1]):
            if count >= args.min_count:
                print(f"  {field:<48s} {count}")

    if defaults_counter:
        print()
        print("Defaults-assumed incidents (bot stated a default without client confirmation):")
        for case, count in sorted(defaults_counter.items(), key=lambda x: -x[1]):
            if count >= args.min_count:
                print(f"  {case[:48]:<48s} {count}")


if __name__ == "__main__":
    main()
