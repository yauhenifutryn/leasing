#!/usr/bin/env python3
"""calc_smoke.py — scripted scenario harness for the chat dispatcher.

Drives /api/text-turn through a deterministic set of calculator scenarios
and reports what the bot did at each step. Faster than manual UI clicking,
catches regressions before live testing.

Usage:
  # Run against a deployed server:
  python scripts/calc_smoke.py --base http://127.0.0.1:8000

  # Run a single scenario:
  python scripts/calc_smoke.py --base http://127.0.0.1:8000 --only multi_param_oneshot

  # Verbose (shows every reply):
  python scripts/calc_smoke.py --base http://127.0.0.1:8000 -v

Each scenario is a list of (user_utterance, expected_check) pairs:
  - user_utterance: what we POST as `message`
  - expected_check: a callable that receives the response dict and returns
    None on pass, or a string describing the failure reason

The harness reports a pass/fail matrix and exits 0 only when ALL pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any, Callable
from urllib import request, error

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
DIM = "\033[2m"
RESET = "\033[0m"


# ── HTTP plumbing ─────────────────────────────────────────────────────────

def post_turn(base: str, session_id: str, message: str, name: str = "Тест", phone: str = "") -> dict:
    """POST one /api/text-turn and return the parsed response."""
    body = json.dumps({
        "message": message,
        "session_id": session_id,
        "name": name,
        "phone": phone,
    }).encode("utf-8")
    req = request.Request(
        f"{base}/api/text-turn",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── Check helpers (a check returns None on pass, str on fail) ──────────────

def reply_contains(*needles: str) -> Callable[[dict], str | None]:
    def _c(r: dict) -> str | None:
        reply = (r.get("reply") or "").lower()
        for n in needles:
            if n.lower() not in reply:
                return f'reply missing "{n}" — got: {r.get("reply")!r}'
        return None
    return _c


def reply_contains_any(*needles: str) -> Callable[[dict], str | None]:
    def _c(r: dict) -> str | None:
        reply = (r.get("reply") or "").lower()
        if any(n.lower() in reply for n in needles):
            return None
        return f'reply matched none of {needles!r} — got: {r.get("reply")!r}'
    return _c


def action_is(*actions: str) -> Callable[[dict], str | None]:
    def _c(r: dict) -> str | None:
        a = r.get("action", "")
        if a in actions:
            return None
        return f'action expected {actions!r}, got {a!r}'
    return _c


def profile_state(*states: str) -> Callable[[dict], str | None]:
    def _c(r: dict) -> str | None:
        s = r.get("profile_state", "")
        if s in states:
            return None
        return f'profile_state expected {states!r}, got {s!r}'
    return _c


def missing_does_not_contain(*fields: str) -> Callable[[dict], str | None]:
    """Use after a turn that should have populated the listed fields."""
    def _c(r: dict) -> str | None:
        m = set(r.get("missing", []) or [])
        leaked = m & set(fields)
        if leaked:
            return f'fields still missing after they should have been captured: {sorted(leaked)}'
        return None
    return _c


def all_checks(*checks: Callable[[dict], str | None]) -> Callable[[dict], str | None]:
    def _c(r: dict) -> str | None:
        for ch in checks:
            err = ch(r)
            if err:
                return err
        return None
    return _c


# ── Scenarios ─────────────────────────────────────────────────────────────
#
# Each scenario is a list of (utterance, check, label) tuples.
# Scenarios get a fresh session_id each run (no cross-talk).
# Add new scenarios at the bottom; they're discovered by name automatically.

SCENARIOS: dict[str, list[tuple[str, Callable[[dict], str | None], str]]] = {
    # --- Issue #3 from 2026-05-07 live test ---
    "phys_litso_alias": [
        ("хочу машину в лизинг", reply_contains_any("новый", "стоимость", "цена", "лица"), "ack-or-ask"),
        ("физ лицо",
            all_checks(
                # The whole point: classifier must accept "физ лицо" as Физическое лицо.
                # If it doesn't, the bot will re-ask the same question, and the reply
                # will contain "физическое или юридическое" again.
                lambda r: ("физическое или юридическое" in (r.get("reply") or "").lower())
                          and 'classifier rejected "физ лицо" alias — bot re-asked'
                          or None,
            ),
            "alias-accept"),
    ],
    "yur_litso_alias": [
        ("хочу лизинг", reply_contains_any("новый", "стоимость", "цена", "лица"), "ack-or-ask"),
        ("юр лицо",
            lambda r: ("физическое или юридическое" in (r.get("reply") or "").lower())
                      and 'classifier rejected "юр лицо" alias — bot re-asked'
                      or None,
            "alias-accept"),
    ],
    "ip_alias": [
        ("хочу лизинг", reply_contains_any("новый", "стоимость", "цена", "лица"), "ack-or-ask"),
        ("ИП",
            lambda r: ("физическое или юридическое" in (r.get("reply") or "").lower())
                      and 'classifier rejected "ИП" alias — bot re-asked'
                      or None,
            "alias-accept"),
    ],
    "used_with_age_oneshot": [
        # Replicates the failing live conversation: user gives condition + age in one turn.
        ("хочу подержанный автомобиль 2 года и 40000 километров",
            missing_does_not_contain("age_years"),
            "age captured from joint utterance"),
    ],
    "condition_persistence": [
        # User says "подержанный" once, then we proceed. The bot should NOT
        # re-ask condition_new at the next turn.
        ("хочу машину подержанный", action_is("EmitClarify", "EmitGenericClarify"), "ack"),
        ("100 тысяч долларов",
            lambda r: ("новый или подержанный" in (r.get("reply") or "").lower())
                      and 'condition_new lost between turns — bot re-asked'
                      or None,
            "condition retained"),
    ],
    "multi_param_oneshot_used": [
        # Big rich utterance — what voice users frequently send.
        (
            "хочу подержанный легковой автомобиль возраст 2 года стоимость 100 тысяч долларов аванс 25 процентов на 36 месяцев",
            all_checks(
                missing_does_not_contain("age_years", "term_months", "prepaid", "type_schedule"),
            ),
            "all params captured one-shot",
        ),
    ],
    "multi_param_oneshot_new": [
        (
            "хочу новый легковой автомобиль за 60 тысяч BYN аванс 30 процентов срок 48 месяцев физ лицо равные платежи",
            all_checks(
                missing_does_not_contain("term_months", "prepaid", "type_schedule"),
            ),
            "all params captured one-shot (new car)",
        ),
    ],
    "currency_usd_to_byn": [
        # Voice flow: USD declared, calc must run with BYN-converted cost.
        ("новый авто 50 тысяч долларов аванс 20 процентов 36 месяцев физ лицо равные платежи",
            action_is("FireCalc", "EmitReadback"),
            "fires calc or readback"),
    ],
    "calc_then_detail_then_sms": [
        # The bug 8 sequential-offer fix. After calc → "Хотите подробный?"
        # → "давай" → detail → "Отправить по СМС?" → "давай" → SMS fires.
        (
            "новый легковой за 60000 BYN аванс 30 процентов 36 месяцев физ лицо равные платежи",
            action_is("FireCalc", "EmitReadback"),
            "calc fires",
        ),
        # If calc fired, next turn is the detail/SMS sequence. If readback first,
        # we need a "да" turn before this. Harness handles both via reply check.
        ("давай",
            reply_contains_any("выкупной", "общая сумма", "удорожание", "смс"),
            "detail returned (or SMS offered if calc was already done)"),
        ("давай",
            reply_contains_any("отправ", "смс", "график"),
            "second yes acknowledges (SMS/detail)"),
    ],
    "owner_kb_clean": [
        # The 2026-05-04 fix. Clean query should hit the new section.
        ("Кто владелец вашей компании?",
            reply_contains("Mikro Kapital"),
            "owner from topical KB"),
    ],
    "lease_types_kb": [
        ("Что такое лизинг и какие виды у вас есть?",
            reply_contains_any("финансовый", "оперативный", "возвратный"),
            "lease types section retrieved"),
    ],
    "minsk_hours_kb": [
        ("Когда работаете в Минске?",
            reply_contains("суббот", "воскресень"),
            "Minsk weekend hours mentioned"),
    ],
    "tts_chat_render_dotby": [
        ("Какой у вас сайт?",
            lambda r: ("точка бай" in (r.get("reply") or "").lower())
                      and 'TTS-phonetic "точка бай" leaked into chat — render module not applied'
                      or None,
            ".by rendered as text not phonetic"),
    ],
    "name_in_context": [
        # If we set name="Иван", later turn the bot should know the user's name.
        ("Как меня зовут?",
            reply_contains_any("иван", "ваше имя"),
            "bot recalls intake name (or asks if not provided)"),
    ],
}


# ── Runner ────────────────────────────────────────────────────────────────

def run_scenario(base: str, name: str, steps: list, *, verbose: bool, intake_name: str = "Иван") -> tuple[str, list[tuple[str, str, str | None]]]:
    """Run one scenario; return (overall_status, list_of_step_results)."""
    sid = f"chat-smoke-{uuid.uuid4().hex[:10]}"
    results: list[tuple[str, str, str | None]] = []
    for i, (utterance, check, label) in enumerate(steps, 1):
        r = post_turn(base, sid, utterance, name=intake_name)
        if not r.get("ok") and "error" in r:
            results.append((label, "FAIL", f"transport error: {r['error']}"))
            return "FAIL", results
        err = check(r) if callable(check) else None
        status = "PASS" if err is None else "FAIL"
        if verbose:
            print(f"    [{i}] {DIM}{utterance!r}{RESET}")
            print(f"        → action={r.get('action')!r} state={r.get('profile_state')!r}")
            print(f"        → reply={DIM}{(r.get('reply') or '')[:160]!r}{RESET}")
            if err:
                print(f"        → {FAIL} {err}")
        results.append((label, status, err))
        time.sleep(0.05)  # gentle pacing
    overall = "PASS" if all(s == "PASS" for _, s, _ in results) else "FAIL"
    return overall, results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--only", action="append", default=[], help="Run only named scenarios (repeatable)")
    p.add_argument("--list", action="store_true", help="List scenario names and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.list:
        for name in SCENARIOS:
            print(name)
        return 0

    targets = args.only if args.only else list(SCENARIOS.keys())
    unknown = [t for t in targets if t not in SCENARIOS]
    if unknown:
        print(f"Unknown scenarios: {unknown}", file=sys.stderr)
        return 2

    print(f"Calc smoke against {args.base} — {len(targets)} scenarios\n")
    overall_pass = True
    summary: list[tuple[str, str, list]] = []
    for name in targets:
        steps = SCENARIOS[name]
        if args.verbose:
            print(f"\n=== {name} ===")
        status, step_results = run_scenario(args.base, name, steps, verbose=args.verbose)
        summary.append((name, status, step_results))
        if status != "PASS":
            overall_pass = False

    # Final report
    print("\n" + "=" * 72)
    print(f"{'Scenario':<36} {'Status':<8} {'Failing step / reason'}")
    print("-" * 72)
    for name, status, step_results in summary:
        status_str = PASS if status == "PASS" else FAIL
        if status == "PASS":
            print(f"{name:<36} {status_str}")
        else:
            first_fail = next((sr for sr in step_results if sr[1] == "FAIL"), None)
            label = first_fail[0] if first_fail else "?"
            reason = (first_fail[2] or "")[:200] if first_fail else ""
            print(f"{name:<36} {status_str}     [{label}] {reason}")
    print("=" * 72)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
