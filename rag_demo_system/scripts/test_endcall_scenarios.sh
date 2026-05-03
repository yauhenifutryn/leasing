#!/usr/bin/env bash
# Bug 22 — terminal smoke tests for EndCall scenarios.
#
# Runs every dispatcher path that affects EndCall through pytest, with
# a one-line summary per scenario. Use BEFORE/AFTER a deploy to verify
# the EndCall behaviour didn't regress (the live SIP test only covers
# one or two scenarios per call).
#
# Usage:
#   bash scripts/test_endcall_scenarios.sh         # standard run
#   bash scripts/test_endcall_scenarios.sh -v      # verbose pytest

set -euo pipefail
cd "$(dirname "$0")/.."

VERBOSE_FLAG=""
if [[ "${1:-}" == "-v" ]]; then
    VERBOSE_FLAG="-v"
fi

# Pick the right Python: prefer the project venv (server / production
# layout), fall back to system python3 (developer laptop). Both invoke
# pytest as a module so PATH isn't relied on.
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo "ERROR: no python interpreter found (.venv/bin/python or python3)" >&2
    exit 1
fi
echo "Using interpreter: $PY"
echo

echo "=== Bug 22 EndCall — full scenario suite ==="
echo

if [[ -n "$VERBOSE_FLAG" ]]; then
    "$PY" -m pytest tests/test_endcall_dispatcher.py "$VERBOSE_FLAG" --no-header -q
else
    "$PY" -m pytest tests/test_endcall_dispatcher.py --no-header -q
fi

echo
echo "=== Surface-form regex coverage (positive + negative) ==="
"$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
from backend.turn_dispatcher import _is_goodbye_utterance

POSITIVE = [
    "до свидания",
    "До свидания.",
    "До свидания!",
    "Пока",
    "пока-пока",
    "всего доброго",
    "всего хорошего",
    "спасибо, всё",
    "Спасибо, всё.",
    "всё, спасибо",
    "больше ничего не нужно",
]

NEGATIVE = [
    "до свидания, а ещё один вопрос",
    "хорошо, спасибо",
    "пока думаю",
    "ладно",
    "спасибо большое за помощь",
    "ты ещё тут?",
    "пока не понял",
]

print(f"{'Surface':<45} | match")
print("-" * 60)
ok = True
for utt in POSITIVE:
    m = _is_goodbye_utterance(utt)
    print(f"{utt[:43]:<45} | {'YES' if m else 'NO ':<3}{' ✓' if m else ' ✗ EXPECTED YES'}")
    if not m:
        ok = False
for utt in NEGATIVE:
    m = _is_goodbye_utterance(utt)
    print(f"{utt[:43]:<45} | {'YES' if m else 'NO ':<3}{' ✗ EXPECTED NO' if m else ' ✓'}")
    if m:
        ok = False
print()
print("Regex result:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo
echo "=== Dispatcher integration: pre_turn_state matrix ==="
"$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
from backend.classifier_schema import ClassifierOutput
from backend.session import ClientProfile, ProfileState
from backend.turn_action import EndCall
from backend.turn_dispatcher import apply_turn

def case(name, state, change_field, intent, utt, expect_endcall):
    p = ClientProfile()
    p.state = state
    co = ClassifierOutput.model_validate(
        {"intent": intent, "is_confirmation": False, "is_stop_request": False,
         **({"change_field": change_field, "change_value": 12, "action": "change_param"} if change_field else {})},
        context={"utterance": utt},
    )
    action = apply_turn(p, co, utterance=utt)
    is_end = isinstance(action, EndCall)
    ok = is_end == expect_endcall
    print(f"  {'✓' if ok else '✗'} {name:<60} action={type(action).__name__:<22} expect_endcall={expect_endcall}")
    return ok

results = []
results.append(case("COLLECTING + END_CALL intent + clean farewell", ProfileState.COLLECTING, None, "END_CALL", "до свидания", True))
results.append(case("CONFIRMED + END_CALL intent + clean farewell", ProfileState.CONFIRMED, None, "END_CALL", "спасибо, до свидания", True))
results.append(case("COLLECTING + CONVERSATION intent + goodbye regex", ProfileState.COLLECTING, None, "CONVERSATION", "до свидания", True))
results.append(case("READBACK_PENDING — must NOT hang up", ProfileState.READBACK_PENDING, None, "END_CALL", "до свидания", False))
results.append(case("CHANGE_PENDING — must NOT hang up", ProfileState.CHANGE_PENDING, None, "END_CALL", "до свидания", False))
results.append(case("change_field set — must NOT hang up", ProfileState.CONFIRMED, "term_months", "END_CALL", "поменяй на 48, до свидания", False))
results.append(case("CONVERSATION + non-goodbye — must NOT hang up", ProfileState.COLLECTING, None, "CONVERSATION", "расскажите про условия", False))
results.append(case("Ambiguous 'до свидания, а ещё вопрос' — must NOT hang up", ProfileState.COLLECTING, None, "RAG", "до свидания, а ещё один вопрос про лизинг", False))

print()
print("Dispatcher result:", "PASS" if all(results) else "FAIL")
sys.exit(0 if all(results) else 1)
PY

echo
echo "=== TtsSink.disconnect() interface check ==="
"$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
from backend.execute_adapters import TtsSink

methods = [m for m in dir(TtsSink) if not m.startswith("_")]
required = {"say", "disconnect"}
missing = required - set(methods)
if missing:
    print(f"  ✗ TtsSink missing methods: {missing}")
    sys.exit(1)
print(f"  ✓ TtsSink has required methods: {sorted(required)}")
print(f"  ✓ TtsSink.disconnect signature is awaitable: {callable(TtsSink.disconnect)}")
PY

echo
echo "=== ALL SCENARIOS COMPLETE ==="
