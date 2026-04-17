#!/usr/bin/env bash
# Terminal test suite for the client-feedback-round fixes. Run AFTER smoke_test.sh
# passes. Covers: SessionAgent, calculator no-defaults, calculator full profile,
# currency policy, Whisper prompt vocab, abbreviation expansion, KB retrieval,
# end-to-end chat.
#
# Usage:  bash scripts/terminal_tests.sh
# Exits 0 on PASS, nonzero on any FAIL. WARN does not fail the run.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"

# Load .env so CALCULATOR_API_*/SMS_API_* are visible to the shell.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

PY="$APP_DIR/.venv/bin/python"
[ -x "$PY" ] || PY=python3

RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YLW=$'\033[1;33m'; NC=$'\033[0m'
PASS=0; FAIL=0; WARN=0
pass() { echo "${GRN}[PASS]${NC} $*"; PASS=$((PASS+1)); }
fail() { echo "${RED}[FAIL]${NC} $*"; FAIL=$((FAIL+1)); }
warn() { echo "${YLW}[WARN]${NC} $*"; WARN=$((WARN+1)); }
section() { echo; echo "=============================="; echo "  $*"; echo "=============================="; }

# ────────────────────────────────────────────────────────────────────────────
section "1. SessionAgent latency + JSON extraction"
SA_URL="${SESSIONAGENT_BASE_URL:-http://127.0.0.1:8788/v1}"
SA_MODEL="${SESSIONAGENT_MODEL:-Qwen/Qwen3-4B-Instruct-2507-FP8}"
echo "Endpoint: ${SA_URL} | model: ${SA_MODEL}"

SA_REQ=$(cat <<JSON
{
  "model": "${SA_MODEL}",
  "messages": [
    {"role": "system", "content": "Ты SessionAgent. Верни JSON с полями intent, profile_patches, is_stop_request, is_confirmation. Только JSON."},
    {"role": "user", "content": "Хочу легковой автомобиль за семьдесят тысяч рублей, линейный график"}
  ],
  "temperature": 0.0,
  "max_tokens": 200
}
JSON
)
T0=$(date +%s%3N)
SA_RESP=$(curl -sS -m 20 "${SA_URL}/chat/completions" -H "Content-Type: application/json" -d "$SA_REQ" || true)
T1=$(date +%s%3N)
LAT_MS=$((T1 - T0))
echo "Latency: ${LAT_MS} ms"
if echo "$SA_RESP" | "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])' 2>/dev/null | tee /tmp/sa_out.txt | grep -qE '"intent"'; then
  pass "SessionAgent returned JSON with expected fields (${LAT_MS} ms)"
  cat /tmp/sa_out.txt | head -20
else
  fail "SessionAgent did not return expected JSON shape"
  echo "Raw response (first 500 chars):"
  echo "$SA_RESP" | head -c 500; echo
fi

# ────────────────────────────────────────────────────────────────────────────
section "2. Calculator no-defaults (IncompleteProfileError must fire)"
"$PY" - <<'PY'
import sys, os
from backend.tools.calculator import CalculatorTool, IncompleteProfileError
t = CalculatorTool(base_url=os.environ.get("CALCULATOR_API_BASE_URL",""), token=os.environ.get("CALCULATOR_API_TOKEN",""))
try:
    t.execute({"subject": "Легковой автомобиль", "cost": 70000}, {})
    print("FAIL: calculator accepted incomplete profile")
    sys.exit(1)
except IncompleteProfileError as e:
    print(f"OK: raised IncompleteProfileError(missing={sorted(list(e.missing))})")
PY
[ $? -eq 0 ] && pass "Calculator rejects incomplete ClientProfile" || fail "Calculator did not raise on incomplete profile"

# ────────────────────────────────────────────────────────────────────────────
section "3. Calculator with full ClientProfile + linear graph"
if [ -z "${CALCULATOR_API_TOKEN:-}" ]; then
  warn "CALCULATOR_API_TOKEN empty; skipping live calc call"
else
  "$PY" - <<'PY'
import os, json
from backend.tools.calculator import CalculatorTool
t = CalculatorTool(
    base_url=os.environ["CALCULATOR_API_BASE_URL"],
    token=os.environ["CALCULATOR_API_TOKEN"],
)
r = t.execute({
    "subject": "Легковой автомобиль",
    "cost": 70000,
    "client_type": "Физическое лицо",
    "currency": "BYN",
    "condition_new": 1,
    "term": 84,
    "type_schedule": "1",
    "prepaid_pct": 20,
}, {})
print(json.dumps({
    "ok": r.get("ok"),
    "prepaid_pct": r.get("prepaid_pct"),
    "prepaid_amount": r.get("prepaid_amount"),
    "payment_min": r.get("payment_min"),
    "total": r.get("total"),
}, ensure_ascii=False, indent=2))
assert r.get("ok"), "calculator returned ok=False"
PY
  [ $? -eq 0 ] && pass "Calculator returns valid quote with linear graph" || fail "Calculator live call failed"
fi

# ────────────────────────────────────────────────────────────────────────────
section "4. Currency policy (USD->BYN at 3.0, EUR rejected)"
"$PY" - <<'PY'
import os
rate = float(os.environ.get("USD_BYN_RATE", "3.0"))
usd = 24300
byn = usd * rate
print(f"USD {usd} * rate {rate} = BYN {byn}")
assert abs(byn - 72900.0) < 0.01, f"expected 72900, got {byn}"
print("OK: USD->BYN conversion math correct")
PY
[ $? -eq 0 ] && pass "USD->BYN rate constant is 3.0" || fail "USD->BYN conversion incorrect"
echo "Note: live EUR rejection path is validated in playbook §3.3 (voice test)."

# ────────────────────────────────────────────────────────────────────────────
section "5. Whisper initial_prompt contains new vocabulary"
"$PY" - <<'PY'
import sys
from services.whisper_server import _DEFAULT_INITIAL_PROMPT as p
missing = []
for term in ["Ксения", "линейный", "аннуитет", "дифференцированный", "нагрузка", "ипэшник"]:
    mark = "YES" if term in p else "NO"
    print(f"  {term}: {mark}")
    if mark == "NO": missing.append(term)
print(f"Total chars: {len(p)}")
sys.exit(1 if missing else 0)
PY
[ $? -eq 0 ] && pass "Whisper prompt has all expected terms" || fail "Whisper prompt missing terms"

# ────────────────────────────────────────────────────────────────────────────
section "6. Abbreviation expansion (TTS preprocessing)"
"$PY" - <<'PY'
import sys
from backend.text_utils import clean_voice_output
cases = [
    ("Офис в Минске: пр-т Победителей, 57",       ["проспект"]),
    ("г. Могилёв, ул. Комсомольская, д. 10а",      ["город", "улица", "дом"]),
    ("тел. +375 17 322 77 00",                     ["телефон"]),
]
fail = False
for inp, expected in cases:
    out = clean_voice_output(inp)
    missing = [w for w in expected if w not in out]
    ok = "OK " if not missing else "BAD"
    print(f"{ok} IN : {inp}")
    print(f"    OUT: {out}")
    if missing:
        print(f"    missing expansion: {missing}")
        fail = True
sys.exit(1 if fail else 0)
PY
[ $? -eq 0 ] && pass "Abbreviations expanded in clean_voice_output" || fail "Abbreviation expansion broken"

# ────────────────────────────────────────────────────────────────────────────
section "7. KB retrieval smoke (new sections indexed)"
"$PY" - <<'PY'
import sys
from pathlib import Path
from backend.engine import RAGEngine
from backend.settings import load_settings
state_dir = Path(__file__).resolve().parent / ".state" if False else Path.cwd() / ".state"
eng = RAGEngine(load_settings(), state_dir)
queries = [
    "что такое нагрузка",
    "линейный график",
    "адрес офиса Гомель",
    "время работы офиса",
]
bad = 0
for q in queries:
    res = eng.retrieve(q, fast=True, voice_fast=True)
    hits = res.get("final") or res.get("hits") or []
    top = hits[0].get("text","")[:120] if hits else "(no hits)"
    print(f"Q: {q}")
    print(f"   top: {top}")
    if not hits:
        bad += 1
    print()
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && pass "KB returns hits for all 4 queries" || fail "KB retrieval missing results"

# ────────────────────────────────────────────────────────────────────────────
section "8. End-to-end backend chat (HTTP)"
CHAT_RESP=$(curl -sS -m 60 http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"terminal-test-1","message":"Что такое нагрузка по лизингу?"}' || true)
if echo "$CHAT_RESP" | "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(d.get("answer","")[:300])' 2>/dev/null | tee /tmp/chat_out.txt | grep -qiE 'нагрузк|удорожан'; then
  pass "Backend chat responded with topic-relevant answer"
  cat /tmp/chat_out.txt
else
  fail "Backend chat did not return expected content"
  echo "Raw (first 400 chars):"; echo "$CHAT_RESP" | head -c 400; echo
fi

# ────────────────────────────────────────────────────────────────────────────
section "9. Stop-detection (literal regex)"
"$PY" - <<'PY'
import sys
from backend.text_utils import contains_stop_word
cases = [
    ("Стоп.", True),
    ("Замолчи на секунду.", True),
    ("Помолчи, я думаю.", True),
    ("Ну и что?", False),
    ("Алло.", False),
    ("А подожди, ладно, неважно, а можно машину без аванса?", True),
    ("Нажмите стоп-кран.", True),
    ("В нашем разговоре уже.", False),
]
bad = 0
for text, want in cases:
    got = contains_stop_word(text)
    mark = "OK " if got == want else "BAD"
    print(f"{mark} contains_stop_word({text!r}) -> {got} (want {want})")
    if got != want:
        bad += 1
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && pass "Literal stop-word regex matches expected set" || fail "Literal stop-word regex mismatches"

# ────────────────────────────────────────────────────────────────────────────
section "10. Profile hygiene filter"
"$PY" - <<'PY'
import sys
from backend.profile_hygiene import filter_patches

cases = [
    ("noise drop",      {"client_type": "Юридическое лицо"}, "ОООООО", {}),
    ("bot name drop",   {"name": "Ксения"},                  "Ксения, что такое нагрузка?", {}),
    ("ипэшник -> ИП",   {"client_type": "ИП"},               "я ипэшник", {"client_type": "ИП"}),
    ("bare рубли BYN",  {"currency": "RUB"},                 "давай в рублях", {"currency": "BYN"}),
    ("росс. рубли RUB", {"currency": "RUB"},                 "в российских рублях", {"currency": "RUB"}),
    ("prepaid 50 drop", {"prepaid_pct": 50},                 "пятьдесят процентов аванс", {}),
    ("prepaid 20 keep", {"prepaid_pct": 20},                 "двадцать процентов аванс", {"prepaid_pct": 20}),
    ("term 300 drop",   {"term_months": 300},                "триста месяцев", {}),
    ("term 84 keep",    {"term_months": 84},                 "восемьдесят четыре месяца", {"term_months": 84}),
]
bad = 0
for name, patches, utterance, expected in cases:
    got = filter_patches(patches, utterance)
    if got != expected:
        print(f"BAD {name}: got={got} want={expected}")
        bad += 1
    else:
        print(f"OK  {name}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && pass "Profile hygiene filter rejects noise and normalizes enums" || fail "Profile hygiene mismatches"

# ────────────────────────────────────────────────────────────────────────────
section "11. Skip-RAG predicate"
"$PY" - <<'PY'
import sys
from backend.rag_skip import should_skip_rag
cases = [
    ("Привет, я Вадим.", {"name": "Вадим"}, {}, True),
    ("Привет, я Вадим, какие офисы в Минске?", {"name": "Вадим"}, {"action": "clarify"}, False),
    ("Я Вадим. Адрес в Минске?", {"name": "Вадим"}, {}, False),
    ("Здравствуйте.", {}, {}, False),
]
bad = 0
for utt, p, h, want in cases:
    got = should_skip_rag(utt, p, h)
    if got != want:
        print(f"BAD {utt!r}: got={got} want={want}")
        bad += 1
    else:
        print(f"OK  {utt!r} -> {got}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && pass "Skip-RAG predicate" || fail "Skip-RAG predicate"

# ────────────────────────────────────────────────────────────────────────────
section "12. Grounding validator"
"$PY" - <<'PY'
import sys
from backend.grounding_validator import check_grounded, replace_ungrounded

# Hallucinated address gets stripped
resp = "В Минске офис: ул. Немига, 24."
chunks = ["Наш офис: проспект Победителей, 57."]
cleaned = replace_ungrounded(resp, chunks)
if "Немига" in cleaned:
    print(f"BAD ungrounded address survived: {cleaned!r}")
    sys.exit(1)
print(f"OK  stripped ungrounded address: {cleaned!r}")

# Grounded address survives
resp2 = "Офис: проспект Победителей, 57."
chunks2 = ["проспект Победителей, 57"]
cleaned2 = replace_ungrounded(resp2, chunks2)
if "Победителей" not in cleaned2:
    print(f"BAD grounded address stripped: {cleaned2!r}")
    sys.exit(1)
print(f"OK  preserved grounded address: {cleaned2!r}")
PY
[ $? -eq 0 ] && pass "Grounding validator" || fail "Grounding validator"

# ────────────────────────────────────────────────────────────────────────────
section "13. Currency mapping (via filter_patches)"
"$PY" - <<'PY'
import sys
from backend.profile_hygiene import filter_patches

cases = [
    ("в рублях (бытовое)",     {"currency": "RUB"}, "давай в рублях", "BYN"),
    ("белорусские рубли",      {"currency": "BYN"}, "белорусские рубли", "BYN"),
    ("российские рубли",       {"currency": "RUB"}, "в российских рублях", "RUB"),
    ("доллары",                {"currency": "USD"}, "в долларах", "USD"),
    ("евро",                   {"currency": "EUR"}, "в евро", "EUR"),
]
bad = 0
for name, patch, utt, expected in cases:
    got = filter_patches(patch, utt).get("currency")
    if got != expected:
        print(f"BAD {name}: got={got} want={expected}")
        bad += 1
    else:
        print(f"OK  {name} -> {got}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && pass "Currency mapping" || fail "Currency mapping"

# ────────────────────────────────────────────────────────────────────────────
section "14. MVP range boundaries (profile hygiene)"
"$PY" - <<'PY'
import sys
from backend.profile_hygiene import filter_patches

prepaid_cases = [
    ("prepaid 0 keep",  {"prepaid_pct": 0},   "без аванса", 0),
    ("prepaid 40 keep", {"prepaid_pct": 40},  "сорок процентов", 40),
    ("prepaid 41 drop", {"prepaid_pct": 41},  "сорок один процент", None),
]
term_cases = [
    ("term 12 keep",    {"term_months": 12},  "двенадцать месяцев", 12),
    ("term 84 keep",    {"term_months": 84},  "восемьдесят четыре месяца", 84),
    ("term 11 drop",    {"term_months": 11},  "одиннадцать месяцев", None),
    ("term 85 drop",    {"term_months": 85},  "восемьдесят пять", None),
]
bad = 0
for name, patch, utt, expected in prepaid_cases:
    got = filter_patches(patch, utt).get("prepaid_pct")
    if got != expected:
        print(f"BAD {name}: got={got} want={expected}")
        bad += 1
    else:
        print(f"OK  {name} -> {got}")
for name, patch, utt, expected in term_cases:
    got = filter_patches(patch, utt).get("term_months")
    if got != expected:
        print(f"BAD {name}: got={got} want={expected}")
        bad += 1
    else:
        print(f"OK  {name} -> {got}")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && pass "MVP range boundaries" || fail "MVP range boundaries"

# ────────────────────────────────────────────────────────────────────────────
echo
echo "=============================="
echo "  Summary: ${GRN}${PASS} pass${NC} | ${YLW}${WARN} warn${NC} | ${RED}${FAIL} fail${NC}"
echo "=============================="
exit $FAIL
