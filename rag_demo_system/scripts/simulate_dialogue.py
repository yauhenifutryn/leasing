"""Replay a scripted conversation against the apply_turn dispatcher.

Calls the LIVE SessionAgent classifier (Qwen3-4B on :8788 by default) with
the same conversation context the production WebSocket flow uses, then runs
apply_turn locally and renders the resulting TurnAction as the bot would
speak it. No audio, no Jambonz — pure dialogue simulation.

Useful for verifying classifier-prompt and dispatcher-logic changes without
audio/SIP/STT noise. Each turn shows: classifier raw JSON, apply_turn
TurnAction, rendered bot text, and the running profile snapshot.

Usage
-----
    cd rag_demo_system
    # MUST use the project venv — pydantic and other backend deps live there:
    .venv/bin/python scripts/simulate_dialogue.py CONVERSATIONS_FILE
    .venv/bin/python scripts/simulate_dialogue.py CONVERSATIONS_FILE \
        --base-url http://VM_IP:8788/v1

(System `python3` will fail with `ModuleNotFoundError: No module named 'pydantic'`.)

Or use the wrapper alongside this file:
    bash scripts/simulate.sh CONVERSATIONS_FILE

CONVERSATIONS_FILE: one user utterance per line. Lines starting with '#' or
empty lines are skipped.

Environment overrides
---------------------
    SA_BASE_URL  — SessionAgent vLLM base url (default http://127.0.0.1:8788/v1)
    SA_MODEL     — SessionAgent model name (default Qwen/Qwen3-4B-Instruct-2507-FP8)

Output
------
USER:    one line with the input utterance
RAW:     classifier JSON (one line)
ACTION:  TurnAction summary
BOT:     spoken text (deterministic for EmitClarify/Readback; placeholder for FireLLMFallback)
PROFILE: one-line state + non-null fields

Limitations
-----------
- FireLLMFallback is not actually invoked — it would require the main 30B model
  + RAG retrieval + KB index. The simulator only confirms the dispatcher routed
  TO LLM, not what the LLM would say.
- FireCalc / FireSMS are noted but the side-effects (calculator API call, SMS
  send) are skipped. The simulator advances state as if they succeeded.
- The SessionAgent system prompt is read directly from app.py at runtime
  (parsed via ast.literal_eval on each string-literal line — no `eval`) so
  there is one source of truth.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Make backend importable when invoked from repo root or from rag_demo_system.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.classifier_schema import parse_classifier_output  # noqa: E402
from backend.profile_prompts import (  # noqa: E402
    build_change_confirm_text,
    build_clarification_prompt,
    build_readback_text,
)
from backend.session import ClientProfile, ProfileState  # noqa: E402
from backend.turn_action import (  # noqa: E402
    EmitChangeConfirm,
    EmitClarify,
    EmitReadback,
    FireCalc,
    FireLLMFallback,
    FireOORMessage,
    FireSMS,
    Noop,
)
from backend.turn_dispatcher import apply_turn  # noqa: E402

DEFAULT_BASE_URL = os.environ.get("SA_BASE_URL", "http://127.0.0.1:8788/v1")
DEFAULT_MODEL = os.environ.get(
    "SA_MODEL", "Qwen/Qwen3-4B-Instruct-2507-FP8"
)


def _extract_session_agent_prompt() -> str:
    """Pull the SessionAgent system_prompt verbatim from app.py.

    Single source of truth: when the production prompt changes, the simulator
    automatically picks up the new text. Parses each string-literal line via
    ast.literal_eval — never eval/exec — so this is safe regardless of what
    app.py contains around the block.
    """
    app_py = (ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    open_marker = "system_prompt=("
    close_marker = "\n                ),"
    open_paren = app_py.find(open_marker)
    if open_paren < 0:
        raise RuntimeError("Could not locate 'system_prompt=(' in backend/app.py.")
    close = app_py.index(close_marker, open_paren)
    block = app_py[open_paren + len(open_marker): close]
    parts: list[str] = []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1]
        # ast.literal_eval accepts only Python literals (str, num, bool,
        # None, tuple, list, dict, set). Refuses anything that requires
        # code evaluation. Lines that aren't string literals get skipped.
        try:
            value = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, str):
            parts.append(value)
    if not parts:
        raise RuntimeError(
            "Parsed system_prompt block but found no string literals. "
            "Check whether the block format in app.py changed."
        )
    return "".join(parts)


SESSION_AGENT_SYSTEM_PROMPT = _extract_session_agent_prompt()


def _call_classifier(
    *,
    base_url: str,
    model: str,
    user_prompt: str,
    timeout: float = 8.0,
) -> dict[str, Any]:
    """POST chat/completions to the SessionAgent vLLM. Returns parsed dict.

    Mirrors the production call: temperature=0.0, max_tokens=160, JSON-only
    response. On HTTP error or non-JSON content, returns an empty dict so
    the dispatcher's classifier_schema fallback kicks in.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SESSION_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 160,
    }
    data = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
        outer = json.loads(body)
        text = outer["choices"][0]["message"]["content"]
        return _extract_first_json_object(text)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as e:
        print(f"  [classifier-error] {type(e).__name__}: {e}", file=sys.stderr)
        return {}


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """Find the first balanced {...} block in the string and json.loads it."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def _render_profile(profile: ClientProfile) -> str:
    """One-line snapshot of state + non-null calc-relevant fields."""
    parts = [f"state={profile.state.name}"]
    for field in (
        "client_type", "subject", "cost", "currency", "condition_new",
        "age_years", "term_months", "prepaid_pct", "type_schedule",
        "name", "last_offer",
    ):
        v = getattr(profile, field, None)
        if v not in (None, ""):
            parts.append(f"{field}={v}")
    return " | ".join(parts)


def _render_action(action: Any, profile: ClientProfile) -> str:
    """Convert a TurnAction to the bot's spoken text (deterministic where
    possible). LLM-fallback is left as a placeholder."""
    if isinstance(action, EmitReadback):
        return build_readback_text(profile)
    if isinstance(action, EmitClarify):
        return build_clarification_prompt(set(action.missing), profile)
    if isinstance(action, EmitChangeConfirm):
        return build_change_confirm_text({"changes": action.changes})
    if isinstance(action, FireCalc):
        return (
            f"[FireCalc] params={json.dumps(action.calc_params, ensure_ascii=False)}"
            " — calc API call skipped in simulation"
        )
    if isinstance(action, FireSMS):
        return "[FireSMS] SMS send skipped in simulation"
    if isinstance(action, FireOORMessage):
        return action.message
    if isinstance(action, FireLLMFallback):
        ctx = "with rag" if action.rag_context else "no rag"
        return f"[FireLLMFallback {ctx}] LLM call skipped in simulation"
    if isinstance(action, Noop):
        return f"[Noop reason={action.reason}]"
    return f"[{type(action).__name__}]"


def _build_user_prompt(transcript: list[dict[str, str]], message: str) -> str:
    """Mirror the production user_prompt construction (app.py line ~1148).

    Production also includes a `_tool_history` line; we omit it because the
    simulator does not actually fire tools. Empty tool history is benign for
    classifier behavior.
    """
    recent = transcript[-6:] if transcript else []
    if not recent:
        ctx = "начало разговора"
    else:
        lines = []
        for turn in recent:
            role = turn.get("role", "")
            text = turn.get("text", "") or ""
            if len(text) > 200:
                text = text[:200] + "…"
            speaker = "Клиент" if role == "user" else "Бот"
            lines.append(f"{speaker}: {text}")
        ctx = "\n".join(lines)
    return f"\n\nДиалог:\n{ctx}\n\nНОВОЕ сообщение: {message}"


def simulate(
    *,
    script: list[tuple[str, str]],
    base_url: str,
    model: str,
) -> None:
    """Run the simulation. `script` is a list of (kind, text) where kind is
    'user' (input utterance) or 'bot' (canned override of the bot's reply
    that gets injected into the transcript context for the NEXT turn).

    The 'bot' kind is the workaround for the FireLLMFallback / FireCalc
    actions that the simulator can't actually invoke. In production those
    paths produce real text; in the simulator we substitute a canned line
    so the next turn's classifier sees production-shaped context.
    """
    profile = ClientProfile(state=ProfileState.COLLECTING)
    transcript: list[dict[str, str]] = []
    session_id = str(uuid.uuid4())[:8]
    pending_bot_override: str | None = None
    print(f"=== Simulation session {session_id} ===")
    print(f"  Classifier: {model} @ {base_url}")
    print()

    user_turn_idx = 0
    for kind, text in script:
        if kind == "bot":
            # Defer until the NEXT user turn — at the end of that turn we
            # write this text into the transcript instead of the rendered
            # action stub. Multiple consecutive bot lines concatenate.
            if pending_bot_override is None:
                pending_bot_override = text
            else:
                pending_bot_override = f"{pending_bot_override} {text}"
            print(f"  [bot-override queued]: {text}")
            print()
            continue

        # kind == 'user'
        user_turn_idx += 1
        print(f"--- turn {user_turn_idx} ---")
        print(f"USER:    {text}")

        user_prompt = _build_user_prompt(transcript, text)
        raw = _call_classifier(
            base_url=base_url, model=model, user_prompt=user_prompt
        )
        print(f"RAW:     {json.dumps(raw, ensure_ascii=False)}")

        co = parse_classifier_output(json.dumps(raw), utterance=text)
        action = apply_turn(profile, co, text, turn_id=user_turn_idx)
        print(f"ACTION:  {type(action).__name__}")
        rendered = _render_action(action, profile)
        print(f"BOT:     {rendered}")
        print(f"PROFILE: {_render_profile(profile)}")
        print()

        transcript.append({"role": "user", "text": text})
        # If a 'bot:' line was queued before this turn was meant to follow
        # an LLM/calc/sms reply, write that canned text to the transcript
        # instead of the action stub. Otherwise use the rendered action.
        if pending_bot_override is not None and isinstance(action, (FireLLMFallback, FireCalc, FireSMS)):
            transcript.append({"role": "assistant", "text": pending_bot_override})
            pending_bot_override = None
        else:
            transcript.append({"role": "assistant", "text": rendered})
            # If a bot override was queued but the action wasn't stub-class,
            # the override is stale — drop it with a warning.
            if pending_bot_override is not None:
                print(
                    f"  [warn] pending bot override dropped — preceded a "
                    f"{type(action).__name__} action that produced concrete "
                    f"text already.",
                    file=sys.stderr,
                )
                pending_bot_override = None


def _read_conversation_file(path: Path) -> list[tuple[str, str]]:
    """Parse a conversation script.

    Lines starting with '#' or empty are skipped.
    Lines starting with 'BOT:' (case-insensitive) are canned bot
    replies that override the next stub action's transcript text.
    All other lines (with or without 'USER:' prefix) are user utterances.
    """
    out: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower.startswith("bot:"):
            out.append(("bot", line[4:].strip()))
        elif lower.startswith("user:"):
            out.append(("user", line[5:].strip()))
        else:
            out.append(("user", line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("conversation_file", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.conversation_file.exists():
        print(f"file not found: {args.conversation_file}", file=sys.stderr)
        return 1

    script = _read_conversation_file(args.conversation_file)
    if not any(kind == "user" for kind, _ in script):
        print("conversation file has no user utterances", file=sys.stderr)
        return 1

    simulate(
        script=script,
        base_url=args.base_url,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
