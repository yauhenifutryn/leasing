"""Chat session transcript persistence.

Mirrors backend.session_analyzer.save_transcript but writes per-turn
(chat has no end-of-session signal like a WebSocket disconnect).
Distinguished from voice transcripts by `transport="chat"` field.
Future account layer will populate `client_id`; today it is always None.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _transcripts_dir(state_dir: Path) -> Path:
    out = state_dir / "transcripts"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_transcript_path(state_dir: Path, session_id: str) -> Path:
    """Build the transcript file path and assert it cannot escape
    `state_dir/transcripts/`. Defense in depth: the route already
    validates session_id, but this guard prevents any future caller
    from accidentally introducing a traversal.

    Codex adversarial review (2026-05-03): a session_id like
    "../sessions" would resolve to .state/sessions.json and overwrite
    the StateStore. Resolving the candidate path and asserting
    `relative_to(transcripts_root)` makes that physically impossible.
    """
    transcripts_root = (state_dir / "transcripts").resolve()
    transcripts_root.mkdir(parents=True, exist_ok=True)
    candidate = (state_dir / "transcripts" / f"{session_id}.json").resolve()
    try:
        candidate.relative_to(transcripts_root)
    except ValueError as exc:
        raise ValueError(
            f"refusing to write outside transcripts dir: session_id={session_id!r}"
        ) from exc
    return candidate


def save_chat_turn(
    *,
    session_id: str,
    transcript: list[dict[str, str]],
    tool_calls: list[dict[str, Any]],
    name: str,
    phone: str,
    state_dir: Path,
) -> Path:
    """Write the current chat session state to disk. Idempotent overwrite.

    `started_at` is preserved across calls; `last_turn_at` is updated
    on every save; `ended_at` stays None until `mark_session_ended`.
    """
    out_path = _safe_transcript_path(state_dir, session_id)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    started_at = now_iso
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            started_at = prior.get("started_at") or now_iso
        except Exception:  # noqa: BLE001
            pass
    record: dict[str, Any] = {
        "session_id": session_id,
        "transport": "chat",
        "client_id": None,  # future: populated by account layer
        "name": name or "",
        "phone": phone or "",
        "started_at": started_at,
        "last_turn_at": now_iso,
        "ended_at": None,
        "turn_count": len(transcript),
        "transcript": transcript,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
    }
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def mark_session_ended(session_id: str, *, state_dir: Path) -> None:
    """Stamp `ended_at` on a saved chat record. Triggered by EndCall action."""
    out_path = _safe_transcript_path(state_dir, session_id)
    if not out_path.exists():
        return
    try:
        record = json.loads(out_path.read_text(encoding="utf-8"))
        record["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
