from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionState:
    session_id: str
    consent_given: bool = False
    consent_denied: bool = False
    transcript: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_path = root / "sessions.json"
        self.logs_path = root / "logs.jsonl"
        self._sessions: dict[str, SessionState] = {}
        self._load()

    def _load(self) -> None:
        if not self.sessions_path.exists():
            return
        payload = json.loads(self.sessions_path.read_text(encoding="utf-8"))
        for item in payload:
            sess = SessionState(
                session_id=item.get("session_id", ""),
                consent_given=bool(item.get("consent_given", False)),
                consent_denied=bool(item.get("consent_denied", False)),
                transcript=item.get("transcript") or [],
                metadata=item.get("metadata") or {},
            )
            if sess.session_id:
                self._sessions[sess.session_id] = sess

    def _persist(self) -> None:
        data = [asdict(s) for s in self._sessions.values()]
        self.sessions_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def create(self, session_id: str) -> SessionState:
        sess = SessionState(session_id=session_id)
        self._sessions[session_id] = sess
        self._persist()
        return sess

    def update(self, session: SessionState) -> None:
        self._sessions[session.session_id] = session
        self._persist()

    def log(self, event: dict[str, Any]) -> None:
        with self.logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False))
            handle.write("\n")

    def tail_logs(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.logs_path.exists():
            return []
        lines = self.logs_path.read_text(encoding="utf-8").splitlines()[-limit:]
        out: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out
