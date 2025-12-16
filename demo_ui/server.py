#!/usr/bin/env python3
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATE_DIR = Path(__file__).resolve().parent / ".state"

STATE_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def safe_join(root: Path, rel: str) -> Path:
    candidate = (root / rel).resolve()
    if not str(candidate).startswith(str(root.resolve())):
        raise ValueError("Path escapes root")
    return candidate


ALLOWED_READ_DIRS = {
    "insights_per_call": REPO_ROOT / "insights_per_call",
    "insights_global": REPO_ROOT / "insights_global",
    "knowledge_base": REPO_ROOT / "knowledge_base",
    "nlu_output": REPO_ROOT / "nlu_output",
    "transcripts_clean": REPO_ROOT / "transcripts_clean",
}


WHITELIST_TASKS: dict[str, list[str]] = {
    "check": ["make", "check"],
    "analyze_calls": ["make", "analyze-calls"],
    "nlu_export": ["make", "nlu-export"],
    "rollup": ["make", "rollup"],
    "aggregate": ["make", "aggregate"],
    "dedup": ["make", "dedup"],
    "kb": ["make", "kb"],
    "kb_markdown": ["make", "kb-markdown"],
    # Optional: can be slow locally; keep behind explicit click in UI
    "transcribe": ["make", "transcribe"],
    # Optional: start streamlit review UI
    "review_ui": [sys.executable, "-m", "streamlit", "run", "scripts/review_app.py", "--server.port", "8501"],
}


@dataclass
class Session:
    id: str
    created_at: str
    name: str
    notes: str = ""


@dataclass
class Run:
    id: str
    session_id: str
    task: str
    command: list[str]
    status: str  # queued|running|success|failed|stopped
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str | None = None


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sessions_path = root / "sessions.json"
        self.runs_path = root / "runs.jsonl"
        self.feedback_path = root / "feedback.jsonl"
        self._lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.runs: dict[str, Run] = {}
        self._load()

    def _load(self) -> None:
        if self.sessions_path.exists():
            payload = json.loads(self.sessions_path.read_text(encoding="utf-8"))
            for item in payload:
                sess = Session(**item)
                self.sessions[sess.id] = sess
        if self.runs_path.exists():
            for line in self.runs_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                run = Run(**item)
                self.runs[run.id] = run

    def _persist_sessions(self) -> None:
        self.sessions_path.write_text(json.dumps([asdict(s) for s in self.sessions.values()], ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_run(self, run: Run) -> None:
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(run), ensure_ascii=False))
            handle.write("\n")

    def create_session(self, name: str) -> Session:
        with self._lock:
            sess = Session(id=str(uuid.uuid4()), created_at=now_iso(), name=name)
            self.sessions[sess.id] = sess
            self._persist_sessions()
            return sess

    def list_sessions(self) -> list[Session]:
        with self._lock:
            return sorted(self.sessions.values(), key=lambda s: s.created_at, reverse=True)

    def create_run(self, session_id: str, task: str, command: list[str]) -> Run:
        log_dir = self.root / "logs" / session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        run_id = str(uuid.uuid4())
        log_path = log_dir / f"{run_id}.log"
        run = Run(
            id=run_id,
            session_id=session_id,
            task=task,
            command=command,
            status="queued",
            created_at=now_iso(),
            log_path=str(log_path),
        )
        with self._lock:
            self.runs[run.id] = run
            self._append_run(run)
        return run

    def update_run(self, run: Run) -> None:
        with self._lock:
            self.runs[run.id] = run
            self._append_run(run)

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            return self.runs.get(run_id)

    def list_runs(self, session_id: str | None = None, limit: int = 50) -> list[Run]:
        with self._lock:
            runs = list(self.runs.values())
        if session_id:
            runs = [r for r in runs if r.session_id == session_id]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    def add_feedback(self, session_id: str | None, payload: dict[str, Any]) -> None:
        record = {
            "ts": now_iso(),
            "session_id": session_id,
            **payload,
        }
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class Runner:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def enqueue(self, run_id: str) -> None:
        self.queue.put(run_id)

    def stop(self, run_id: str) -> bool:
        with self._lock:
            proc = self.processes.get(run_id)
        if proc is None:
            return False
        try:
            proc.terminate()
            return True
        except Exception:
            return False

    def _worker(self) -> None:
        while True:
            run_id = self.queue.get()
            run = self.store.get_run(run_id)
            if run is None:
                continue
            run.status = "running"
            run.started_at = now_iso()
            self.store.update_run(run)

            log_path = Path(run.log_path) if run.log_path else (STATE_DIR / "logs" / f"{run.id}.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"[{now_iso()}] START {run.task}: {' '.join(run.command)}\n")
                log.flush()
                proc = subprocess.Popen(
                    run.command,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env={**os.environ},
                )
                with self._lock:
                    self.processes[run.id] = proc
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                finally:
                    return_code = proc.wait()
                    with self._lock:
                        self.processes.pop(run.id, None)

            run.return_code = return_code
            run.finished_at = now_iso()
            if return_code == 0:
                run.status = "success"
            elif return_code == -15:
                run.status = "stopped"
            else:
                run.status = "failed"
            self.store.update_run(run)


STORE = StateStore(STATE_DIR)
RUNNER = Runner(STORE)


class Handler(BaseHTTPRequestHandler):
    server_version = "LeasingDemoUI/0.1"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, "application/json; charset=utf-8", json_bytes(payload))

    def _bad(self, message: str, status: int = 400) -> None:
        self._json(status, {"error": message})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = (STATIC_DIR / "index.html").read_bytes()
            return self._send(200, "text/html; charset=utf-8", body)
        if parsed.path.startswith("/static/"):
            rel = parsed.path.removeprefix("/static/")
            fp = safe_join(STATIC_DIR, rel)
            if not fp.exists() or fp.is_dir():
                return self._send(404, "text/plain; charset=utf-8", b"Not found")
            ctype = "application/octet-stream"
            if fp.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif fp.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif fp.suffix == ".svg":
                ctype = "image/svg+xml; charset=utf-8"
            return self._send(200, ctype, fp.read_bytes())

        if parsed.path == "/api/health":
            return self._json(200, {"ok": True, "ts": now_iso()})
        if parsed.path == "/api/sessions":
            sessions = [asdict(s) for s in STORE.list_sessions()]
            return self._json(200, {"sessions": sessions})
        if parsed.path == "/api/runs":
            qs = parse_qs(parsed.query)
            session_id = (qs.get("session_id") or [None])[0]
            runs = [asdict(r) for r in STORE.list_runs(session_id=session_id)]
            return self._json(200, {"runs": runs})
        if parsed.path == "/api/tasks":
            return self._json(200, {"tasks": sorted(WHITELIST_TASKS.keys())})
        if parsed.path == "/api/env":
            keys = [
                "OPENAI_MODEL",
                "REVIEW_OPENAI_MODEL",
                "OPENAI_API_KEY",
                "HUGGINGFACE_TOKEN",
            ]
            env = {k: ("SET" if os.getenv(k) else "MISSING") for k in keys}
            return self._json(200, {"env": env})
        if parsed.path == "/api/log":
            qs = parse_qs(parsed.query)
            run_id = (qs.get("run_id") or [None])[0]
            if not run_id:
                return self._bad("run_id required")
            run = STORE.get_run(run_id)
            if run is None or not run.log_path:
                return self._bad("unknown run_id", 404)
            fp = Path(run.log_path)
            if not fp.exists():
                return self._json(200, {"lines": []})
            limit = int((qs.get("limit") or ["300"])[0])
            lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
            return self._json(200, {"lines": lines})
        if parsed.path == "/api/files":
            qs = parse_qs(parsed.query)
            kind = (qs.get("kind") or [None])[0]
            if kind not in ALLOWED_READ_DIRS:
                return self._bad("invalid kind", 400)
            root = ALLOWED_READ_DIRS[kind]
            if not root.exists():
                return self._json(200, {"files": []})
            files = sorted([p.name for p in root.glob("*.json")])[:1000]
            return self._json(200, {"files": files})
        if parsed.path == "/api/file":
            qs = parse_qs(parsed.query)
            kind = (qs.get("kind") or [None])[0]
            name = (qs.get("name") or [None])[0]
            if kind not in ALLOWED_READ_DIRS or not name:
                return self._bad("kind and name required")
            root = ALLOWED_READ_DIRS[kind]
            fp = safe_join(root, name)
            if not fp.exists():
                return self._bad("not found", 404)
            if fp.suffix != ".json":
                return self._bad("only .json allowed", 400)
            text = fp.read_text(encoding="utf-8", errors="ignore")
            return self._json(200, {"name": name, "text": text})

        return self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except Exception:
            return self._bad("invalid json")

        if parsed.path == "/api/session":
            name = (payload.get("name") or "").strip() or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            sess = STORE.create_session(name=name)
            return self._json(200, {"session": asdict(sess)})

        if parsed.path == "/api/run":
            session_id = payload.get("session_id")
            task = payload.get("task")
            if not session_id or session_id not in STORE.sessions:
                return self._bad("invalid session_id")
            if task not in WHITELIST_TASKS:
                return self._bad("invalid task")
            command = list(WHITELIST_TASKS[task])
            if command and command[0] == "make":
                command = ["make", f"PY={sys.executable}", *command[1:]]
            run = STORE.create_run(session_id=session_id, task=task, command=command)
            RUNNER.enqueue(run.id)
            return self._json(200, {"run": asdict(run)})

        if parsed.path == "/api/stop":
            run_id = payload.get("run_id")
            if not run_id:
                return self._bad("run_id required")
            ok = RUNNER.stop(run_id)
            return self._json(200, {"ok": ok})

        if parsed.path == "/api/feedback":
            session_id = payload.get("session_id")
            message = (payload.get("message") or "").strip()
            if not message:
                return self._bad("message required")
            STORE.add_feedback(session_id=session_id, payload={"message": message})
            return self._json(200, {"ok": True})

        return self._send(404, "text/plain; charset=utf-8", b"Not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("DEMO_UI_PORT", "8787")))
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[demo_ui] Serving on http://{args.host}:{args.port} (repo: {REPO_ROOT})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
