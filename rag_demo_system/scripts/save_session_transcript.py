#!/usr/bin/env python3
"""Save individual session transcripts from the state store.

Run after calls to export each session as a separate JSON file.
Idempotent: skips sessions already exported.

Usage:
    python scripts/save_session_transcript.py
    python scripts/save_session_transcript.py --state-dir .state --out-dir .state/transcripts
"""
import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=".state", help="State directory")
    parser.add_argument("--out-dir", default=".state/transcripts", help="Output directory")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions_path = state_dir / "sessions.json"
    if not sessions_path.exists():
        print("No sessions.json found")
        return

    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    exported = 0
    skipped = 0

    for sess in sessions:
        sid = sess.get("session_id", "")
        transcript = sess.get("transcript", [])
        if not sid or not transcript:
            continue

        out_path = out_dir / f"{sid}.json"
        if out_path.exists():
            skipped += 1
            continue

        record = {
            "session_id": sid,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "turn_count": len(transcript),
            "transcript": transcript,
            "metadata": sess.get("metadata", {}),
        }
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        exported += 1

    print(f"Exported {exported} sessions, skipped {skipped} (already exist)")


if __name__ == "__main__":
    main()
