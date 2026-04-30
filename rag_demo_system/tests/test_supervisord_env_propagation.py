"""Regression: supervisord must wrap `. .env` with `set -a` / `set +a`
so the sourced KEY=value lines are exported to the program subprocess.

Symptom (live evidence 2026-04-29): SILERO_TTS_RATE_PCT=120 in .env
became a wrapper-shell variable but never reached the silero_tts
subprocess; os.getenv read None and the service ran at 100%, byte-
identical to the default. Same trap applies to every .env entry whose
default differs from the value we want at runtime.

Fix: prepend `set -a` and append `set +a` around the dotfile source in
each program block. This test asserts the wrap is present in all five
program blocks (backend, qwen, sessionagent, whisper, silero_tts), so
any future supervisord.conf edit that drops the wrap fails CI before
the regression reaches a deployed VM.
"""
from pathlib import Path
import re

SUPERVISORD_CONF = (
    Path(__file__).resolve().parents[1] / "scripts" / "supervisord.conf"
)
PROGRAM_BLOCKS = ("backend", "qwen", "sessionagent", "whisper", "silero_tts")


def _command_for(program: str, conf_text: str) -> str:
    """Return the `command=...` line of `[program:NAME]` block."""
    in_block = False
    header = f"[program:{program}]"
    for line in conf_text.splitlines():
        stripped = line.strip()
        if stripped == header:
            in_block = True
            continue
        if in_block:
            if stripped.startswith("[") and stripped.endswith("]"):
                break
            if line.startswith("command="):
                return line[len("command="):]
    raise AssertionError(
        f"program block {header} not found or has no command= line"
    )


def test_each_program_wraps_env_source_with_set_a():
    conf = SUPERVISORD_CONF.read_text(encoding="utf-8")
    for program in PROGRAM_BLOCKS:
        cmd = _command_for(program, conf)
        # Both pieces must be present and ordered: set -a → source → set +a.
        assert "set -a" in cmd, f"[{program}] missing `set -a` before .env source"
        assert "set +a" in cmd, f"[{program}] missing `set +a` after .env source"
        i_on = cmd.index("set -a")
        i_src = cmd.index(".env")
        i_off = cmd.index("set +a")
        assert i_on < i_src < i_off, (
            f"[{program}] order wrong: expected `set -a` < .env < `set +a`, "
            f"got positions {i_on}, {i_src}, {i_off} in:\n  {cmd}"
        )
