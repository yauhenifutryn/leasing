from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

VOICE_PROFILES = {"local", "yandex_speechkit", "oss_russian", "yandex_realtime"}
OPTIONAL_PROGRAM_ENV = {
    "qwen": "STACK_QWEN_CMD",
    "sensevoice": "STACK_SENSEVOICE_CMD",
    "whisper": "STACK_WHISPER_CMD",
    "cosyvoice": "STACK_COSYVOICE_CMD",
    "vosk": "STACK_VOSK_CMD",
    "vosk_tts": "STACK_VOSK_TTS_CMD",
    "ngrok": "STACK_NGROK_CMD",
}
OPTIONAL_PROGRAMS = sorted([*OPTIONAL_PROGRAM_ENV.keys(), "frontend"])


def resolve_stack_mode(requested: str | None, docker_available: bool) -> str:
    mode = (requested or "docker").strip().lower()
    if mode == "docker" and docker_available:
        return "docker"
    return "supervisor"


def resolve_voice_profile(requested: str | None) -> str:
    profile = (requested or "local").strip().lower()
    if profile in VOICE_PROFILES:
        return profile
    return "local"


def _env_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def build_program_selection(voice_profile: str, env: dict[str, str] | None = None) -> dict[str, list[str]]:
    env_map = env or dict(os.environ)
    start = {"backend"}
    if env_map.get("STACK_QWEN_CMD", "").strip():
        start.add("qwen")
    if _env_enabled(env_map.get("STACK_START_FRONTEND")):
        start.add("frontend")
    if env_map.get("STACK_NGROK_CMD", "").strip():
        start.add("ngrok")
    if voice_profile == "local":
        if env_map.get("STACK_SENSEVOICE_CMD", "").strip():
            start.add("sensevoice")
        if env_map.get("STACK_COSYVOICE_CMD", "").strip():
            start.add("cosyvoice")
        if env_map.get("STACK_WHISPER_CMD", "").strip():
            start.add("whisper")
    elif voice_profile == "oss_russian":
        if env_map.get("STACK_VOSK_CMD", "").strip():
            start.add("vosk")
        if env_map.get("STACK_VOSK_TTS_CMD", "").strip():
            start.add("vosk_tts")
    stop = sorted(program for program in OPTIONAL_PROGRAMS if program not in start)
    ordered_start = [
        name
        for name in ["backend", "qwen", "sensevoice", "whisper", "cosyvoice", "vosk", "vosk_tts", "frontend", "ngrok"]
        if name in start
    ]
    return {"start": ordered_start, "stop": stop}


def build_up_commands(repo_root: str, mode: str, dify_dir: str | None = None) -> list[list[str]]:
    repo_root_path = Path(repo_root)
    commands: list[list[str]] = []
    if mode == "docker":
        commands.append(
            [
                "docker",
                "compose",
                "-f",
                str(repo_root_path / "rag_demo_system" / "docker-compose.yml"),
                "up",
                "-d",
            ]
        )
        if dify_dir:
            commands.append(["docker", "compose", "up", "-d"])
    return commands


def _supervisor_conf(repo_root: Path) -> str:
    return str(repo_root / "rag_demo_system" / "scripts" / "supervisord.conf")


def _resolve_binary(repo_root: Path, env_var: str, venv_relative: str, fallback_name: str) -> str:
    explicit = os.getenv(env_var, "").strip()
    if explicit:
        return explicit
    venv_candidate = repo_root / "rag_demo_system" / venv_relative
    if venv_candidate.exists():
        return str(venv_candidate)
    found = shutil.which(fallback_name)
    if found:
        return found
    raise RuntimeError(f"{fallback_name} is not available. Set {env_var} or run the setup script first.")


def _supervisord_bin(repo_root: Path) -> str:
    return _resolve_binary(repo_root, "STACK_SUPERVISORD_BIN", ".venv/bin/supervisord", "supervisord")


def _supervisorctl_bin(repo_root: Path) -> str:
    return _resolve_binary(repo_root, "STACK_SUPERVISORCTL_BIN", ".venv/bin/supervisorctl", "supervisorctl")


def _supervisorctl(repo_root: Path, *args: str, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_supervisorctl_bin(repo_root), "-c", _supervisor_conf(repo_root), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _check_gpu_memory_leak() -> None:
    """Abort early if GPU memory is leaked (used but no processes)."""
    if not shutil.which("nvidia-smi"):
        return
    try:
        used = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split("\n")[0].strip()
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        used_mib = int(used) if used else 0
        proc_count = len([l for l in procs.splitlines() if l.strip()])
        if used_mib > 5000 and proc_count == 0:
            print(
                f"ERROR: Leaked GPU memory ({used_mib}MiB used, 0 processes).\n"
                "Cannot start vLLM. Restart the instance from your provider's dashboard,\n"
                "then re-run this command.",
                file=sys.stderr,
            )
            raise SystemExit(1)
    except (ValueError, FileNotFoundError):
        pass


def ensure_supervisor_running(repo_root: Path) -> None:
    state_dir = repo_root / "rag_demo_system" / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    status = _supervisorctl(repo_root, "status", check=False)
    if status.returncode == 0:
        return
    _check_gpu_memory_leak()
    pidfile = state_dir / "supervisord.pid"
    pidfile.unlink(missing_ok=True)
    subprocess.run([_supervisord_bin(repo_root), "-c", _supervisor_conf(repo_root)], check=True)


def sync_supervisor_programs(repo_root: Path, selection: dict[str, list[str]]) -> None:
    ensure_supervisor_running(repo_root)
    for program in selection["stop"]:
        _supervisorctl(repo_root, "stop", program, check=False)
    for program in selection["start"]:
        result = _supervisorctl(repo_root, "start", program, check=False)
        if result.returncode != 0 and "already started" not in (result.stdout + result.stderr):
            raise RuntimeError(f"failed to start supervisor program '{program}': {(result.stdout + result.stderr).strip()}")


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    command = args[0] if args else "status"
    repo_root = Path(__file__).resolve().parents[2]
    requested_mode = os.getenv("STACK_MODE", "docker")
    mode = resolve_stack_mode(requested_mode, docker_available=shutil.which("docker") is not None)
    dify_dir = os.getenv("DIFY_DOCKER_DIR")
    voice_profile = resolve_voice_profile(os.getenv("STACK_VOICE_PROFILE"))
    selection = build_program_selection(voice_profile=voice_profile)

    if command == "status":
        print(f"mode={mode}")
        print(f"repo_root={repo_root}")
        print(f"voice_profile={voice_profile}")
        print(f"start={','.join(selection['start'])}")
        print(f"stop={','.join(selection['stop'])}")
        return 0
    if command == "up":
        for cmd in build_up_commands(str(repo_root), mode, dify_dir=dify_dir):
            kwargs = {"check": True}
            if cmd[:2] == ["docker", "compose"] and len(cmd) == 4 and dify_dir:
                kwargs["cwd"] = dify_dir
            subprocess.run(cmd, **kwargs)
        sync_supervisor_programs(repo_root, selection)
        return 0
    if command == "down":
        if mode == "docker":
            subprocess.run(
                ["docker", "compose", "-f", str(repo_root / "rag_demo_system" / "docker-compose.yml"), "down"],
                check=False,
            )
            if dify_dir:
                subprocess.run(["docker", "compose", "down"], cwd=dify_dir, check=False)
        subprocess.run([_supervisorctl_bin(repo_root), "-c", _supervisor_conf(repo_root), "shutdown"], check=False)
        return 0
    if command == "smoke":
        subprocess.run([str(repo_root / "rag_demo_system" / "scripts" / "smoke_test.sh")], check=True)
        return 0
    if command == "benchmark":
        print("benchmark runner not implemented yet")
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
