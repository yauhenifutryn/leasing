from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def resolve_stack_mode(requested: str | None, docker_available: bool) -> str:
    mode = (requested or "docker").strip().lower()
    if mode == "docker" and docker_available:
        return "docker"
    return "supervisor"


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
    commands.append(
        [
            "supervisord",
            "-c",
            str(repo_root_path / "rag_demo_system" / "scripts" / "supervisord.conf"),
        ]
    )
    return commands


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    command = args[0] if args else "status"
    repo_root = Path(__file__).resolve().parents[2]
    requested_mode = os.getenv("STACK_MODE", "docker")
    mode = resolve_stack_mode(requested_mode, docker_available=shutil.which("docker") is not None)
    dify_dir = os.getenv("DIFY_DOCKER_DIR")

    if command == "status":
        print(f"mode={mode}")
        print(f"repo_root={repo_root}")
        return 0
    if command == "up":
        for cmd in build_up_commands(str(repo_root), mode, dify_dir=dify_dir):
            kwargs = {"check": True}
            if cmd[:2] == ["docker", "compose"] and len(cmd) == 4 and dify_dir:
                kwargs["cwd"] = dify_dir
            subprocess.run(cmd, **kwargs)
        return 0
    if command == "down":
        if mode == "docker":
            subprocess.run(
                ["docker", "compose", "-f", str(repo_root / "rag_demo_system" / "docker-compose.yml"), "down"],
                check=False,
            )
            if dify_dir:
                subprocess.run(["docker", "compose", "down"], cwd=dify_dir, check=False)
        subprocess.run(["supervisorctl", "-c", str(repo_root / "rag_demo_system" / "scripts" / "supervisord.conf"), "shutdown"], check=False)
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
