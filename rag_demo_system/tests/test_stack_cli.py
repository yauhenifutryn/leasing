from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_module():
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("scripts.stack_cli")
    assert spec is not None, "scripts.stack_cli module is missing"
    return importlib.import_module("scripts.stack_cli")


def test_resolve_stack_mode_prefers_docker_when_available() -> None:
    stack_cli = _load_module()

    mode = stack_cli.resolve_stack_mode(requested="docker", docker_available=True)

    assert mode == "docker"


def test_resolve_stack_mode_falls_back_to_supervisor_without_docker() -> None:
    stack_cli = _load_module()

    mode = stack_cli.resolve_stack_mode(requested="docker", docker_available=False)

    assert mode == "supervisor"


def test_build_up_commands_includes_dify_compose_when_configured() -> None:
    stack_cli = _load_module()

    commands = stack_cli.build_up_commands(
        repo_root="/workspace/leasing",
        mode="docker",
        dify_dir="/opt/dify/docker",
    )

    assert commands[0] == [
        "docker",
        "compose",
        "-f",
        "/workspace/leasing/rag_demo_system/docker-compose.yml",
        "up",
        "-d",
    ]
    assert commands[1] == [
        "docker",
        "compose",
        "up",
        "-d",
    ]
    assert commands[2][0] == "supervisord"
