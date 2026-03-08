from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stack_shell_sources_env_file() -> None:
    content = (ROOT / "scripts" / "stack.sh").read_text(encoding="utf-8")

    assert 'if [ -f "$ROOT_DIR/rag_demo_system/.env" ]; then' in content
    assert '. "$ROOT_DIR/rag_demo_system/.env"' in content
