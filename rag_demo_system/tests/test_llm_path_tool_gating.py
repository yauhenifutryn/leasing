"""Contract test: the RAG branch in _stream_voice_response must reset tool_schemas=[]
so the LLM cannot autonomously re-invoke calculator on non-tool turns."""
from pathlib import Path


_APP_PY = Path(__file__).resolve().parents[1] / "backend" / "app.py"


def test_rag_branch_clears_tool_schemas():
    src = _APP_PY.read_text(encoding="utf-8")
    # Locate the RAG branch (# RAG path: full context for KB questions)
    marker = "# RAG path: full context for KB questions"
    assert marker in src, "RAG branch marker missing - refactor broke this contract test"
    # Between the marker and the next `llm_messages = [` the branch body lives.
    idx_marker = src.index(marker)
    idx_next_llm_messages = src.index("llm_messages", idx_marker)
    rag_body = src[idx_marker:idx_next_llm_messages]
    assert "tool_schemas = []" in rag_body, (
        "Fix 13 regression: RAG branch no longer resets tool_schemas=[] - the LLM "
        "will re-invoke calculator on non-tool turns."
    )
