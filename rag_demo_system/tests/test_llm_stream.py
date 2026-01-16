from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.llm_stream import iter_openai_stream_text


def test_iter_openai_stream_text_yields_content() -> None:
    lines = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}",
        "data: [DONE]",
    ]
    out = "".join(iter_openai_stream_text(lines))
    assert out == "Hello"
