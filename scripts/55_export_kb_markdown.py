import argparse
import json
from pathlib import Path

from kb_export import render_flat_markdown, render_structured_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="knowledge_base/kb_faq_ru.json")
    parser.add_argument("--out", dest="out_path", default="knowledge_base/kb_faq_ru.md")
    parser.add_argument(
        "--out-structured",
        dest="out_structured_path",
        default="knowledge_base/kb_faq_ru_structured.md",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    flat_md = render_flat_markdown(data)
    Path(args.out_path).write_text(flat_md, encoding="utf-8")
    structured_md = render_structured_markdown(data)
    Path(args.out_structured_path).write_text(structured_md, encoding="utf-8")


if __name__ == "__main__":
    main()
