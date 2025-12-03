import argparse
import json
from pathlib import Path


def to_md_list(items, label):
    """Render a Markdown list section."""
    if not items:
        return ""
    lines = [f"### {label}", ""]
    for item in items:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="knowledge_base/kb_faq_ru.json")
    parser.add_argument("--out", dest="out_path", default="knowledge_base/kb_faq_ru.md")
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    lines = ["# Knowledge Base", ""]

    for entry in data:
        intent = entry.get("intent", "").strip()
        question = entry.get("canonical_question", "").strip()
        answer = entry.get("best_answer", "").strip()
        lines.append(f"## {intent or question}")
        if question:
            lines.append(f"**Вопрос:** {question}")
        if answer:
            lines.append("")
            lines.append("**Ответ:**")
            lines.append("")
            lines.append(answer)
        lines.append("")

        lines.append(to_md_list(entry.get("eligibility_rules"), "Условия"))
        lines.append(to_md_list(entry.get("required_fields"), "Необходимые данные"))
        lines.append(to_md_list(entry.get("compliance_notes"), "Комплаенс / ограничения"))
        lines.append(to_md_list(entry.get("handoff_when"), "Эскалация / передать специалисту"))
        lines.append(to_md_list(entry.get("empathy_patterns"), "Эмпатия"))
        lines.append(to_md_list(entry.get("followups"), "Доп. вопросы"))

    Path(args.out_path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
