from collections import defaultdict


def _to_md_list(items, label):
    if not items:
        return ""
    lines = [f"### {label}", ""]
    for item in items:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _render_entry(entry):
    intent = entry.get("intent", "").strip()
    question = entry.get("canonical_question", "").strip()
    answer = entry.get("best_answer", "").strip()
    title = intent or question or "Без названия"
    lines = [f"## {title}"]
    if question:
        lines.append(f"**Вопрос:** {question}")
    if answer:
        lines.extend(["", "**Ответ:**", "", answer])
    lines.append("")
    lines.append(_to_md_list(entry.get("eligibility_rules"), "Условия"))
    lines.append(_to_md_list(entry.get("required_fields"), "Необходимые данные"))
    lines.append(_to_md_list(entry.get("compliance_notes"), "Комплаенс / ограничения"))
    lines.append(_to_md_list(entry.get("handoff_when"), "Эскалация / передать специалисту"))
    lines.append(_to_md_list(entry.get("empathy_patterns"), "Эмпатия"))
    lines.append(_to_md_list(entry.get("followups"), "Доп. вопросы"))
    return "\n".join(lines).strip()


def render_flat_markdown(entries):
    lines = ["# Knowledge Base", ""]
    for entry in entries:
        lines.append(_render_entry(entry))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_structured_markdown(entries):
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        category = entry.get("category") or "Общее"
        subtopic = entry.get("subtopic") or "Разное"
        grouped[category][subtopic].append(entry)

    lines = ["# Knowledge Base (Structured)", ""]
    for category in sorted(grouped.keys()):
        lines.append(f"## {category}")
        lines.append("")
        subtopics = grouped[category]
        for subtopic in sorted(subtopics.keys()):
            lines.append(f"### {subtopic}")
            lines.append("")
            for entry in subtopics[subtopic]:
                question = entry.get("canonical_question", "").strip()
                answer = entry.get("best_answer", "").strip()
                if question:
                    lines.append(f"**Вопрос:** {question}")
                if answer:
                    lines.extend(["", "**Ответ:**", "", answer])
                lines.append("")
                lines.append(_to_md_list(entry.get("keywords"), "Ключевые слова"))
                lines.append(_to_md_list(entry.get("tags"), "Теги"))
                lines.append(_to_md_list(entry.get("references"), "Ссылки"))
            lines.append("")
    return "\n".join(lines).strip() + "\n"
