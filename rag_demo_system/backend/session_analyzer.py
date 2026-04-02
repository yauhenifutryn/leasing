"""Post-session analysis: runs after each voice call to detect quality issues.

Saves per-session reports to .state/analysis/ as JSONL.
Uses the same LLM to analyze the transcript with a dedicated analysis prompt.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


_ANALYSIS_PROMPT = """Ты аналитик качества голосового бота компании «Микро Лизинг».
Проанализируй транскрипт разговора и верни JSON-отчет.

Оцени по каждому критерию (0-10, где 10 = отлично):

1. banned_phrases: использовал ли бот запрещенные фразы ("к сожалению", "понимаю вашу ситуацию", "в базе знаний нет")? Перечисли найденные.
2. specialist_overuse: предлагал ли бот специалиста после ответов, где уже дал информацию? Сколько раз из скольки ответов?
3. hallucinations: называл ли бот конкретные адреса, числа, ставки, которых нет в предоставленном контексте? Перечисли подозрительные.
4. name_usage: сколько раз бот обращался к клиенту по имени? Оптимально 2-3 раза за разговор.
5. humor_and_tone: был ли бот живым и дружелюбным или сухим и формальным? Были ли уместные шутки?
6. answer_completeness: отвечал ли бот по существу или уходил от ответа?
7. kb_gaps: какие вопросы клиента остались без ответа из-за отсутствия данных в базе знаний? Перечисли темы.
8. response_variety: были ли ответы структурно разнообразными или однотипными?
9. client_satisfaction_signals: были ли сигналы раздражения или удовлетворенности клиента?

Верни строго JSON:
{
  "scores": {"banned_phrases": N, "specialist_overuse": N, "humor_and_tone": N, "answer_completeness": N, "response_variety": N},
  "issues": [{"type": "тип", "severity": "critical|important|minor", "detail": "описание", "suggested_fix": "предложение"}],
  "kb_gaps": ["тема1", "тема2"],
  "banned_phrases_found": ["фраза1"],
  "specialist_count": {"offered": N, "appropriate": N},
  "name_count": N,
  "overall_score": N,
  "summary": "одно предложение"
}"""


def analyze_session(
    transcript: list[dict[str, str]],
    llm_call_fn: Any,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    """Analyze a completed session transcript and return a quality report."""
    if not transcript or len(transcript) < 4:
        return {"skipped": True, "reason": "transcript_too_short"}

    formatted = []
    for turn in transcript:
        role = "Клиент" if turn.get("role") == "user" else "Бот"
        formatted.append(f"{role}: {turn.get('text', '')}")
    transcript_text = "\n".join(formatted)

    try:
        resp = llm_call_fn(
            base_url=base_url,
            model=model,
            system_prompt=_ANALYSIS_PROMPT,
            user_prompt=f"Транскрипт разговора:\n\n{transcript_text}\n\nВерни JSON-отчет.",
            temperature=0.1,
            max_tokens=800,
            timeout_sec=30,
        )
        # Extract JSON from response
        text = resp.text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            report = json.loads(text[start:end])
            report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            report["turn_count"] = len(transcript)
            return report
    except Exception as exc:
        return {"error": str(exc), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    return {"error": "no_json_in_response", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def save_report(report: dict[str, Any], state_dir: Path) -> Path:
    """Append report to the analysis log file."""
    analysis_dir = state_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    log_path = analysis_dir / "session_reports.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False))
        f.write("\n")
    return log_path
