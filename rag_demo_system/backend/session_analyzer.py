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
10. tool_use_quality: если бот использовал инструменты (калькулятор, СМС), оцени:
    - Был ли инструмент вызван уместно (клиент действительно просил расчёт или отправку)?
    - Правильно ли бот заполнил параметры (предмет, стоимость, валюта)?
    - Прошёл ли бот через read-back (перечислил параметры, спросил "всё верно?") ДО первого вызова калькулятора?
    - Подтвердил ли бот изменение параметра ДО повторного расчёта?
    - Предложил ли бот отправить график по СМС после расчёта?
    - Правильно ли применил конвертацию USD->BYN для физлиц (с озвучкой курса)?
    - Правильно ли отклонил EUR/RUB для физлиц с понятным сообщением?
    - Корректно ли передал тип графика (аннуитетный/линейный) в калькулятор?
    - Если клиент просил линейный график, сработало ли это?
    - Если инструмент не был вызван, но клиент спрашивал про расчёт, это ошибка.
    Если инструменты не использовались и не требовались, ставь 10.
11. profile_hygiene: сколько раз бот ПОВТОРНО спрашивал у клиента информацию, которую тот уже сообщал в этом разговоре (тип клиента, предмет, стоимость, валюта, срок, аванс, тип графика)? Каждый повторный вопрос — потеря очков.
12. stop_command_respect: если клиент говорил "стоп", "подожди", "помолчи", "хватит", правильно ли бот замолчал и не продолжал говорить? Перечисли случаи игнорирования.
13. defaults_assumed: называл ли бот параметры по умолчанию ("аванс 30%", "срок 36 мес") как будто это факт, без подтверждения клиентом? Перечисли случаи.

Верни строго JSON:
{
  "scores": {"banned_phrases": N, "specialist_overuse": N, "humor_and_tone": N, "answer_completeness": N, "response_variety": N, "tool_use_quality": N, "profile_hygiene": N, "stop_command_respect": N},
  "issues": [{"type": "тип", "severity": "critical|important|minor", "detail": "описание", "suggested_fix": "предложение"}],
  "kb_gaps": ["тема1", "тема2"],
  "banned_phrases_found": ["фраза1"],
  "specialist_count": {"offered": N, "appropriate": N},
  "tool_calls": {"calculator_called": true/false, "sms_called": true/false, "missed_opportunity": "описание или null", "readback_before_first_calc": true/false, "change_confirmed_before_recalc": true/false, "sms_offered_after_calc": true/false, "usd_to_byn_conversion_done": true/false, "eur_rub_rejected_cleanly": true/false, "type_schedule_forwarded_correctly": true/false, "linear_graph_honored": true/false, "linear_requests_count": N, "linear_successes_count": N},
  "profile_hygiene": {"repeat_asks": [{"field": "имя поля", "count": N}], "total_repeat_asks": N},
  "stop_command_events": [{"client_said": "фраза", "bot_respected": true/false}],
  "defaults_assumed": ["случай1"],
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


def save_transcript(
    session_id: str,
    transcript: list[dict[str, str]],
    state_dir: Path,
    transport: str = "browser",
    phone: str = "",
) -> Path:
    """Save individual session transcript as a separate JSON file."""
    transcripts_dir = state_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{session_id}.json"
    record = {
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "transport": transport,
        "phone": phone,
        "turn_count": len(transcript),
        "transcript": transcript,
    }
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
