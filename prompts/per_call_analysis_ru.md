Вход: расшифровка телефонного разговора (агент–клиент) на русском языке в формате JSON:
{
  "conversation_id": "...",
  "segments": [{"role":"agent|client|null","text":"..."}, ...]
}

Задача: извлечь структурированную информацию.
Вернуть строго JSON со следующими полями:
{
  "conversation_id": "...",
  "main_issue": "краткое описание главной темы запроса",
  "client_intent": "намерение клиента (таксономия уровня 1-2)",
  "subtopics": ["...","..."],
  "agent_actions": ["..."],
  "required_data": ["номер договора","паспорт","телефон" ...],
  "resolution_status": "resolved|partially_resolved|unresolved",
  "handoff": "не нужен|нужна передача в отдел ...|перезвон",
  "emotions": {"client":"neutral|angry|stressed|confused|positive","agent":"empathetic|neutral|rushed"},
  "quality_flags": ["нет внятного ответа","долгие ожидания", ...],
  "verbatim_QA_pairs": [{"q":"вопрос клиента","a":"ответ агента"}],
  "faq_candidate_question": "каноническая формулировка вопроса",
  "faq_candidate_answer": "краткий лучше-практики ответ (на основе разговора)"
}
