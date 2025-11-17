Вход: массив карточек per-call (10–20 шт.). Задача: обобщить без дубликатов.
Верните JSON:
{
  "batch_id": "...",
  "top_intents": [{"intent":"...", "count":N, "examples":["..."]}],
  "faq_clusters": [
    {
      "cluster_label": "кратко о чем кластер",
      "canonical_q": "канонический вопрос",
      "best_answer": "обобщенный ответ",
      "source_conversation_ids": ["...","..."]
    }
  ],
  "quality_findings": ["..."],
  "handoff_patterns": ["..."],
  "data_requirements": ["номер договора", "..."]
}
