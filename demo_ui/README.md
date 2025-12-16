# Demo UI (локальный демо‑интерфейс)

Цель: добавить “коммерческий демо” UI поверх существующего пайплайна, **не меняя core‑логику**. Всё лежит в `demo_ui/` и может быть удалено в любой момент.

## Запуск

Из корня репозитория:

```bash
cd /Users/jenyafutrin/Desktop/leasing
source .venv/bin/activate
python demo_ui/server.py
```

Откройте: `http://127.0.0.1:8787`

> Если порт занят: `DEMO_UI_PORT=9000 python demo_ui/server.py`

## Что умеет

- **Start New Session** — создаёт новую “сессию”.
- Запуск whitelisted задач через кнопки: `make check`, `make analyze-calls`, `make kb`, `make kb-markdown`, и т.д.
- **Logs** — просмотр логов запусков.
- **Metrics** — простые бизнес‑графики по данным в `insights_*` (рендерятся локально в SVG, без внешних JS библиотек).
- **JSON Viewer** — просмотр файлов из допустимых директорий (`insights_*`, `knowledge_base`, `nlu_output`, `transcripts_clean`).
- **Feedback** — сохранение обратной связи в `demo_ui/.state/feedback.jsonl` (не коммитится).

## Важно

- UI **ничего не коммитит** и не трогает core‑скрипты.
- Runtime‑состояние хранится в `demo_ui/.state/` и игнорируется Git’ом.
- Для выполнения `make ...` желательно запускать сервер в активированной `.venv` (или хотя бы с корректным `python`/зависимостями).
