# RAG Demo System (Micro Leasing)

Демо‑система полностью изолирована и живёт в `rag_demo_system/`. Удаление этой папки **не влияет** на основной проект.

Цели:
- Воссоздать **качество RAG ElevenLabs** на собственной инфраструктуре.
- Строгое grounding: **только** из Markdown KB.
- Быстрая, лаконичная выдача, без галлюцинаций.

## Структура

```
rag_demo_system/
  backend/          # FastAPI API
  frontend/         # UI (HTML/CSS/JS)
  config/           # YAML конфиги и системный промпт
  scripts/          # утилиты запуска
  tests/            # проверки
  .state/           # локальное состояние и логи (не коммитится)
```

## ВАЖНО (Read‑only core)
- Ничего в корневых папках не изменяем.
- Вся демо‑логика — только в `rag_demo_system/`.

---

## 1) Быстрый локальный запуск

### 1.1 Qdrant (Vector DB)
```bash
docker compose -f rag_demo_system/docker-compose.yml up -d
```

### 1.2 Backend
```bash
cd rag_demo_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export RAG_LLM_BASE_URL="https://<public-tunnel-host>/v1"
export RAG_LLM_MODEL="Qwen3-30B-A3B-Instruct"

./scripts/run_backend.sh
```

Backend: `http://127.0.0.1:8000`

### 1.3 Frontend
Откройте `rag_demo_system/frontend/index.html` в браузере.

### Полный старт одной командой
```bash
./rag_demo_system/scripts/run_all.sh
```

---

## 2) Индексация базы знаний

Markdown KB используется как **единственный источник правды**:
`knowledge_base/kb_faq_ru.md`

Запустить индексацию:
```bash
curl -X POST http://127.0.0.1:8000/api/index
```

## RAG Quality Defaults (Demo)
- Hybrid retrieval: Vector topK=8 + BM25 topK=8 → merge → rerank → final topN=6
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (CPU по умолчанию)
- Строгая проверка уверенности: если evidence слабый → отказ
- Контекстный бюджет: 1800 токенов (предпочитаем меньше, но увереннее)

---

## 3) Демонстрация диалога

UI отправляет запросы в `/api/chat`.

**Внимание:**
- До первого ответа ассистент **обязательно** запрашивает согласие на обработку ПДн.
- Без согласия он завершает диалог.

---

## 4) Серверный запуск (AWS / on‑prem)

1. Поднять Qdrant через `docker-compose`.
2. Запустить backend:
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```
3. Пробросить порт 8000 наружу (reverse proxy / ingress).
4. Frontend — статические файлы (можно отдать через Nginx).

---

## 5) Google Colab (vLLM + Qwen)

### Коллаб:
- Запустить vLLM OpenAI‑compatible сервер.
- Пробросить через публичный туннель.

Пример:
```
https://<public-tunnel-host>/v1
```

### Конфиг (rag_demo_system/config/app.yaml):
```
llm:
  base_url: "https://<public-tunnel-host>/v1"
  model: "Qwen3-30B-A3B-Instruct"
```

**TODO:** уточнить точное имя модели в vLLM.

---

## 6) VS Code ↔ Colab (optional)

Рекомендация: использовать `ssh` туннель или `ngrok/cloudflared` для стабильного URL.

TODO:
- Добавить инструкцию для VS Code Remote + Colab.

---

## 7) Voice (ElevenLabs)

Voice‑интеграция **будет добавлена позже** (в этой итерации выключена).
UI уже содержит переключатель, но backend возвращает `501 Not Implemented`.

---

## Smoke test

Минимальная проверка (health → index → chat → used_knowledge):
```bash
./rag_demo_system/scripts/smoke_test.sh
```

---

## 8) Qwen local model connection (on‑prem)

Если Qwen будет запущен локально (vLLM):
```
RAG_LLM_BASE_URL=http://localhost:8000/v1
RAG_LLM_MODEL=Qwen3-30B-A3B-Instruct
```

---

## TODO (креды/инфра)
- [ ] RAG_LLM_BASE_URL / RAG_LLM_MODEL для vLLM
- [ ] ELEVENLABS_API_KEY (на будущее)
- [ ] Добавить HTTPS reverse proxy для продакшена

---

## API endpoints

- `POST /api/index`
- `POST /api/retrieve`
- `POST /api/chat`
- `GET  /api/health`
- `GET  /api/logs`

---

## Требования к качеству
- Никаких галлюцинаций.
- Только ответы из retrieved chunks.
- Если контекста нет — строгий отказ.
- Короткие ответы (1–4 предложения по умолчанию).
