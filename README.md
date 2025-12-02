# Leasing AI Pipeline

## Repository Layout

```
leasing-ai/
├─ audio/                         # put .wav/.mp3 here (20 test files first)
├─ transcripts_clean/             # WhisperX JSON (ready for analysis)
├─ insights_per_call/             # JSON with intents, issues, outcomes per call
├─ nlu_output/                    # flat Q&A records for NLU / KHUB ingestion
├─ insights_batches/              # “batch” summaries (10–20 calls per file)
├─ insights_global/               # global rollups (top issues, playbooks)
├─ knowledge_base/                # final KB (FAQ/flows) JSON+YAML
├─ scripts/
│  ├─ 00_setup_checks.py
│  ├─ 10_transcribe_whisperx.py
│  ├─ 11_transcribe_whisper_cli.sh
│  ├─ 20_clean_and_diarize.py
│  ├─ 30_analyze_per_call.py
│  ├─ 31_analyze_batch_rollup.py
│  ├─ 32_global_aggregation.py
│  ├─ 40_deduplicate_embeddings.py
│  ├─ 50_build_kb.py
│  └─ utils.py
├─ prompts/
│  ├─ per_call_analysis_ru.md
│  ├─ batch_rollup_ru.md
│  └─ kb_entry_synthesis_ru.md
├─ requirements.txt
├─ .env.example
└─ Makefile
```

> Note: data-heavy folders (`audio/`, `transcripts_clean/`, `insights_*`, `knowledge_base/`, `nlu_output/`, etc.) are `.gitignore`d. They’ll be created automatically when their respective scripts run. The only directory you need to prepare manually is `audio/` so you can drop source recordings before running `make transcribe`.

## Getting Started

```bash
git clone git@github.com:yauhenifutryn/leasing.git
cd leasing
```

All commands below assume you are inside this project directory.

## Environment & Tooling

Install required system packages, set up a Python virtual environment, and install Python dependencies.

### System Dependencies

**macOS**

```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu)**

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### Python Environment

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .\.venv\Scripts\activate

# Upgrade build tooling
pip install --upgrade pip wheel setuptools==65.6.3

# Project dependencies (WhisperX/pyannote via tarballs, CUDA 11.8 wheels)
pip install -r requirements.txt
```

Notes on Torch/CUDA:
- `requirements.txt` pins `torch/torchaudio/torchvision` to 2.0.0/2.0.1/0.15.1 with `+cu118` wheels (CUDA 11.8), which match WhisperX 3.1.1 expectations.
- If your GPUs require a different CUDA runtime, adjust the `--extra-index-url` and torch* versions accordingly.

### Notes

- WhisperX and pyannote.audio are installed from release tarballs (no `git clone` during `pip install`), avoiding build-backend warnings and git HTTPS issues.
- WhisperX provides accurate timestamps and optional diarization. Whisper CLI is included as a fallback.
- To enable diarization with WhisperX, create a free Hugging Face token (pyannote models) and place it in `.env` (see `.env.example`).
- Set `OPENAI_MODEL` (pipeline scripts) and `REVIEW_OPENAI_MODEL` (Streamlit UI, default `gpt-5.1`) to the chat-completions models you plan to use, e.g., `gpt-5.1` for `make analyze-calls` and the review app.
- Ensure you comply with client privacy requirements before exporting any data.
- На сервере используйте только одно окружение `conda` (`lease`); не смешивайте с `.venv`.
- VAD (silero) отключён по умолчанию (`vad_model=None`) для стабильности; диаризацию pyannote можно включать флагом `--disable-diarization`/токеном HF при необходимости.

## Makefile Targets

```Makefile
make check             # run setup checks (ffmpeg, API keys)
make transcribe        # GPU default: WhisperX -> clean/diarize (transcripts_clean)
make transcribe-gpu    # explicit GPU run (same as `make transcribe`)
make transcribe-cpu    # CPU fallback (slow)
make transcribe-cli    # Whisper CLI fallback (CPU)
make analyze-calls     # per-call analysis using OpenAI
make nlu-export        # flat Q&A export (JSONL) for NLU systems
make rollup            # batch-level rollups (deduplicated)
make aggregate         # global aggregation step
make dedup             # embedding-based FAQ deduplication
make kb                # build final knowledge base entries (JSON + YAML)
```

## Pipeline Overview

1. **Transcription** – `make transcribe` (GPU default) or `make transcribe-cpu` runs WhisperX (`scripts/10_transcribe_whisperx.py`) and writes directly to `transcripts_clean/` (ready for analysis).
2. **Per-Call Analysis** – `scripts/30_analyze_per_call.py` sends structured prompts to OpenAI for intent, resolution, and QA extraction.
3b. **Flat Q&A Export (optional)** – `scripts/35_export_nlu_pairs.py` flattens every question/answer pair into `nlu_output/nlu_pairs.jsonl` with hashtags for NLU/KHUB ingestion.
4. **Batch Rollups** – `scripts/31_analyze_batch_rollup.py` summarizes groups of calls to avoid duplicates.
5. **Global Aggregation** – `scripts/32_global_aggregation.py` produces consolidated views of intents and FAQ clusters.
6. **Embedding Deduplication** – `scripts/40_deduplicate_embeddings.py` clusters similar questions using SentenceTransformers.
7. **Knowledge Base Build** – `scripts/50_build_kb.py` synthesizes final FAQ/KB entries (JSON & YAML).

## Practical Guidance

- **Batch Audio Processing**: Queue 10–20 files at a time. Use multiprocessing carefully if you have GPU resources to spare.
- **Hierarchical GPT Summaries**: Extract per-call insights, then deduplicate/roll up in batches of 10–20 before global aggregation to control token costs and repetition.
- **Speaker Roles**: Start with heuristics in `20_clean_and_diarize.py`. Enable diarization via `--enable_diarization` for higher accuracy once you configure `HUGGINGFACE_TOKEN`.
- **Quality & Compliance**: Mask sensitive data before uploading anywhere. Add guardrails in prompts to prevent leaking PII.
- **Scaling to 1,000+ Calls**: Keep transcription and GPT analysis in sequential batches. Persist intermediate artifacts so you can resume from any stage.
- **Retell AI Integration (Future)**: The knowledge base JSON/YAML can be adapted as a Retell routing table with minimal code.

## Quick Start

1. Clone the repo and `cd` into it (see above).
2. Populate `audio/` with your `.wav/.mp3/.m4a/.flac` files (start with ~20 for smoke testing).
3. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`, `HUGGINGFACE_TOKEN` (optional), and preferred `OPENAI_MODEL`.
4. Follow the setup commands above to create/activate `.venv` and install dependencies from `requirements.txt`.
5. Run the pipeline via the Makefile targets in order. Inspect outputs in the respective directories before proceeding to the next stage.

## Accuracy Review UI

To let reviewers validate and correct entries without touching JSON manually, run the Streamlit app:

```bash
source .venv/bin/activate
streamlit run scripts/review_app.py
```

The UI cycles through `knowledge_base/kb_faq_ru.json`. For каждой записи можно:

- отметить ответ корректным (кнопка «Подтвердить корректность» снимает `pending_review`) или указать исправленную формулировку;
- добавить комментарий/причину правки;
- связанные Q&A (из `nlu_output/nlu_pairs.jsonl`) подбираются автоматически, показываются с исходными ответами и обновляются вместе с записью; LLM корректирует только неточные фрагменты;
- автоматически обновить `knowledge_base`, `insights_global/global_faq_clusters_dedup.json`, `nlu_output/nlu_pairs.jsonl` и сохранить запись в `corrections/corrections.jsonl`.
- панель "История правок" отображает последние действия и позволяет откатить последнюю правку (кнопка «Отменить последнюю правку» возвращает исходный ответ и пересобирает `nlu_output`).

Перед запуском убедитесь, что выполнены `make analyze-calls`, `make dedup`, `make kb` и `make nlu-export`, чтобы все необходимые файлы существовали.

## Testing & Validation

- `make check` verifies that `ffmpeg` and API keys are available.
- Inspect intermediate outputs (`transcripts_*`, `insights_*`) for anomalies before running downstream stages.
- Adjust heuristics, prompts, and clustering thresholds as you observe real data.

## Staying Up to Date

- I’ll keep pushing fixes/enhancements to `main` in this GitHub repo.
- On your machine, run `git pull` inside the project folder to pick up the latest changes before starting a new processing run.

## Серверный запуск (GPU, Rus)

1) Выбор GPU  
   - Рекомендуемый вариант: **A100 40 GB**. Скорость: ~9 мин 20 с на 20 аудио ~10 мин. Цена: ~**$0.6/час** на vast.ai.  
   - Альтернатива: **4090** (дешевле, но менее стабильна под длительной нагрузкой).  
   - Не брать **5090/Blackwell** — требует свежих драйверов, часто не работает “из коробки”.

2) Подготовка окружения на сервере  
   ```bash
   cd /workspace
   rm -rf leasing
   git clone https://github.com/yauhenifutryn/leasing.git   # auth: SSH key или personal access token
   cd leasing

   conda create -y -n lease python=3.10
   conda activate lease
   make install   # ставит PyTorch cu121 для A100 + прочее
   ```

3) Папка с аудио  
   ```bash
   mkdir -p /workspace/leasing/audio
   ```
   Пример загрузки с Mac (нужен доступ по SSH к серверу; подставьте свой порт/хост):
   ```bash
   rsync -avz --partial --progress -e "ssh -p <PORT>" \
     audio/ root@<HOST>:/workspace/leasing/audio/
   ```

4) Запуск транскрипции на сервере  
   ```bash
   cd /workspace/leasing
   conda activate lease
   make transcribe-gpu
   ```
   Результаты: `/workspace/leasing/transcripts_clean/`.

5) Скачать результаты на локальный компьютер (Mac → Downloads)  
   ```bash
   rsync -avz --progress -e "ssh -p <PORT>" \
     root@<HOST>:/workspace/leasing/transcripts_clean/ \
     ~/Downloads/transcripts_clean/
   ```

   Если нужен HF токен для диаризации, перед запуском экспортируйте его:
   ```bash
   export HUGGINGFACE_TOKEN="hf_..."  # или свой токен; примите условия модели https://huggingface.co/pyannote/speaker-diarization-3.1
   ```

   Важно: на сервере используйте только окружение `conda activate lease` (не активируйте `.venv`), VAD отключён в коде.

6) “Пассивный” запуск (чтобы не упало при обрыве SSH) — tmux  
   ```bash
   tmux new -s work          # создать сессию
   make transcribe-gpu       # запустить внутри
   # отсоединиться: Ctrl+b, затем d
   tmux attach -t work       # вернуться позже
   tmux ls                   # список сессий
   tmux kill-session -t work # убить сессию при необходимости
   ```
