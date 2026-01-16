# AGENTS

## Project overview
This repository contains a Python-based pipeline for audio transcription (WhisperX) and downstream analysis to build a knowledge base for Micro Leasing, plus optional demo UIs for review and RAG.

## Setup and commands
- **Install deps (GPU server or local, per Makefile):** `make install`
- **Env check:** `make check`
- **Transcribe (default):** `make transcribe` (GPU)  
  - CPU test: `make transcribe-cpu`
  - CLI fallback: `make transcribe-cli`
- **Analyze pipeline:**  
  - `make analyze-calls`  
  - `make nlu-export`  
  - `make rollup`  
  - `make aggregate`  
  - `make dedup`  
  - `make kb`  
  - `make kb-markdown`
- **Review UI (Streamlit):** `streamlit run scripts/review_app.py`
- **Demo UI:** `python demo_ui/server.py`
- **RAG demo system:** `bash rag_demo_system/scripts/run_all.sh` (Qdrant + backend)
- **Tests (only where present):** `pytest rag_demo_system/tests`

## Repo map
- `scripts/` — core transcription + analysis pipeline (primary logic)
- `knowledge_base/` — generated KB outputs (JSON/Markdown)
- `transcripts_clean/` — transcription outputs
- `insights_*` — intermediate analysis artifacts
- `nlu_output/` — exported NLU pairs
- `corrections/` — review corrections log
- `demo_ui/` — local demo UI for running make steps/logs/metrics
- `rag_demo_system/` — self-contained RAG demo (API + UI)
- `audio/` — input audio files (local or synced from server)
- `prompts/` — prompt templates/configs

## Non-negotiables
- Do not commit secrets or `.env` files.
- Do not modify core pipeline behavior unless explicitly requested.
- Avoid large refactors; keep changes scoped to the requested task.
- Do not push to GitHub unless explicitly asked.

## Coding conventions
- Python-first repo; keep scripts simple and explicit.
- Prefer clear error messages and safe file handling.
- Keep output formats stable (JSON schema, filenames, folder paths).
- Use ASCII in source unless the file already uses Unicode.

## Testing expectations
- Run `make check` after changes to pipeline scripts when feasible.
- If working in `rag_demo_system/`, run `pytest rag_demo_system/tests` (if backend running, smoke test script can be used instead).

## Change safety
- Ask before adding new production dependencies.
- Avoid unrelated edits to generated data folders (`transcripts_*`, `insights_*`, `knowledge_base`, `nlu_output`) unless explicitly requested.
