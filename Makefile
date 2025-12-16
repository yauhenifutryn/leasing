PY=python
SRC=scripts

install:
	# 1. Install PyTorch (CUDA 12.1) first — stable wheels for A100/4090
	$(PY) -m pip install --upgrade torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
	# 2. Install other dependencies (will use the installed torch)
	$(PY) -m pip install -r requirements.txt
	# 3. Install WhisperX (no deps to avoid downgrading torch)
	$(PY) -m pip install "git+https://github.com/m-bain/whisperx.git" --no-deps

check:
	$(PY) $(SRC)/00_setup_checks.py

# default: use GPU pipeline if available
transcribe: transcribe-gpu

# 1a) Transcribe (WhisperX) – Local/CPU (for testing)
transcribe-cpu:
	$(PY) $(SRC)/10_transcribe_whisperx.py --in audio --out transcripts_clean --device cpu --compute-type int8

# 1b) Transcribe (WhisperX) – Server/GPU (Production)
transcribe-gpu:
	$(PY) $(SRC)/10_transcribe_whisperx.py --in audio --out transcripts_clean --device cuda --compute-type float16 --batch-size 16

# 1c) Transcribe (Whisper CLI) – quick fallback
transcribe-cli:
	bash $(SRC)/11_transcribe_whisper_cli.sh audio transcripts_clean

# 2) Per-call analysis (JSON per call)
analyze-calls:
	$(PY) $(SRC)/30_analyze_per_call.py --in transcripts_clean --out insights_per_call --batch-size 10

# 3b) Flat Q&A export for NLU pipelines
nlu-export:
	$(PY) $(SRC)/35_export_nlu_pairs.py --in insights_per_call --out nlu_output/nlu_pairs.jsonl

# 4) Batch rollups (dedupe within batches)
rollup:
	$(PY) $(SRC)/31_analyze_batch_rollup.py --in insights_per_call --out insights_batches

# 5) Global aggregation (merge batch rollups)
aggregate:
	$(PY) $(SRC)/32_global_aggregation.py --in insights_batches --out insights_global

# 6) Embedding-based dedup (merge near-duplicate Qs)
dedup:
	$(PY) $(SRC)/40_deduplicate_embeddings.py --in insights_global --out insights_global

# 7) Build Knowledge Base (FAQ + playbooks)
kb:
	$(PY) $(SRC)/50_build_kb.py --in insights_global --out knowledge_base

# 8) Export KB to Markdown for Retell
kb-markdown:
	$(PY) $(SRC)/55_export_kb_markdown.py --in knowledge_base/kb_faq_ru.json --out knowledge_base/kb_faq_ru.md

# Alias for convenience
markdown: kb-markdown
