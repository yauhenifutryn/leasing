PY=python
SRC=scripts

install:
	# 1. Install PyTorch Nightly FIRST (RTX 5090 / sm_120 support)
	$(PY) -m pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
	# 2. Install other dependencies (will use the installed torch)
	$(PY) -m pip install -r requirements.txt
	# 3. Install WhisperX (no deps to avoid downgrading torch)
	$(PY) -m pip install "git+https://github.com/m-bain/whisperx.git" --no-deps

check:
	$(PY) $(SRC)/00_setup_checks.py

# 1a) Transcribe (WhisperX) – Local/CPU (for testing)
transcribe-cpu:
	$(PY) $(SRC)/10_transcribe_whisperx.py --in audio --out transcripts_raw --device cpu --compute-type int8

# 1b) Transcribe (WhisperX) – Server/GPU (Production)
transcribe-gpu:
	$(PY) $(SRC)/10_transcribe_whisperx.py --in audio --out transcripts_raw --device cuda --compute-type float16 --batch-size 16

# 1c) Transcribe (Whisper CLI) – quick fallback
transcribe-cli:
	bash $(SRC)/11_transcribe_whisper_cli.sh audio transcripts_raw

# 2) Clean + diarize (map SPEAKER_00/01 to client/agent)
clean:
	$(PY) $(SRC)/20_clean_and_diarize.py --in transcripts_raw --out transcripts_clean

# 3) Per-call analysis (JSON per call)
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
