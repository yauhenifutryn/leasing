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
- On the server, use only the `conda` environment (`lease`); do not mix with `.venv`.
- VAD (silero) is disabled by default (`vad_model=None`) for stability; pyannote diarization can be enabled via `--disable-diarization` flag / HF token when needed.

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
make kb-markdown       # export KB JSON → Markdown (.md) for Retell KB (flat + structured)
make markdown          # alias for kb-markdown
```

Note: `make kb-markdown` produces both `knowledge_base/kb_faq_ru.md` and `knowledge_base/kb_faq_ru_structured.md`.

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

The UI cycles through `knowledge_base/kb_faq_ru.json`. For each entry you can:

- Mark the answer as correct (the "Confirm correctness" button clears `pending_review`) or provide a corrected version;
- Add a comment or reason for the edit;
- Related Q&A pairs (from `nlu_output/nlu_pairs.jsonl`) are matched automatically, shown with their original answers, and updated together with the entry; the LLM only corrects inaccurate fragments;
- Automatically update `knowledge_base`, `insights_global/global_faq_clusters_dedup.json`, `nlu_output/nlu_pairs.jsonl` and save the entry to `corrections/corrections.jsonl`.
- The "Edit History" panel shows recent actions and allows undoing the last edit (the "Undo last edit" button restores the original answer and rebuilds `nlu_output`).

Before running, make sure `make analyze-calls`, `make dedup`, `make kb`, and `make nlu-export` have been executed so that all required files exist.

## Testing & Validation

- `make check` verifies that `ffmpeg` and API keys are available.
- Inspect intermediate outputs (`transcripts_*`, `insights_*`) for anomalies before running downstream stages.
- Adjust heuristics, prompts, and clustering thresholds as you observe real data.

## Staying Up to Date

- I’ll keep pushing fixes/enhancements to `main` in this GitHub repo.
- On your machine, run `git pull` inside the project folder to pick up the latest changes before starting a new processing run.

## Server Run (GPU)

1) GPU selection  
   - Recommended: **A100 40 GB**. Speed: ~9 min 20 sec for 20 audio files (~10 min each). Price: ~**$0.6/hr** on vast.ai.  
   - Alternative: **4090** (cheaper, but less stable under sustained load).  
   - Avoid **5090/Blackwell**: requires bleeding-edge drivers, often does not work out of the box.

2) Server environment setup  
   ```bash
   cd /workspace
   rm -rf leasing
   git clone https://github.com/yauhenifutryn/leasing.git   # auth: SSH key or personal access token
   cd leasing

   conda create -y -n lease python=3.10
   conda activate lease
   make install   # installs PyTorch cu121 for A100 + dependencies
   ```

3) Audio folder  
   ```bash
   mkdir -p /workspace/leasing/audio
   ```
   Upload from Mac (requires SSH access to the server; substitute your port/host):
   ```bash
   rsync -avz --partial --progress -e "ssh -p <PORT>" \
     audio/ root@<HOST>:/workspace/leasing/audio/
   ```

4) Run transcription on server  
   ```bash
   cd /workspace/leasing
   conda activate lease
   make transcribe-gpu
   ```
   Results: `/workspace/leasing/transcripts_clean/`.

5) Download results to local machine (Mac, Downloads)  
   ```bash
   rsync -avz --progress -e "ssh -p <PORT>" \
     root@<HOST>:/workspace/leasing/transcripts_clean/ \
     ~/Downloads/transcripts_clean/
   ```

   If you need an HF token for diarization, export it before running:
   ```bash
   export HUGGINGFACE_TOKEN="hf_..."  # accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
   ```

   Important: on the server, use only the `conda activate lease` environment (do not activate `.venv`); VAD is disabled in the code.

6) Background run (survives SSH disconnects) with tmux  
   ```bash
   tmux new -s work          # create session
   make transcribe-gpu       # run inside
   # detach: Ctrl+b, then d
   tmux attach -t work       # reconnect later
   tmux ls                   # list sessions
   tmux kill-session -t work # kill session if needed
   ```

## Demo UI (local)

Local demo interface for running `make ...` steps, viewing logs, basic metrics visualization, and JSON inspection.

```bash
cd leasing
source .venv/bin/activate
python demo_ui/server.py
```

Open: `http://127.0.0.1:8787`

## Voice Assistant (`rag_demo_system/`)

Production voice assistant built on top of the knowledge base generated by the call analysis pipeline above. Browser-based, Russian language, knowledge-grounded answers about leasing products.

**Stack:** Whisper STT + Silero TTS + Qwen3.5-35B-A3B-FP8 (vLLM) + Qdrant RAG

```bash
git clone --branch feature/voice-pipeline https://github.com/yauhenifutryn/leasing.git
cd leasing/rag_demo_system
HF_TOKEN=hf_YOUR_TOKEN bash scripts/provision_server.sh
```

See [rag_demo_system/README.md](rag_demo_system/README.md) for full details and [docs/server_deployment_playbook.md](docs/server_deployment_playbook.md) for deployment guide.

## Branches

| Branch | Purpose |
|--------|---------|
| `feature/voice-pipeline` | Production voice assistant (clean, client-ready) |
| `claude/qwen-voice-next` | Experimental: benchmark framework, multi-model testing (Qwen3-Omni, Qwen3-TTS, Qwen3-ASR, Voxtral, SenseVoice) |

## License

This project is proprietary. It is strictly forbidden to use this code for commercial purposes.
The code is open for public viewing solely for portfolio demonstration and evaluation.
See the [LICENSE](LICENSE) file for specific terms.
