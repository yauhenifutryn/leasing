# PersonaPlex Demo (Experiment)

This folder is isolated from the production RAG system.
It provides a minimal CLI for trying NVIDIA PersonaPlex with text or audio input.

## Hardware and OS
- NVIDIA GPU required, A100 or H100 recommended
- Linux preferred
- CUDA-enabled PyTorch

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=your_token
```

## Usage
Text input:
```bash
python run_demo.py --text "Hello, how can you help me?" --out out.wav
```

Audio input from file:
```bash
python run_demo.py --audio ./sample.wav --out out.wav
```

Microphone input (records 5 seconds):
```bash
python run_demo.py --mic 5 --out out.wav
```

## Notes
- PersonaPlex is English-only and does not provide RAG grounding.
- This is a best-effort demo wrapper because upstream APIs can change.
- If model loading or inference fails, check the upstream repo for updates.

## Limitations
- Russian support is not expected with this model.
- Latency is GPU bound and depends on model size and CUDA stack.
- This demo is not a production service.
