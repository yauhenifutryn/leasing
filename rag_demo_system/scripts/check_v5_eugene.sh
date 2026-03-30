#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"$APP_DIR/.venv-voice-oss/bin/python" << 'PYEOF'
import torch

# Try v5 via torch.hub (not pip silero package)
try:
    model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models", model="silero_tts", language="ru", speaker="v5_4_ru", trust_repo=True)
    speakers = model.speakers if hasattr(model, "speakers") else "unknown"
    print(f"v5_4_ru (hub): speakers={speakers}")

    # Try eugene on v5
    try:
        audio = model.apply_tts(text="Привет, это тест.", speaker="eugene", sample_rate=24000)
        print("v5_4_ru eugene: WORKS")
    except Exception as e:
        print(f"v5_4_ru eugene: FAILED ({e})")
except Exception as e:
    print(f"v5_4_ru (hub): FAILED ({e})")
PYEOF
