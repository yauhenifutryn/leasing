#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"$APP_DIR/.venv-voice-oss/bin/python" << 'PYEOF'
import torch

variants = ["v3_1_ru", "v4_ru"]
for v in variants:
    try:
        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models", model="silero_tts", language="ru", speaker=v, trust_repo=True)
        has_random = hasattr(model, "sample_random_speaker")
        speakers = model.speakers if hasattr(model, "speakers") else "unknown"
        print(f"{v}: loaded OK | random_speaker={has_random} | speakers={speakers}")
    except Exception as e:
        print(f"{v}: FAILED ({e})")

# Also check current v5_4_ru via silero package
from silero import silero_tts
model5, _ = silero_tts(language="ru", speaker="v5_4_ru")
has_random5 = hasattr(model5, "sample_random_speaker")
print(f"v5_4_ru (pip): loaded OK | random_speaker={has_random5}")
PYEOF
