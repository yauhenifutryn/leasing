#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p /tmp/voice_samples

echo "=== Random Voice Generator (Silero v4_ru) ==="

"$APP_DIR/.venv-voice-oss/bin/python" << 'PYEOF'
import torch, wave

model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models", model="silero_tts", language="ru", speaker="v4_ru", trust_repo=True)

TEXT = "Здравствуйте! Я голосовой помощник компании Микро Лизинг. Для оформления лизинга вам понадобятся паспорт и подтверждение дохода. Рассказать подробнее?"

def save_wav(audio, name, sr=24000):
    pcm = (audio * 32767).to(torch.int16).numpy().tobytes()
    path = f"/tmp/voice_samples/{name}.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    print(f"  {name}: {path}")

# Native voices including eugene (not in v5)
print("\n--- Native Voices (v4_ru) ---")
for speaker in ["aidar", "eugene", "baya", "xenia", "kseniya"]:
    audio = model.apply_tts(text=TEXT, speaker=speaker, sample_rate=24000)
    save_wav(audio, f"v4_{speaker}")

# 10 random voices
print("\n--- Random Voices (v4_ru, 10 samples) ---")
for i in range(10):
    audio = model.apply_tts(text=TEXT, speaker="random", sample_rate=24000)
    save_wav(audio, f"v4_random_{i+1:02d}")

print("\nDone! 15 samples generated.")
print("Download: scp -P 43999 \"root@185.151.171.35:/tmp/voice_samples/v4_*.wav\" ~/Downloads/voice_samples/")
PYEOF
