#!/usr/bin/env bash
# test_voices.sh - Generate voice samples, download to local machine for listening
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="/tmp/voice_samples"
mkdir -p "$OUT_DIR"

echo "=== Voice Sample Generator ==="
echo "Generating samples in $OUT_DIR ..."

"$APP_DIR/.venv-voice-oss/bin/python" << 'PYEOF'
import torch, wave
from silero import silero_tts

model, _ = silero_tts(language="ru", speaker="v5_4_ru")
model.to(torch.device("cpu"))

TEXT = "Здравствуйте! Меня зовут Виктор, я голосовой помощник компании Микро Лизинг. Для оформления лизинга вам понадобятся паспорт и подтверждение дохода. Рассказать подробнее?"

def save_wav(audio, name, sr=24000):
    pcm = (audio * 32767).to(torch.int16).numpy().tobytes()
    path = f"/tmp/voice_samples/{name}.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    print(f"  {name}: {path}")

# All native voices (aidar=male, rest=female)
print("\n--- Native Voices ---")
for speaker in ["aidar", "baya", "xenia", "kseniya"]:
    audio = model.apply_tts(text=TEXT, speaker=speaker, sample_rate=24000, put_accent=True, put_yo=True)
    save_wav(audio, f"native_{speaker}")

print("\nDone! 4 native samples generated.")
print("Note: this Silero model version does not support random voice generation.")
print("Download to your Mac:")
print("  scp -P 50576 root@<IP>:/tmp/voice_samples/*.wav ~/Downloads/voice_samples/")
PYEOF

echo ""
echo "=== Samples ready ==="
echo "Download: scp -r -P 50576 root@$(hostname -I | awk '{print $1}'):/tmp/voice_samples/ ~/Downloads/voice_samples/"
