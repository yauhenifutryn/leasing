#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p /tmp/voice_compare

echo "=== Voice Comparison: v4 eugene (normalized) vs v5 aidar ==="

"$APP_DIR/.venv-voice-oss/bin/python" << 'PYEOF'
import torch, wave, numpy as np

TEXT = "Здравствуйте! Меня зовут Евгений, я голосовой помощник компании Микро Лизинг. Для оформления лизинга вам понадобятся паспорт и подтверждение дохода. Генеральный директор нашей компании Дедков Вадим Николаевич. Рассказать подробнее?"

def save_wav(audio_np, name, sr=24000):
    # Normalize to -1dB peak
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.9
    pcm = (audio_np * 32767).astype(np.int16).tobytes()
    path = f"/tmp/voice_compare/{name}.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    rms = np.sqrt(np.mean(audio_np**2))
    print(f"  {name}: peak={np.abs(audio_np).max():.3f} rms={rms:.4f}")

# v4 eugene (normalized)
print("\n--- v4_ru eugene (normalized) ---")
model_v4, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker="v4_ru", trust_repo=True)
audio = model_v4.apply_tts(text=TEXT, speaker="eugene", sample_rate=24000)
save_wav(audio.detach().cpu().numpy(), "v4_eugene_normalized")

# v4 aidar (normalized)
print("\n--- v4_ru aidar (normalized) ---")
audio = model_v4.apply_tts(text=TEXT, speaker="aidar", sample_rate=24000)
save_wav(audio.detach().cpu().numpy(), "v4_aidar_normalized")

# v5 aidar (normalized)
print("\n--- v5_4_ru aidar (normalized) ---")
from silero import silero_tts
model_v5, _ = silero_tts(language="ru", speaker="v5_4_ru")
audio = model_v5.apply_tts(text=TEXT, speaker="aidar", sample_rate=24000, put_accent=True, put_yo=True)
save_wav(audio.detach().cpu().numpy(), "v5_aidar_normalized")

# v5 baya for reference
print("\n--- v5_4_ru baya (normalized) ---")
audio = model_v5.apply_tts(text=TEXT, speaker="baya", sample_rate=24000, put_accent=True, put_yo=True)
save_wav(audio.detach().cpu().numpy(), "v5_baya_normalized")

print("\nDone! Download and compare:")
print('scp -P 43999 "root@185.151.171.35:/tmp/voice_compare/*.wav" ~/Downloads/voice_compare/')
PYEOF
