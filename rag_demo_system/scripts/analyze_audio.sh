#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

"$APP_DIR/.venv-voice-oss/bin/python" << 'PYEOF'
import wave, struct, math, numpy as np

files = {
    "v4_eugene_norm": "/tmp/voice_compare/v4_eugene_normalized.wav",
    "v4_aidar_norm": "/tmp/voice_compare/v4_aidar_normalized.wav",
    "v5_aidar_norm": "/tmp/voice_compare/v5_aidar_normalized.wav",
    "v5_baya_norm": "/tmp/voice_compare/v5_baya_normalized.wav",
}

print("=== Audio Analysis ===\n")
for name, path in files.items():
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        samples = np.array(struct.unpack(f"<{len(frames)//2}h", frames), dtype=np.float64) / 32768.0
        duration = len(samples) / sr

        # Split into speech and silence segments (threshold-based)
        frame_size = int(sr * 0.025)  # 25ms frames
        energies = []
        for i in range(0, len(samples) - frame_size, frame_size):
            frame = samples[i:i+frame_size]
            energies.append(np.sqrt(np.mean(frame**2)))
        energies = np.array(energies)

        speech_threshold = np.percentile(energies, 30)  # bottom 30% is silence
        speech_frames = energies[energies > speech_threshold]
        silence_frames = energies[energies <= speech_threshold]

        # Signal-to-noise ratio (speech energy vs silence energy)
        speech_rms = np.mean(speech_frames) if len(speech_frames) > 0 else 0
        silence_rms = np.mean(silence_frames) if len(silence_frames) > 0 else 0.0001
        snr_db = 20 * math.log10(speech_rms / silence_rms) if silence_rms > 0 else 0

        # Reverb indicator: energy decay after speech segments
        # Higher silence_rms relative to speech = more reverb
        reverb_ratio = silence_rms / speech_rms if speech_rms > 0 else 0

        # Spectral centroid (brightness)
        fft = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1/sr)
        centroid = np.sum(freqs * fft) / np.sum(fft) if np.sum(fft) > 0 else 0

        print(f"{name}:")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Speech RMS: {speech_rms:.4f}")
        print(f"  Silence RMS: {silence_rms:.4f}")
        print(f"  SNR: {snr_db:.1f} dB")
        print(f"  Reverb ratio: {reverb_ratio:.4f} (lower=cleaner)")
        print(f"  Spectral centroid: {centroid:.0f} Hz")
        print()
    except Exception as e:
        print(f"{name}: ERROR ({e})\n")

print("--- Interpretation ---")
print("Reverb ratio: <0.15 = clean, 0.15-0.25 = slight reverb, >0.25 = noticeable echo")
print("SNR: >20dB = clean, 15-20dB = acceptable, <15dB = noisy/reverby")
PYEOF
