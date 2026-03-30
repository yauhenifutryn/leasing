#!/usr/bin/env python3
"""Voice Lab: audition, rate, and persist Silero TTS voices.

Usage:
    python voice_lab.py [--text "Custom test sentence"] [--batch-size 5]

Starts with native Russian voices, then generates random voices in batches.
Ratings are persisted to voices/catalog.json. Random voice embeddings are
saved as .pt files for reuse. Re-running continues where you left off.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path

import torch

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
CATALOG_PATH = VOICES_DIR / "catalog.json"

NATIVE_VOICES = ["aidar", "baya", "xenia", "kseniya"]

DEFAULT_TEXT = "Добро пожаловать в Микро Лизинг. Чем могу помочь?"

RATING_DIMENSIONS = ["quality", "warmth", "clarity"]


def load_catalog() -> dict:
    """Load or create the voice catalog."""
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH) as f:
            return json.load(f)
    return {"voices": [], "active_voice": None}


def save_catalog(catalog: dict) -> None:
    """Persist the catalog to disk."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def load_silero_model():
    """Load Silero TTS model for Russian."""
    from silero import silero_tts

    model, _ = silero_tts(language="ru", speaker="v5_4_ru")
    model.to(torch.device("cpu"))
    return model


def synthesize(model, text: str, speaker: str = None, speaker_embedding=None) -> tuple[bytes, int]:
    """Synthesize text to PCM16 audio."""
    sample_rate = 24000
    if speaker_embedding is not None:
        audio = model.apply_tts(
            text=text,
            speaker=speaker_embedding,
            sample_rate=sample_rate,
            put_accent=True,
            put_yo=True,
        )
    else:
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=True,
            put_yo=True,
        )
    pcm16 = (audio * 32767).to(torch.int16).numpy().tobytes()
    return pcm16, sample_rate


def save_wav(pcm16: bytes, sample_rate: int, path: Path) -> None:
    """Save PCM16 audio as WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)


def play_audio(wav_path: Path) -> None:
    """Play a WAV file using whatever player is available."""
    for player in ["ffplay", "aplay", "play", "paplay"]:
        if shutil.which(player):
            args = [player]
            if player == "ffplay":
                args += ["-nodisp", "-autoexit", "-loglevel", "quiet"]
            args.append(str(wav_path))
            subprocess.run(args, check=False)
            return
    print(f"  [No audio player found. WAV saved at: {wav_path}]")


def get_rating(voice_id: str) -> dict:
    """Prompt user for ratings on each dimension."""
    ratings = {"voice_id": voice_id}
    for dim in RATING_DIMENSIONS:
        while True:
            raw = input(f"  Rate (1-5) {dim}: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 5:
                ratings[dim] = int(raw)
                break
            print("  Enter a number 1-5.")
    notes = input("  Notes (optional): ").strip()
    if notes:
        ratings["notes"] = notes
    ratings["total"] = sum(ratings[d] for d in RATING_DIMENSIONS)
    return ratings


def find_voice_in_catalog(catalog: dict, voice_id: str) -> dict | None:
    """Find a voice entry by ID."""
    for v in catalog["voices"]:
        if v["voice_id"] == voice_id:
            return v
    return None


def show_top(catalog: dict, n: int = 5) -> None:
    """Show top-rated voices."""
    rated = [v for v in catalog["voices"] if "total" in v]
    rated.sort(key=lambda v: v["total"], reverse=True)
    print(f"\n--- Top {min(n, len(rated))} Voices ---")
    for i, v in enumerate(rated[:n], 1):
        vtype = v.get("type", "?")
        scores = "/".join(str(v.get(d, "?")) for d in RATING_DIMENSIONS)
        print(f"  {i}. [{v['voice_id']}] ({vtype}) total={v['total']} ({scores}) {v.get('notes', '')}")
    print()


def pick_voice(catalog: dict, voice_id: str) -> None:
    """Set a voice as active."""
    entry = find_voice_in_catalog(catalog, voice_id)
    if entry is None:
        print(f"  Voice '{voice_id}' not found in catalog.")
        return

    catalog["active_voice"] = voice_id

    pt_path = VOICES_DIR / f"{voice_id}.pt"
    active_path = VOICES_DIR / "active_voice.pt"
    if pt_path.exists():
        shutil.copy2(pt_path, active_path)
        print(f"\n  Active voice set to: {voice_id}")
        print(f"  Embedding saved to: {active_path}")
        print(f"  Set in .env: SILERO_TTS_SPEAKER_PT={active_path}")
    else:
        print(f"\n  Active voice set to native speaker: {voice_id}")
        print(f"  Set in .env: SILERO_TTS_SPEAKER={voice_id}")
        if active_path.exists():
            active_path.unlink()

    save_catalog(catalog)


def run_native_voices(model, catalog: dict, text: str) -> None:
    """Rate all native voices that have not been rated yet."""
    unrated = [v for v in NATIVE_VOICES if find_voice_in_catalog(catalog, f"native_{v}") is None]
    if not unrated:
        print("All native voices already rated. Skipping.\n")
        return

    print(f"\n--- Native Voices ({len(unrated)} unrated) ---\n")
    for i, speaker in enumerate(unrated, 1):
        voice_id = f"native_{speaker}"
        print(f"[{i}/{len(unrated)}] {speaker} ... ", end="", flush=True)

        wav_path = VOICES_DIR / f"{voice_id}.wav"
        if not wav_path.exists():
            pcm16, sr = synthesize(model, text, speaker=speaker)
            save_wav(pcm16, sr, wav_path)

        print("playing")
        play_audio(wav_path)

        ratings = get_rating(voice_id)
        ratings["type"] = "native"
        ratings["speaker"] = speaker
        catalog["voices"].append(ratings)
        save_catalog(catalog)
        print()


def run_random_voices(model, catalog: dict, text: str, batch_size: int) -> None:
    """Generate and rate random voices in batches."""
    batch_num = 1
    existing_random = [v for v in catalog["voices"] if v.get("type") == "random"]
    total_generated = len(existing_random)

    while True:
        print(f"\n--- Random Voices (batch {batch_num}, generating {batch_size}) ---\n")

        for i in range(batch_size):
            voice_id = f"random_{total_generated + i + 1:04d}"
            print(f"[{total_generated + i + 1}] {voice_id} ... ", end="", flush=True)

            embedding = model.sample_random_speaker()
            pt_path = VOICES_DIR / f"{voice_id}.pt"
            torch.save(embedding, pt_path)

            wav_path = VOICES_DIR / f"{voice_id}.wav"
            pcm16, sr = synthesize(model, text, speaker_embedding=embedding)
            save_wav(pcm16, sr, wav_path)

            print("playing")
            play_audio(wav_path)

            ratings = get_rating(voice_id)
            ratings["type"] = "random"
            ratings["pt_file"] = str(pt_path)
            catalog["voices"].append(ratings)
            save_catalog(catalog)
            print()

        total_generated += batch_size
        batch_num += 1

        show_top(catalog)

        while True:
            cmd = input("Continue? [y]es / [n]o / [pick] ID / [top] / [replay] ID: ").strip().lower()
            if cmd in ("y", "yes", ""):
                break
            elif cmd in ("n", "no", "quit", "q"):
                return
            elif cmd.startswith("pick"):
                parts = cmd.split()
                if len(parts) == 2:
                    pick_voice(catalog, parts[1])
                else:
                    rated = [v for v in catalog["voices"] if "total" in v]
                    if rated:
                        best = max(rated, key=lambda v: v["total"])
                        pick_voice(catalog, best["voice_id"])
                continue
            elif cmd == "top":
                show_top(catalog, 10)
                continue
            elif cmd.startswith("replay"):
                parts = cmd.split()
                if len(parts) == 2:
                    rid = parts[1]
                    wav = VOICES_DIR / f"{rid}.wav"
                    if wav.exists():
                        play_audio(wav)
                    else:
                        print(f"  No WAV found for {rid}")
                continue
            else:
                print("  Unknown command. Try: y, n, pick <id>, top, replay <id>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Lab: audition Silero voices")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Test sentence to synthesize")
    parser.add_argument("--batch-size", type=int, default=5, help="Random voices per batch")
    args = parser.parse_args()

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Voice Lab ===")
    print(f'Test sentence: "{args.text}"\n')
    print("Loading Silero TTS model...")
    model = load_silero_model()
    print("Model loaded.\n")

    catalog = load_catalog()

    run_native_voices(model, catalog, args.text)
    run_random_voices(model, catalog, args.text, args.batch_size)

    show_top(catalog)
    print("Done. Run again to continue rating or generate more voices.")


if __name__ == "__main__":
    main()
