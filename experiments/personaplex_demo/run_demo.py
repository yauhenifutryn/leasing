#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_audio(path: Path):
    import soundfile as sf

    data, sr = sf.read(path)
    return data, sr


def _record_mic(seconds: int, sample_rate: int):
    import sounddevice as sd
    import numpy as np

    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze(), sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="PersonaPlex demo wrapper")
    parser.add_argument("--model", default="nvidia/personaplex-7b-v1")
    parser.add_argument("--text", default="")
    parser.add_argument("--audio", default="")
    parser.add_argument("--mic", type=int, default=0, help="record N seconds")
    parser.add_argument("--out", default="out.wav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.text and not args.audio and args.mic <= 0:
        print("Provide --text, --audio, or --mic", file=sys.stderr)
        return 2

    try:
        import torch
        from transformers import AutoProcessor, AutoModel
    except Exception as exc:
        print(f"Missing deps: {exc}", file=sys.stderr)
        return 1

    audio = None
    sr = None
    if args.audio:
        audio, sr = _load_audio(Path(args.audio))
    elif args.mic > 0:
        audio, sr = _record_mic(args.mic, args.sample_rate)

    try:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        model = model.to(args.device)
        model.eval()
    except Exception as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 1

    inputs = {}
    if args.text:
        inputs.update(processor(text=args.text, return_tensors="pt"))
    if audio is not None:
        if sr is None:
            sr = args.sample_rate
        inputs.update(processor(audio=audio, sampling_rate=sr, return_tensors="pt"))

    for key, value in list(inputs.items()):
        if hasattr(value, "to"):
            inputs[key] = value.to(args.device)

    try:
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_tokens)
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    text_out = None
    audio_out = None

    if isinstance(output, dict):
        text_out = output.get("text")
        audio_out = output.get("audio") or output.get("speech")
    else:
        try:
            text_out = processor.decode(output[0])
        except Exception:
            text_out = None

    if audio_out is not None:
        import soundfile as sf

        sf.write(args.out, audio_out, args.sample_rate)
        print(f"Saved audio to {args.out}")
    if text_out:
        print(text_out)

    if audio_out is None and not text_out:
        print("No output produced. Check upstream model API.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
