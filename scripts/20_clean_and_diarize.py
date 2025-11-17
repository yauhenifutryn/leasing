import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]

try:
    import torchaudio  # type: ignore
    if not hasattr(torchaudio, "set_audio_backend"):
        def _noop_set_audio_backend(_backend: str) -> None:
            return None
        torchaudio.set_audio_backend = _noop_set_audio_backend  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        def _noop_get_audio_backend() -> str:
            return "soundfile"
        torchaudio.get_audio_backend = _noop_get_audio_backend  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "list_audio_backends"):
        def _noop_list_audio_backends() -> list[str]:
            return ["soundfile"]
        torchaudio.list_audio_backends = _noop_list_audio_backends  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

import whisperx

from utils import read_json, write_json, normalize_text, should_skip

# Maps speakers (SPEAKER_00/01) to roles (agent/client) using heuristics.
# If you want true diarization, enable pyannote via WhisperX DiarizationPipeline.

AGENT_HEURISTICS = [
    "чем могу помочь",
    "добрый день",
    "компания",
    "здравствуйте, вы позвонили",
    "назовите, пожалуйста, номер договора",
    "секунду, я проверю",
]

def guess_role(text_segment: str) -> str | None:
    lower_text = text_segment.lower()
    if any(key_phrase in lower_text for key_phrase in AGENT_HEURISTICS):
        return "agent"
    return None

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enable_diarization", action="store_true", help="Use pyannote via WhisperX")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    diarizer = None
    if args.enable_diarization:
        from os import getenv

        diarizer = whisperx.DiarizationPipeline(use_auth_token=getenv("HUGGINGFACE_TOKEN"), device=args.device)

    for file_path in tqdm(sorted(in_dir.glob("*.json")), desc="Cleaning/Diarizing"):
        raw = read_json(file_path)
        audio_fp = None
        segments = raw.get("segments", [])

        if diarizer:
            name = file_path.stem.split(".")[0]
            for ext in [".wav", ".mp3", ".m4a", ".flac"]:
                candidate = Path("audio") / f"{name}{ext}"
                if candidate.exists():
                    audio_fp = str(candidate)
                    break
            if audio_fp:
                diarization_segments = diarizer(audio_fp)
                aligned = whisperx.align(raw["segments"], diarization_segments)
                segments = aligned

        conversation_id = file_path.stem.replace(".whisperx", "").replace(".whisper", "")
        out_path = out_dir / f"{conversation_id}.json"
        if should_skip(out_path, args.overwrite):
            continue

        conversation = {
            "conversation_id": conversation_id,
            "segments": [],
        }

        for segment in segments:
            text = normalize_text(segment.get("text", ""))
            if not text:
                continue
            speaker = segment.get("speaker") or "SPEAKER_00"
            role = segment.get("role") or guess_role(text)
            conversation["segments"].append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "speaker": speaker,
                    "role": role,
                    "text": text,
                }
            )

        write_json(out_path, conversation)

if __name__ == "__main__":
    main()
