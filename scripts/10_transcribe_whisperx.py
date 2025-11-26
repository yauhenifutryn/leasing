import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import whisperx
from whisperx.diarize import DiarizationPipeline

from utils import list_audio, write_json, should_skip


AGENT_HINTS = (
    "компания микролизинг",
    "чем могу помочь",
    "благодарим вас",
    "оставайтесь на линии",
)

CLIENT_HINTS = (
    "у меня вопрос",
    "хотим приобрести",
    "интересует",
    "подскажите",
)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray):
        return float(value.tolist())
    return value  # type: ignore[return-value]


def guess_role(text: str, speaker: str, cache: Dict[str, str]) -> str:
    if speaker in cache:
        return cache[speaker]

    lowered = text.lower()
    if any(key in lowered for key in AGENT_HINTS):
        cache[speaker] = "agent"
    elif any(key in lowered for key in CLIENT_HINTS):
        cache[speaker] = "client"
    else:
        if "agent" not in cache.values():
            cache[speaker] = "agent"
        elif "client" not in cache.values():
            cache[speaker] = "client"
        else:
            cache[speaker] = "other"
    return cache[speaker]


def merge_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        if merged[-1]["speaker"] == segment["speaker"]:
            merged[-1]["end"] = segment["end"]
            merged[-1]["text"] += f" {segment['text']}"
        else:
            merged.append(segment)
    return merged


def collect_segments(aligned_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    role_map: Dict[str, str] = {}
    cleaned: List[Dict[str, Any]] = []
    for idx, segment in enumerate(aligned_segments):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        speaker = segment.get("speaker") or segment.get("speaker_id") or "SPEAKER_00"
        role = guess_role(text, str(speaker), role_map)
        cleaned.append(
            {
                "id": idx,
                "start": as_float(segment.get("start")),
                "end": as_float(segment.get("end")),
                "speaker": str(speaker),
                "role": role,
                "text": text,
            }
        )
    return merge_segments(cleaned)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--align-device", default=None)
    parser.add_argument("--diarization-device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--hf-token", default=os.getenv("HUGGINGFACE_TOKEN", ""))
    parser.add_argument("--disable-diarization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    align_device = args.align_device or args.device
    diar_device = args.diarization_device or args.device
    hf_token = (args.hf_token or "").strip()

    model = whisperx.load_model(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        vad_model="silero_vad",
        vad_device="cpu",
    )

    align_model = None
    metadata = None
    current_align_language = None
    diar_pipeline = None

    audio_files = sorted(list_audio(in_dir))
    for audio_path in tqdm(audio_files, desc="WhisperX transcribe"):
        out_path = out_dir / f"{audio_path.stem}.whisperx.json"
        if should_skip(out_path, args.overwrite):
            continue

        try:
            transcription = model.transcribe(
                str(audio_path),
                batch_size=args.batch_size,
                language=args.language,
                task="transcribe",
            )

            segments = transcription["segments"]
            language = transcription.get("language", args.language)

            if current_align_language != language:
                align_model, metadata = whisperx.load_align_model(
                    language_code=language,
                    device=align_device,
                )
                current_align_language = language

            aligned = whisperx.align(
                segments,
                align_model,
                metadata,
                str(audio_path),
                align_device,
                return_char_alignments=False,
            )

            if hf_token and not args.disable_diarization:
                if diar_pipeline is None:
                    diar_pipeline = DiarizationPipeline(
                        use_auth_token=hf_token,
                        device=diar_device,
                    )
                diar_segments = diar_pipeline(str(audio_path))
                aligned = whisperx.assign_word_speakers(diar_segments, aligned)

            cleaned_segments = collect_segments(aligned["segments"])
            result = {
                "conversation_id": audio_path.stem,
                "task": "transcribe",
                "language": language,
                "segments": cleaned_segments,
                "text": " ".join(seg["text"] for seg in cleaned_segments),
            }
            write_json(out_path, result)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {audio_path.name}: {exc}")


if __name__ == "__main__":
    main()
