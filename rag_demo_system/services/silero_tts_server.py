from __future__ import annotations

import base64
import os
import subprocess
import tempfile

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def _rubberband_time_stretch(audio_np: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Speed up audio by `rate` factor (e.g. 1.15 = 15% faster) using the
    rubberband CLI binary directly. Pitch preserved.

    Why we shell out instead of using pyrubberband: pyrubberband 0.3.0
    (latest on PyPI) imports the removed `imp` module in setup.py and
    fails to install on Python 3.12. We use exactly the same underlying
    binary; pyrubberband was only a ~50-line wrapper around the same
    subprocess call.

    Args:
        audio_np: float32 mono PCM samples in [-1, 1]
        sr:       sample rate in Hz
        rate:     speedup factor; > 1 = faster, < 1 = slower

    Returns:
        float32 mono PCM samples at the same sample rate, with duration
        compressed by `rate`. On rubberband-cli failure (missing binary,
        invalid input), returns the original audio unchanged so synth
        never crashes mid-call.
    """
    import soundfile as sf

    # rubberband -t expects a DURATION ratio (output_duration / input_duration).
    # We want to make the output 1/rate as long as the input.
    time_ratio = 1.0 / rate

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f:
        in_path = in_f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
        out_path = out_f.name

    try:
        sf.write(in_path, audio_np, sr, subtype="FLOAT")
        # Flags tuned for speech (Rubberband 2.0+, R3 engine default):
        #   -F  preserve formants — keeps voice timbre stable, prevents
        #       the "underwater / hollow" coloration that plain time-
        #       stretching can introduce on vowels.
        #   --crisp 6  highest transient sharpness preset — keeps
        #              consonants (т, ц, ч, ш, щ, к) from smearing into
        #              the echo-like artifact reported on the librosa
        #              version. Default is 5; 6 is recommended for speech.
        result = subprocess.run(
            [
                "rubberband",
                "-t", f"{time_ratio:.6f}",
                "-F",
                "--crisp", "6",
                in_path, out_path,
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(
                f"[silero_tts] rubberband failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:200]}",
                flush=True,
            )
            return audio_np
        out_audio, _ = sf.read(out_path, dtype="float32")
        return out_audio
    except FileNotFoundError:
        # rubberband-cli not installed. Fall back to original audio.
        print(
            "[silero_tts] rubberband-cli binary not found on PATH — "
            "speedup disabled this call. Run: apt install rubberband-cli",
            flush=True,
        )
        return audio_np
    except Exception as exc:  # noqa: BLE001
        print(f"[silero_tts] rubberband error: {exc}", flush=True)
        return audio_np
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


class SpeakRequest(BaseModel):
    text: str
    session_id: str
    language: str = "ru"


class SileroTTSSynthesizer:
    def __init__(
        self, speaker: str, sample_rate: int, model_variant: str = "v5_4_ru", speaker_pt: str | None = None
    ) -> None:
        if model_variant.startswith("v5"):
            from silero import silero_tts
            self._model, _ = silero_tts(language="ru", speaker=model_variant)
        else:
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="ru",
                speaker=model_variant,
                trust_repo=True,
            )
        self._model.to(torch.device("cpu"))
        self._speaker = speaker
        self._speaker_embedding: torch.Tensor | None = None
        self._sample_rate = sample_rate

        if speaker_pt and os.path.isfile(speaker_pt):
            self._speaker_embedding = torch.load(speaker_pt, map_location="cpu")
            print(f"[silero_tts] Loaded custom voice from {speaker_pt}")

    def synthesize(self, text: str) -> tuple[bytes, int]:
        rate_pct = int(os.getenv("SILERO_TTS_RATE_PCT") or "100")
        speaker_arg = (
            self._speaker_embedding if self._speaker_embedding is not None else self._speaker
        )
        common_kwargs = {
            "speaker": speaker_arg,
            "sample_rate": self._sample_rate,
            "put_accent": True,
            "put_yo": True,
        }
        # Silero v5_4_ru's SSML <prosody rate> is parsed but silently ignored
        # (verified live 2026-04-28: 100% and 120% produced byte-for-byte
        # identical audio).
        #
        # Speed-up algorithm history on this codebase:
        #   1. np.interp (1cbbee7+9caf11b) — fast but raises pitch ~2 semitones
        #      on speech. Rejected: "chipmunk voice" 2026-04-29.
        #   2. librosa.effects.time_stretch (e981e87) — phase vocoder preserves
        #      pitch but introduces echo/reverb artifacts on Russian sibilants
        #      and fricatives. Rejected: "weird echo" 2026-04-29.
        #   3. pyrubberband (THIS commit) — wraps the Rubberband C library
        #      that's used in professional DAWs (and is the same DSP class
        #      browsers use for HTMLMediaElement.playbackRate). Best open-
        #      source quality available for speech time-stretch. Cost:
        #      requires `apt install rubberband-cli` on the server (handled
        #      by provision_server.sh).
        audio = self._model.apply_tts(text=text, **common_kwargs)
        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        if rate_pct != 100 and len(audio_np) > 1:
            # Direct rubberband-cli subprocess (see _rubberband_time_stretch
            # docstring for why we don't use pyrubberband). Per-phrase
            # overhead: ~50-100ms for fork + WAV I/O + DSP. Falls back to
            # original audio if the binary is missing or fails.
            rate = rate_pct / 100.0
            audio_np = _rubberband_time_stretch(audio_np, self._sample_rate, rate)
        pcm16 = (audio_np * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
        return pcm16, self._sample_rate


def create_app(synthesizer: SileroTTSSynthesizer) -> FastAPI:
    app = FastAPI(title="Silero TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        voice_source = (
            "custom_pt" if synthesizer._speaker_embedding is not None else "native"
        )
        return {
            "ok": True,
            "provider": "silero_tts",
            "speaker": synthesizer._speaker,
            "voice_source": voice_source,
            "rate_pct": int(os.getenv("SILERO_TTS_RATE_PCT") or "100"),
        }

    @app.post("/speak")
    async def speak(payload: SpeakRequest) -> dict[str, object]:
        audio_bytes, sample_rate_hz = synthesizer.synthesize(payload.text)
        return {
            "ok": True,
            "provider": "silero_tts",
            "session_id": payload.session_id,
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate_hz": sample_rate_hz,
        }

    return app


def create_unavailable_app(reason: str) -> FastAPI:
    app = FastAPI(title="Silero TTS Service")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": False, "provider": "silero_tts", "reason": reason}

    @app.post("/speak")
    async def speak(_: SpeakRequest) -> dict[str, object]:
        raise HTTPException(status_code=503, detail=reason)

    return app


def _build_default_app() -> FastAPI:
    speaker = (os.getenv("SILERO_TTS_SPEAKER") or "xenia").strip()
    model_variant = (os.getenv("SILERO_TTS_MODEL") or "v5_4_ru").strip()
    sample_rate = int(os.getenv("SILERO_TTS_SAMPLE_RATE") or "24000")
    speaker_pt = os.getenv("SILERO_TTS_SPEAKER_PT")
    try:
        synthesizer = SileroTTSSynthesizer(speaker, sample_rate, model_variant=model_variant, speaker_pt=speaker_pt)
    except Exception as exc:  # noqa: BLE001
        return create_unavailable_app(f"silero_tts_not_ready: {exc}")
    return create_app(synthesizer)


app = _build_default_app()
