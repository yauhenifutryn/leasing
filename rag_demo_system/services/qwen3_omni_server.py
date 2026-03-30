from __future__ import annotations

import base64
import io
import os
import tempfile
import time

from fastapi import FastAPI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("QWEN3_OMNI_MODEL_PATH", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
SPEAKER = os.getenv("QWEN3_OMNI_SPEAKER", "Chelsie")
ATTN_IMPL = os.getenv("QWEN3_OMNI_ATTN_IMPL", "sdpa")

# Grounding system prompt in Russian (D-05). Uses {context_block} placeholder.
SYSTEM_PROMPT_TEMPLATE = (
    "Вы — голосовой ассистент компании Mikro Leasing. "
    "Отвечайте СТРОГО на основе предоставленного контекста. "
    "Не используйте знания, не содержащиеся в контексте. "
    "Если информация по вопросу отсутствует в контексте, "
    "скажите: 'Извините, у меня нет информации по этому вопросу.'\n\n"
    "Контекст:\n{context_block}"
)


# ---------------------------------------------------------------------------
# Pydantic request / response models (D-09)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    audio_b64: str
    context_chunks: list[str]
    system_prompt: str = ""


class ChatResponse(BaseModel):
    audio_b64: str
    text: str
    sample_rate_hz: int = 24000
    t_omni_first_audio: float


# ---------------------------------------------------------------------------
# Inference class (D-08: transformers, not vLLM; deferred imports)
# ---------------------------------------------------------------------------


class Qwen3OmniInference:
    """Wraps Qwen3-Omni-30B-A3B-Instruct for audio-in, audio+text-out inference.

    Deferred imports of transformers classes keep module-level import errors
    contained inside _build_default_app() so that test code can import this
    module without a GPU or transformers==4.57.3 installed.
    """

    def __init__(self, model_path: str, device: str, attn_impl: str) -> None:
        # Deferred import: avoids load-time failure in environments without
        # transformers==4.57.3 (e.g., the shared .venv pinned to 4.37.2).
        from transformers import (  # type: ignore[import]
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )
        from qwen_omni_utils import process_mm_info  # type: ignore[import]

        self._process_mm_info = process_mm_info

        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            attn_implementation=attn_impl,
        )
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    def chat(self, req: ChatRequest) -> ChatResponse:
        """Run a single Omni voice turn.

        Per Pitfall 2: no batching -- one request at a time.
        Per Pitfall 3: always include grounding system prompt (D-03).
        Per Pitfall 6: always clean up the temporary WAV file.
        """
        import soundfile as sf  # type: ignore[import]

        # Determine grounding system prompt (D-05)
        if req.system_prompt:
            system_text = req.system_prompt
        else:
            context_block = "\n\n".join(req.context_chunks)
            system_text = SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block)

        # Materialise base64 audio to a temporary WAV file for process_mm_info
        # (Pitfall 6: always delete, even on exception)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                tmp_path = tmp_f.name
                import wave
                pcm_data = base64.b64decode(req.audio_b64)
                with wave.open(tmp_f, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(pcm_data)

            # Build conversation in the Qwen3-Omni multi-modal format
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_text}],
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": tmp_path}],
                },
            ]

            # Tokenise + prepare multimodal inputs
            text_tmpl = self._processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            audios, images, videos = self._process_mm_info(
                conversation, use_audio_in_video=False
            )
            inputs = self._processor(
                text=text_tmpl,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            # Move all tensor inputs to the model's device and dtype
            inputs = inputs.to(self._model.device).to(self._model.dtype)

            # Run generation (thinker-talker; D-06: audio output collapses
            # llm_first_token == tts_first_chunk to the same timestamp)
            text_ids, audio_tensor = self._model.generate(
                **inputs,
                speaker=SPEAKER,
                thinker_return_dict_in_generate=True,
                return_audio=True,
                use_audio_in_video=False,
            )

            # Record first-audio timestamp immediately after generate() (D-06)
            t_omni_first_audio = time.time()

        finally:
            # Always clean up temp file (Pitfall 6)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Decode generated text tokens
        answer_text: str = self._processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Convert audio tensor to PCM16 WAV and base64-encode it
        audio_np = audio_tensor.reshape(-1).detach().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, samplerate=24000, format="WAV", subtype="PCM_16")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return ChatResponse(
            audio_b64=audio_b64,
            text=answer_text,
            sample_rate_hz=24000,
            t_omni_first_audio=t_omni_first_audio,
        )


# ---------------------------------------------------------------------------
# FastAPI app factory (D-09, D-10)
# ---------------------------------------------------------------------------


def create_app(inference: Qwen3OmniInference) -> FastAPI:
    """Create the FastAPI application wired to the given inference instance."""
    app_instance = FastAPI(title="Qwen3 Omni Sidecar")

    @app_instance.get("/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "provider": "qwen3_omni"}

    @app_instance.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        return inference.chat(req)

    return app_instance


# ---------------------------------------------------------------------------
# Default app (called at module level for uvicorn)
# ---------------------------------------------------------------------------


def _build_default_app() -> FastAPI:
    print(
        f"[qwen3_omni_server] Loading model: {MODEL_PATH} "
        f"| speaker: {SPEAKER} | attn: {ATTN_IMPL}"
    )
    inference = Qwen3OmniInference(MODEL_PATH, "auto", ATTN_IMPL)
    return create_app(inference)


app = _build_default_app()
