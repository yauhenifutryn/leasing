"""Mock Silero TTS server for local SIP testing. Returns a short sine tone."""
import math
import struct

from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()


def _generate_tone(freq: int = 440, duration_ms: int = 500, sample_rate: int = 24000) -> bytes:
    """Generate a simple sine wave tone as PCM16."""
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = int(16000 * math.sin(2 * math.pi * freq * t))
        samples.append(max(-32768, min(32767, value)))
    return struct.pack(f"<{n_samples}h", *samples)


@app.get("/tts")
async def tts(text: str = "", speaker: str = "xenia"):
    audio = _generate_tone(440, min(len(text) * 30, 3000))
    return Response(content=audio, media_type="audio/pcm")


@app.post("/tts")
async def tts_post(text: str = "", speaker: str = "xenia"):
    audio = _generate_tone(440, min(len(text) * 30, 3000))
    return Response(content=audio, media_type="audio/pcm")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50006)
