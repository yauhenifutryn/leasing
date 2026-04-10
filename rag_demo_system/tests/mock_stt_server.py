"""Mock Whisper STT server for local SIP testing. Returns hardcoded transcription."""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    size = len(audio_bytes)
    return JSONResponse({
        "text": "Здравствуйте, сколько стоит лизинг на Тигуан?",
        "language": "ru",
        "duration": round(size / 32000, 2),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=50002)
