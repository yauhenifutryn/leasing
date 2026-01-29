# Yandex Realtime Voice Demo

Minimal realtime voice assistant prototype using Yandex AI Studio Realtime API. The browser streams PCM16 24 kHz audio to the backend over WebSocket, the backend relays to Yandex, and audio plus text deltas are streamed back to the UI for playback and display.

## Requirements
- Node.js 20
- Yandex Cloud folder and API credentials

## Setup

```bash
cd experiments/yandex_realtime_voice
cp .env.example .env
npm install
npm run dev
```

Open `http://localhost:8787`.

## Environment variables

- `YC_FOLDER_ID` Required.
- `YC_API_KEY` or `YC_IAM_TOKEN` Required, choose one.
- `YC_VECTOR_STORE_ID` Optional, enables file_search tool for RAG.
- `YC_VOICE` Optional, default `ermil`.
- `YC_MODEL` Optional, default `gpt://<folder_ID>/speech-realtime-250923`.
- `YC_REALTIME_WS_URL` Optional, overrides Realtime WebSocket URL.
- `PORT` Optional, default 8787.

## Usage
1. Click Connect.
2. Hold to Talk, speak, then release.
3. Watch transcript and assistant text update, hear streaming audio response.

## Notes
- Use headphones to avoid feedback loops.
- If you pass `YC_VECTOR_STORE_ID`, the assistant can use file search for RAG. If not set, tools are disabled.
- The UI is intentionally minimal to keep latency low.
