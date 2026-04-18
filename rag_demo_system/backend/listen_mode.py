"""Listen_mode auto-exit: per-session asyncio task that emits "Слушаю Вас."

When the bot enters listen_mode, this task sleeps until listen_mode_until
then checks whether the mode is still active. If yes, it synthesizes a
"Слушаю Вас." TTS prompt and clears the flag. If the client speaks first,
the normal turn-taking path clears listen_mode; the task wakes, sees the
flag is false, and exits silently.

Transport-agnostic: uses the existing `send_json` + `response.output_audio.delta`
contract, which is handled natively by the web WebSocket client and by the
``_JambonzWebSocketShim`` (decodes base64 -> raw PCM frames).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

WAKE_PROMPT = "Слушаю Вас."


def spawn_auto_exit_task(
    session: Any,
    websocket: Any,
    session_id: str,
) -> asyncio.Task:
    """Spawn and return the auto-exit task.

    Caller stores the returned task on ``session.listen_mode_task`` so that
    session teardown can cancel it.
    """
    return asyncio.create_task(_auto_exit_loop(session, websocket, session_id))


async def _auto_exit_loop(session: Any, websocket: Any, session_id: str) -> None:
    try:
        # Sleep until listen_mode_until. Re-check flag on wake.
        now = time.time()
        sleep_for = max(0.0, session.listen_mode_until - now)
        await asyncio.sleep(sleep_for)

        # Check whether we still need to fire.
        if not getattr(session, "listen_mode", False):
            return
        if time.time() < session.listen_mode_until:
            # Mode was extended while we slept; exit. A new task will be
            # spawned by the hybrid gate when re-entry happens.
            return

        # Clear flags BEFORE emitting so re-entry is possible AND so the
        # audio chunks actually play. Note: session.interrupted was set True
        # when listen_mode was entered (to kill the then-current TTS). The
        # Jambonz shim's chunk loop (since Fix 6) honors that flag and will
        # drop our "Слушаю Вас" audio if we don't reset it here.
        session.listen_mode = False
        session.interrupted = False
        session.assistant_speaking = True  # we're about to speak

        # Emit "Слушаю Вас." through the same inline TTS pattern used by the
        # rest of the voice path: clean_voice_output -> synthesize_audio ->
        # send_json (response.output_audio.delta). This contract is handled
        # by both the browser client and the Jambonz shim.
        from .text_utils import clean_voice_output
        from .voice_adapters import synthesize_audio

        text = clean_voice_output(WAKE_PROMPT)
        if not text:
            return

        # Emit text delta first (mirrors _send_tts_message / _stream_voice_response).
        try:
            await websocket.send_json({
                "type": "response.output_text.delta",
                "session_id": session_id,
                "delta": text,
            })
        except Exception as _text_exc:
            print(
                f"[listen_mode] text delta send failed session={session_id[:8]}: {_text_exc}",
                flush=True,
            )

        # Synthesize TTS off the event loop.
        audio_resp = await asyncio.to_thread(synthesize_audio, text, session_id)
        audio_b64 = audio_resp.get("audio_b64") or ""
        if audio_b64:
            try:
                await websocket.send_json({
                    "type": "response.output_audio.delta",
                    "session_id": session_id,
                    "delta": audio_b64,
                    "sample_rate_hz": audio_resp.get("sample_rate_hz"),
                })
            except Exception as _audio_exc:
                print(
                    f"[listen_mode] audio delta send failed session={session_id[:8]}: {_audio_exc}",
                    flush=True,
                )

        # Final response.done so both transports know the turn ended.
        try:
            await websocket.send_json({
                "type": "response.done",
                "session_id": session_id,
                "backend": getattr(session, "backend", ""),
                "used_knowledge": [],
                "citations": [],
                "timings": {},
            })
        except Exception:
            pass

        # Mark us as done speaking so barge-in / VAD work correctly on the next turn.
        try:
            session.assistant_speaking = False
        except Exception:  # noqa: BLE001
            pass

        print(
            f"[listen_mode] auto-exit fired for session={session_id[:8]}, emitted prompt",
            flush=True,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(
            f"[listen_mode] auto-exit error session={session_id[:8]}: {exc}",
            flush=True,
        )
