#!/usr/bin/env python3
"""Automated voice/chat regression harness.

Two modes:

  --mode=chat   (working today)
      Drives /api/text-turn end-to-end, scenario-by-scenario. Captures
      total per-turn latency + bot reply. Scores against a rubric (regex
      list of substrings expected / forbidden in bot reply). Writes a
      CSV + readable report.

  --mode=audio  (scaffolded, see TODO blocks)
      Same scenarios, but driven through the live SIP path:
        1. Connect to /ws/jambonz (subprotocol ws.jambonz.org)
        2. Send session:new — receive ack with audio_ws_url
        3. Connect to /ws/jambonz-audio
        4. For each scenario turn:
             a. Synthesize user speech via Silero TTS (POST /speak)
                — must resample 24kHz→16kHz mono PCM
             b. Stream PCM frames over the audio WS (320 bytes per
                20ms at 16kHz), measure VAD speech_end → first bot
                audio out
             c. Buffer bot's 24kHz response, save WAV
             d. Transcribe with Whisper service for rubric scoring
        5. Send call:status=completed, close.

Goal of the harness: kill the manual "call via Zoiper, listen, judge"
cycle. Each run produces:
  - per-stage latency: STT, classifier, LLM brain, TTS first-chunk,
    total speech-end-to-first-audio-out
  - barge-in accuracy: how often bot stopped speaking when caller
    started talking mid-response (audio mode only)
  - rubric pass/fail per turn (substring match in bot reply)

Scenario format (YAML):
  - name: usd_calc_to_byn_switch
    seed_phone: "+375296838707"
    turns:
      - say: "Сергей."
        expect: ["Сергей"]
      - say: "Хочу взять в лизинг машину за 100 тысяч рублей."
        expect: ["100", "BYN"]
        forbid: ["USD"]
      - say: "Срок 36 месяцев."
        expect: ["36"]

Usage:
  python scripts/voice_harness.py \
      --mode=chat --base-url=http://localhost:8000 \
      --scenarios=tests/scenarios/smoke.yaml \
      --report=.state/harness_report.txt
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import sys
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    import websockets  # type: ignore
except ImportError:
    websockets = None  # type: ignore

try:
    from scipy.signal import resample_poly  # type: ignore
except ImportError:
    resample_poly = None  # type: ignore


@dataclass
class TurnResult:
    turn_idx: int
    user_msg: str
    bot_reply: str
    latency_ms: int
    expect_pass: list[str] = field(default_factory=list)
    expect_fail: list[str] = field(default_factory=list)
    forbid_violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.expect_fail and not self.forbid_violations


@dataclass
class ScenarioResult:
    name: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.turns)

    @property
    def total_latency_ms(self) -> int:
        return sum(t.latency_ms for t in self.turns)


def _post_text_turn(base_url: str, session_id: str, message: str, phone: str) -> tuple[str, int]:
    """Hit /api/text-turn. Returns (bot_reply_text, ms)."""
    url = f"{base_url.rstrip('/')}/api/text-turn"
    payload = json.dumps({
        "session_id": session_id,
        "message": message,
        "phone": phone,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ms = int((time.monotonic() - t0) * 1000)
    reply = data.get("reply") or data.get("text") or data.get("response") or ""
    return reply, ms


def _check_rubric(reply: str, expect: list[str], forbid: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Returns (matched_expects, missed_expects, forbid_violations)."""
    rl = reply.lower()
    matched = [e for e in expect if e.lower() in rl]
    missed = [e for e in expect if e.lower() not in rl]
    violations = [f for f in forbid if f.lower() in rl]
    return matched, missed, violations


def _post_chat_end(base_url: str, session_id: str) -> None:
    """Tell the backend to clear in-memory state + broadcast sip.call.end
    so the operator monitor doesn't replay this harness session forever
    on each reload (events linger in _sip_event_history until call.end
    arrives — see app.py:3404)."""
    payload = json.dumps({"session_id": session_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat/end",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as _resp:
            _resp.read()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


def run_chat_scenario(base_url: str, scenario: dict) -> ScenarioResult:
    name = scenario.get("name", "unnamed")
    phone = scenario.get("seed_phone") or "+375290000000"
    # session_id must match _VALID_SESSION_ID = ^chat-[A-Za-z0-9_-]{6,64}$
    # (defense against path traversal in chat_persistence — see app.py:138).
    sid = f"chat-h{uuid.uuid4().hex[:10]}"
    result = ScenarioResult(name=name)

    for idx, turn in enumerate(scenario.get("turns", []), start=1):
        user_msg = turn["say"]
        expect = turn.get("expect") or []
        forbid = turn.get("forbid") or []
        try:
            reply, ms = _post_text_turn(base_url, sid, user_msg, phone)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] turn {idx} HTTP error: {exc}", file=sys.stderr)
            reply, ms = f"<error: {exc}>", 0
        matched, missed, violations = _check_rubric(reply, expect, forbid)
        result.turns.append(TurnResult(
            turn_idx=idx,
            user_msg=user_msg,
            bot_reply=reply,
            latency_ms=ms,
            expect_pass=matched,
            expect_fail=missed,
            forbid_violations=violations,
        ))
    # Tear down: clear server-side state + monitor history.
    _post_chat_end(base_url, sid)
    return result


# --- Audio mode -----------------------------------------------------------
#
# Drives the live SIP path end-to-end without Zoiper. Talks to the same
# endpoints Jambonz feature-server hits in production:
#
#   /ws/jambonz       (subprotocol "ws.jambonz.org") — call control
#   /ws/jambonz-audio                                 — bidirectional audio
#
# The handlers' wire format was extracted from backend/app.py
# (jambonz_control_ws @ 2826, jambonz_audio_ws @ 2924, _speak_tts @ 174).
# The non-obvious bits the protocol-sketch in HANDOVER.md missed:
#
#   * The audio handler runs a DTMF consent gate BEFORE conversation
#     starts. Without sending {"type": "dtmf", "dtmf": "1"} on the audio
#     WS, the call disconnects after a 20s timer.
#   * Caller PCM is L16 16kHz mono; bot PCM comes back at 24kHz mono in
#     ~40ms (1920-byte) binary chunks via ws.send_bytes.
#   * Silero /speak returns 24kHz PCM16 base64-wrapped; we resample
#     24kHz → 16kHz before streaming as caller audio.


def _pcm_resample_24k_to_16k(pcm_24k: bytes) -> bytes:
    """24kHz PCM16 mono → 16kHz PCM16 mono.

    Inverse of the server's 16k→24k step at app.py:3210 (uses
    scipy.signal.resample_poly up=3 down=2). Here we go the other way:
    up=2, down=3.
    """
    if resample_poly is None:
        raise RuntimeError(
            "scipy not available; install scipy to use audio mode "
            "(or run on the server's .venv-voice-oss venv which has it)"
        )
    import numpy as np  # noqa: PLC0415  (scipy implies numpy)

    samples = np.frombuffer(pcm_24k, dtype=np.int16)
    out = resample_poly(samples, up=2, down=3).astype(np.int16)
    return out.tobytes()


def _silero_synth(silero_url: str, text: str) -> tuple[bytes, int]:
    """POST silero_url/speak → (pcm16 bytes, sample_rate_hz)."""
    payload = json.dumps({
        "text": text,
        "session_id": "harness",
        "language": "ru",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{silero_url.rstrip('/')}/speak",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"silero /speak failed: {data}")
    return base64.b64decode(data["audio_b64"]), int(data["sample_rate_hz"])


def _whisper_transcribe(whisper_url: str, pcm_bytes: bytes, sample_rate_hz: int) -> str:
    """POST whisper_url/transcribe → text."""
    payload = json.dumps({
        "audio_b64": base64.b64encode(pcm_bytes).decode("ascii"),
        "session_id": "harness",
        "language": "ru",
        "sample_rate_hz": sample_rate_hz,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{whisper_url.rstrip('/')}/transcribe",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("text") or "").strip()


async def _drain_bot_audio(
    audio_ws,
    silence_timeout: float,
    overall_timeout: float,
    deadline_first_frame: float = 8.0,
) -> tuple[bytes, float | None]:
    """Read bot audio frames until a silence_timeout gap of no inbound bytes.

    Waits up to deadline_first_frame for the first byte. After that,
    closes the segment once silence_timeout elapses with no further
    bytes. Hard cap at overall_timeout. Text frames (e.g. killAudio)
    are silently consumed.

    Returns (combined_pcm, first_byte_monotonic_or_None).
    """
    chunks: list[bytes] = []
    t_start = time.monotonic()
    last_byte_at: float | None = None
    first_byte_at: float | None = None
    while True:
        now = time.monotonic()
        if last_byte_at is None:
            wait = deadline_first_frame - (now - t_start)
        else:
            wait = silence_timeout - (now - last_byte_at)
        if wait <= 0:
            break
        if (now - t_start) > overall_timeout:
            break
        try:
            msg = await asyncio.wait_for(audio_ws.recv(), timeout=wait)
        except asyncio.TimeoutError:
            break
        if isinstance(msg, (bytes, bytearray)):
            chunks.append(bytes(msg))
            now2 = time.monotonic()
            if first_byte_at is None:
                first_byte_at = now2
            last_byte_at = now2
        # else: text control frame (killAudio, etc.) — ignore
    return b"".join(chunks), first_byte_at


async def run_audio_scenario(
    base_url: str,
    silero_url: str,
    whisper_url: str,
    scenario: dict,
    audio_dir: Path,
) -> ScenarioResult:
    """Drive a scenario through /ws/jambonz + /ws/jambonz-audio."""
    if websockets is None:
        raise RuntimeError(
            "websockets package not installed. "
            "Use the .venv-voice-oss venv on the server, or `pip install websockets`."
        )
    name = scenario.get("name", "unnamed")
    phone = scenario.get("seed_phone") or "+375290000000"
    call_sid = uuid.uuid4().hex
    result = ScenarioResult(name=name)

    ws_base = base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    ctrl_url = f"{ws_base}/ws/jambonz"
    # ws_authority is everything after "ws://" or "wss://" up to the path.
    ws_authority = ws_base.split("://", 1)[1].split("/", 1)[0]

    async with websockets.connect(ctrl_url, subprotocols=["ws.jambonz.org"]) as ctrl:
        # 1. session:new
        await ctrl.send(json.dumps({
            "type": "session:new",
            "msgid": f"m-{call_sid[:8]}",
            "call_sid": call_sid,
            "data": {
                "from": phone,
                "callerName": f"harness-{name}",
            },
        }))
        ack_raw = await asyncio.wait_for(ctrl.recv(), timeout=10)
        ack = json.loads(ack_raw)
        audio_url = ack["data"][0]["url"]
        # The server hands out host.docker.internal because it's the
        # callback URL Jambonz inside Docker uses. The harness reaches
        # the audio WS via whatever authority we used for control.
        audio_url = audio_url.replace("host.docker.internal:8000", ws_authority)
        audio_url = audio_url.replace("host.docker.internal", ws_authority.split(":")[0])

        async with websockets.connect(audio_url) as audio_ws:
            # 2. metadata frame
            await audio_ws.send(json.dumps({
                "callSid": call_sid,
                "sampleRate": 16000,
                "metadata": {"from": phone, "callSid": call_sid},
            }))

            # 3. Drain consent TTS, then send DTMF=1, then drain intro TTS.
            # The server sends consent ("Нажмите 1 для согласия...") TTS as
            # background task. We don't strictly need to drain it before
            # sending DTMF — the consent gate accepts DTMF any time during
            # the 20s window. Send DTMF early to minimise wall-clock waste.
            await asyncio.sleep(0.3)  # let metadata + consent task spin up
            await audio_ws.send(json.dumps({"type": "dtmf", "dtmf": "1"}))
            # Drain consent + intro TTS until we see SUSTAINED silence
            # (intro finished). Bot intro is ~6s of audio with ~1s gaps
            # between phrases — silence_timeout=1.0 was racing those gaps
            # and returning mid-intro, which let the first user utterance
            # arrive while the bot was still speaking → bot ignored it,
            # subsequent t1 reported "no audio received".
            await _drain_bot_audio(
                audio_ws,
                silence_timeout=2.5,
                overall_timeout=25.0,
                deadline_first_frame=10.0,
            )

            # 4. Per-turn loop
            audio_dir.mkdir(parents=True, exist_ok=True)
            for idx, turn in enumerate(scenario.get("turns", []), start=1):
                user_msg = turn["say"]
                expect = turn.get("expect") or []
                forbid = turn.get("forbid") or []

                # 4a. Synth user audio
                try:
                    pcm_24k, _sr = await asyncio.to_thread(_silero_synth, silero_url, user_msg)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [{name}] turn {idx} silero error: {exc}", file=sys.stderr)
                    result.turns.append(TurnResult(
                        turn_idx=idx, user_msg=user_msg,
                        bot_reply=f"<silero error: {exc}>", latency_ms=0,
                        expect_fail=expect, forbid_violations=[],
                    ))
                    continue

                pcm_16k = _pcm_resample_24k_to_16k(pcm_24k)

                # 4b. Stream as 20ms (640-byte) frames at realtime pacing.
                FRAME_BYTES = 640  # 20ms @ 16kHz int16 mono
                for i in range(0, len(pcm_16k), FRAME_BYTES):
                    frame = pcm_16k[i:i + FRAME_BYTES]
                    if len(frame) < FRAME_BYTES:
                        frame = frame + b"\x00" * (FRAME_BYTES - len(frame))
                    await audio_ws.send(frame)
                    await asyncio.sleep(0.02)
                # Send a half-second of silence so VAD's 900ms silence
                # timer (VAD_SILENCE_MS default) can latch speech_end.
                _silence_frame = b"\x00" * FRAME_BYTES
                for _ in range(50):  # 50 * 20ms = 1.0s
                    await audio_ws.send(_silence_frame)
                    await asyncio.sleep(0.02)
                t_send_done = time.monotonic()

                # 4c. Wait for bot reply.
                bot_pcm_24k, first_byte_at = await _drain_bot_audio(
                    audio_ws,
                    silence_timeout=0.8,
                    overall_timeout=30.0,
                    deadline_first_frame=12.0,
                )
                if first_byte_at is not None:
                    latency_ms = int((first_byte_at - t_send_done) * 1000)
                else:
                    latency_ms = -1  # no audio came back

                # 4d. Save WAV (24kHz)
                wav_path = audio_dir / f"{name}_t{idx}.wav"
                with wave.open(str(wav_path), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(24000)
                    w.writeframes(bot_pcm_24k)

                # 4e. Transcribe via Whisper
                if bot_pcm_24k:
                    try:
                        bot_reply = await asyncio.to_thread(
                            _whisper_transcribe, whisper_url, bot_pcm_24k, 24000,
                        )
                    except Exception as exc:  # noqa: BLE001
                        bot_reply = f"<whisper error: {exc}>"
                else:
                    bot_reply = "<no audio received>"

                matched, missed, violations = _check_rubric(bot_reply, expect, forbid)
                result.turns.append(TurnResult(
                    turn_idx=idx,
                    user_msg=user_msg,
                    bot_reply=bot_reply,
                    latency_ms=latency_ms,
                    expect_pass=matched,
                    expect_fail=missed,
                    forbid_violations=violations,
                ))

            # 5. Clean shutdown
            try:
                await audio_ws.send(json.dumps({"type": "disconnect"}))
            except Exception:  # noqa: BLE001
                pass

        try:
            await ctrl.send(json.dumps({
                "type": "call:status",
                "callStatus": "completed",
                "call_sid": call_sid,
            }))
        except Exception:  # noqa: BLE001
            pass

    return result


def write_report(results: list[ScenarioResult], report_path: Path) -> None:
    lines = []
    total_pass = sum(1 for r in results if r.passed)
    lines.append(f"=== Harness report ({total_pass}/{len(results)} scenarios passed) ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(
            f"\n[{status}] {r.name} — {len(r.turns)} turns, "
            f"total {r.total_latency_ms}ms"
        )
        for t in r.turns:
            tag = "✓" if t.passed else "✗"
            lines.append(f"  {tag} t{t.turn_idx} ({t.latency_ms}ms)  USER: {t.user_msg}")
            lines.append(f"      BOT : {t.bot_reply[:200]}")
            if t.expect_fail:
                lines.append(f"      MISSED: {t.expect_fail}")
            if t.forbid_violations:
                lines.append(f"      FORBID: {t.forbid_violations}")
    text = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    print(text)


def write_csv(results: list[ScenarioResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "turn", "user_msg", "bot_reply", "latency_ms", "passed"])
        for r in results:
            for t in r.turns:
                w.writerow([r.name, t.turn_idx, t.user_msg, t.bot_reply, t.latency_ms, t.passed])


def load_scenarios(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml") and yaml is not None:
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported scenario format: {path.suffix}")
    if isinstance(data, dict) and "scenarios" in data:
        return data["scenarios"]
    if isinstance(data, list):
        return data
    raise ValueError("scenario file must be a list of scenarios or {scenarios: [...]}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["chat", "audio"], default="chat")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--scenarios", required=True, type=Path)
    p.add_argument("--report", type=Path, default=Path(".state/harness_report.txt"))
    p.add_argument("--csv", type=Path, default=Path(".state/harness_report.csv"))
    # Audio-mode only:
    p.add_argument("--silero-url", default="http://localhost:50006",
                   help="audio mode: Silero TTS service URL")
    p.add_argument("--whisper-url", default="http://localhost:50002",
                   help="audio mode: Whisper STT service URL")
    p.add_argument("--audio-dir", type=Path, default=Path(".state/harness_audio"),
                   help="audio mode: where to save bot WAVs")
    args = p.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    print(f"Running {len(scenarios)} scenarios via {args.mode}...")
    results: list[ScenarioResult] = []

    if args.mode == "chat":
        for sc in scenarios:
            print(f"  → {sc.get('name', '?')}")
            results.append(run_chat_scenario(args.base_url, sc))
    else:  # audio
        async def _run_all() -> list[ScenarioResult]:
            out: list[ScenarioResult] = []
            for sc in scenarios:
                print(f"  → {sc.get('name', '?')}")
                try:
                    r = await run_audio_scenario(
                        base_url=args.base_url,
                        silero_url=args.silero_url,
                        whisper_url=args.whisper_url,
                        scenario=sc,
                        audio_dir=args.audio_dir,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"    scenario error: {exc}", file=sys.stderr)
                    r = ScenarioResult(name=sc.get("name", "?"))
                out.append(r)
            return out
        results = asyncio.run(_run_all())

    write_report(results, args.report)
    write_csv(results, args.csv)
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
