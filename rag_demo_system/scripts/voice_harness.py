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
import csv
import json
import re
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


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


def run_chat_scenario(base_url: str, scenario: dict) -> ScenarioResult:
    name = scenario.get("name", "unnamed")
    phone = scenario.get("seed_phone") or "+375290000000"
    sid = f"harness-{uuid.uuid4().hex[:8]}"
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
    return result


# TODO(audio mode): build using websockets + Silero /speak + resampler.
# Skeleton signature so the next session can fill in:
#
# async def run_audio_scenario(
#     base_url: str,
#     silero_url: str,         # http://<host>:<silero_port>
#     whisper_url: str,        # for bot-reply ASR
#     scenario: dict,
# ) -> ScenarioResult: ...
#
# Steps inside:
#   1. ws = await websockets.connect(f"{base_url}/ws/jambonz",
#                                    subprotocols=["ws.jambonz.org"])
#   2. send session:new → recv ack → extract audio_ws_url
#   3. audio_ws = await websockets.connect(audio_ws_url)
#      send first frame: JSON metadata (callSid, sampleRate=16000, from=phone)
#   4. for each turn:
#        - POST silero_url/speak {"text": user_msg} → 24kHz pcm16 bytes
#        - resample to 16kHz mono (audioop.ratecv is enough)
#        - chunk into 20ms frames (320 bytes), send over audio_ws
#        - wait for VAD speech_end signal in backend.log (tail-grep)
#        - record incoming bot frames (24kHz pcm16) until ~600ms silence
#        - POST whisper_url/transcribe with bot WAV → bot_reply text
#        - run rubric check
#   5. send call:status=completed on control_ws


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
    args = p.parse_args(argv)

    if args.mode == "audio":
        print("audio mode is scaffolded — see TODO block in this file.", file=sys.stderr)
        return 2

    scenarios = load_scenarios(args.scenarios)
    print(f"Running {len(scenarios)} scenarios via {args.mode}...")
    results: list[ScenarioResult] = []
    for sc in scenarios:
        print(f"  → {sc.get('name', '?')}")
        results.append(run_chat_scenario(args.base_url, sc))

    write_report(results, args.report)
    write_csv(results, args.csv)
    failed = sum(1 for r in results if not r.passed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
