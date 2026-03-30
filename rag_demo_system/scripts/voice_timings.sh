#!/usr/bin/env bash
tail -10 "$(dirname "$0")/../.state/logs.jsonl" | python3 -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except:
        continue
    if d.get('event') == 'voice_turn':
        stt = (d['stt_done'] - d['speech_stopped']) * 1000
        rag = (d['retrieval_done'] - d['stt_done']) * 1000
        llm = (d['llm_first_token'] - d['retrieval_done']) * 1000
        tts = (d['playback_started'] - d['llm_first_token']) * 1000
        total = d['primary_kpi_ms']
        print(f'STT: {stt:.0f}ms | RAG: {rag:.0f}ms | LLM: {llm:.0f}ms | TTS: {tts:.0f}ms | Total: {total:.0f}ms')
"
