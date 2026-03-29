#!/usr/bin/env bash
# Test RAG retrieval speed through the text chat endpoint
BASE="http://localhost:8000"
SID="rag_timing_$$"

# Give consent first
curl -s -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"message\":\"да, согласен\",\"session_id\":\"$SID\"}" > /dev/null

# Time the actual question
echo "Sending question..."
curl -s -X POST "$BASE/api/chat" \
  -H 'Content-Type: application/json' \
  -d "{\"message\":\"какой аванс по лизингу\",\"backend\":\"our_rag\",\"session_id\":\"$SID\"}" \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
t = d.get('timings', {})
print(f'embed:  {t.get(\"embed_ms\",0):.0f}ms')
print(f'qdrant: {t.get(\"qdrant_ms\",0):.0f}ms')
print(f'bm25:   {t.get(\"bm25_ms\",0):.0f}ms')
print(f'rerank: {t.get(\"rerank_ms\",0):.0f}ms')
print(f'total:  {t.get(\"total_ms\",0):.0f}ms')
print(f'llm:    {t.get(\"llm_total_ms\",0):.0f}ms')
print(f'chunks: {len(d.get(\"used_knowledge\",[]))}')
print(f'answer: {d.get(\"answer\",\"\")[:150]}')
"
