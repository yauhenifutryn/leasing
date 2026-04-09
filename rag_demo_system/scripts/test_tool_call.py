#!/usr/bin/env python3
"""Test tool calling end-to-end via terminal. Simulates the voice pipeline without audio.

Usage: .venv/bin/python scripts/test_tool_call.py "Рассчитай лизинг на машину за 30 тысяч"
"""
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.settings import load_settings
from backend.tools import init_tools, get_tool_schemas, get_tool
from backend.tools.filler import get_filler
from backend.llm import iter_openai_compatible_stream_events
from backend.llm_stream import parse_tool_calls_from_events
from pathlib import Path


def main():
    message = sys.argv[1] if len(sys.argv) > 1 else "Рассчитай лизинг на машину за 30 тысяч"

    print(f"=== Tool Call Test ===")
    print(f"Message: {message}")
    print()

    # Load settings and init tools
    settings = load_settings()
    init_tools(settings)
    schemas = get_tool_schemas()
    print(f"Tools registered: {len(schemas)}")
    for s in schemas:
        print(f"  - {s['function']['name']}")
    print()

    # Load system prompt + inject tool instructions at placeholder
    system_prompt = settings.app.system_prompt_path.read_text(encoding="utf-8")
    tool_instructions_path = settings.app.system_prompt_path.parent / "tool_instructions.txt"
    if tool_instructions_path.exists() and schemas:
        tool_instructions = tool_instructions_path.read_text(encoding="utf-8")
        system_prompt = system_prompt.replace("{TOOL_INSTRUCTIONS}", tool_instructions)
    else:
        system_prompt = system_prompt.replace("{TOOL_INSTRUCTIONS}", "")
    print(f"System prompt: {len(system_prompt)} chars (with tool instructions)")

    # Build messages: same two-path structure as app.py
    # For calc requests: clean system prompt + clean user message (no RAG)
    # This matches Test 2 which was proven to trigger tool calls.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    base_url = settings.llm.fast_base_url or settings.llm.base_url
    model = settings.llm.model
    print(f"LLM: {base_url} / {model}")
    print()

    # Iteration loop (same as llm_producer)
    max_iterations = 3
    for iteration in range(max_iterations + 1):
        print(f"--- LLM Call #{iteration + 1} ---")
        t0 = time.time()

        events = []
        has_content = False
        content_text = ""

        try:
            stream = iter_openai_compatible_stream_events(
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=settings.llm.temperature,
                max_tokens=120 if iteration == 0 else 220,
                timeout_sec=settings.llm.timeout_sec,
                tools=schemas if iteration < max_iterations else None,
            )
            for event in stream:
                events.append(event)
                choice = (event.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                token = delta.get("content") or ""
                if token:
                    has_content = True
                    content_text += token
                    print(token, end="", flush=True)
        except Exception as exc:
            print(f"\nERROR: {exc}")
            break

        elapsed = time.time() - t0
        print(f"\n[{elapsed:.1f}s]")

        if has_content:
            print(f"\nResult: TEXT response ({len(content_text)} chars)")
            break

        # Check for tool calls
        tool_calls = parse_tool_calls_from_events(events)
        if not tool_calls:
            print("No content and no tool calls. Unexpected.")
            break

        print(f"\nResult: TOOL CALL detected!")
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args = json.loads(tc["function"]["arguments"])
            print(f"  Tool: {func_name}")
            print(f"  Args: {json.dumps(func_args, ensure_ascii=False)}")
            print(f"  Filler: {get_filler(func_name)}")

            # Execute tool
            tool = get_tool(func_name)
            filled_params, defaulted = tool.fill_defaults(func_args)
            print(f"  Filled: {json.dumps(filled_params, ensure_ascii=False)}")
            print(f"  Defaulted: {defaulted}")

            t1 = time.time()
            result = tool.execute(filled_params, {})
            print(f"  Execute: {time.time() - t1:.1f}s, ok={result.get('ok')}")

            if result.get("ok"):
                result["defaulted"] = defaulted
                summary = tool.format_voice_summary(result)
                print(f"  Voice summary:\n{summary}")
                sms = tool.format_sms_body(result)
                if sms:
                    print(f"  SMS body:\n{sms}")
            else:
                summary = result.get("error", "Error")
                print(f"  Error: {summary}")

            # Append to messages for next iteration
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc.get("id", f"call_{func_name}"),
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(func_args, ensure_ascii=False),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{func_name}"),
                "content": summary,
            })

        print()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
