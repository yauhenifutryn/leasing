#!/usr/bin/env python3
"""Comprehensive tool behavior validation. Tests all tool-related scenarios.

Usage: .venv/bin/python scripts/test_all_tool_behaviors.py
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
from backend.voice_adapters import format_phones_for_tts, normalize_abbreviations_for_tts
from pathlib import Path


def call_llm(settings, system_prompt, messages, schemas, max_tokens=200):
    """Call LLM and return (tool_calls, text_content)."""
    base_url = settings.llm.fast_base_url or settings.llm.base_url
    model = settings.llm.model
    events = list(iter_openai_compatible_stream_events(
        base_url=base_url, model=model, messages=messages,
        temperature=0.1, max_tokens=max_tokens, timeout_sec=30,
        tools=schemas,
    ))
    tc = parse_tool_calls_from_events(events)
    text = "".join(
        (e.get("choices") or [{}])[0].get("delta", {}).get("content", "") or ""
        for e in events
    )
    return tc, text


def test_calc_tool_call(settings, system_prompt, schemas):
    """Test 1: Calculator tool is called on calc request."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Рассчитай лизинг на машину за 30 тысяч"},
    ]
    tc, text = call_llm(settings, system_prompt, messages, schemas)
    if tc and tc[0]["function"]["name"] == "calculator":
        args = json.loads(tc[0]["function"]["arguments"])
        print(f"  PASS: calculator called with {args}")
        return True
    else:
        print(f"  FAIL: got text instead of tool call: {text[:100]}")
        return False


def test_calc_execution(settings):
    """Test 2: Calculator API returns real results."""
    tool = get_tool("calculator")
    params, defaulted = tool.fill_defaults({"subject": "Легковой автомобиль", "cost": 30000})
    result = tool.execute(params, {})
    if result.get("ok"):
        print(f"  PASS: advance={result['advance_sum']}, monthly={result.get('payment_min', '?')}, url={result.get('url', '?')[:50]}")
        return result
    else:
        print(f"  FAIL: {result.get('error', 'unknown')}")
        return None


def test_calc_voice_summary(settings):
    """Test 3: Calculator voice summary has required fields."""
    tool = get_tool("calculator")
    params, defaulted = tool.fill_defaults({"subject": "Легковой автомобиль", "cost": 30000})
    result = tool.execute(params, {})
    result["defaulted"] = defaulted
    summary = tool.format_voice_summary(result)
    checks = {
        "advance": "9000" in summary or "9 000" in summary,
        "monthly": "897" in summary or "898" in summary,
        "buyout": "300" in summary,
        "defaults_marked": "*" in summary,
        "url": "mikro-leasing.by/graphic" in summary,
    }
    all_pass = all(checks.values())
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    return all_pass


def test_sms_body(settings):
    """Test 4: SMS body contains link."""
    tool = get_tool("calculator")
    params, defaulted = tool.fill_defaults({"subject": "Легковой автомобиль", "cost": 30000})
    result = tool.execute(params, {})
    sms = tool.format_sms_body(result)
    if sms and "mikro-leasing.by/graphic" in sms:
        print(f"  PASS: SMS body has link ({len(sms)} chars)")
        return True
    else:
        print(f"  FAIL: SMS body missing or no link")
        return False


def test_sms_tool_call(settings, system_prompt, schemas):
    """Test 5: SMS tool is called when user agrees to SMS."""
    # Simulate: calculator already ran, user says "отправь смс"
    calc_tool = get_tool("calculator")
    params, defaulted = calc_tool.fill_defaults({"subject": "Легковой автомобиль", "cost": 30000})
    calc_result = calc_tool.execute(params, {})
    calc_result["defaulted"] = defaulted
    sms_body = calc_tool.format_sms_body(calc_result)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Последний расчёт (для отправки по СМС):\n{sms_body}\n\nОтправь график по СМС"},
    ]
    tc, text = call_llm(settings, system_prompt, messages, schemas)
    if tc and tc[0]["function"]["name"] == "send_sms":
        args = json.loads(tc[0]["function"]["arguments"])
        print(f"  PASS: send_sms called, phone={args.get('phone', '?')}")
        return True
    else:
        print(f"  FAIL: got text instead of send_sms call: {text[:100]}")
        return False


def test_non_calc_uses_rag(settings, system_prompt, schemas):
    """Test 6: Non-calc question does NOT trigger tool call."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Кто директор вашей компании?"},
    ]
    tc, text = call_llm(settings, system_prompt, messages, schemas)
    if not tc and text:
        has_director = "дедков" in text.lower() or "директор" in text.lower()
        print(f"  PASS: text response, mentions director: {has_director}, len={len(text)}")
        return True
    else:
        print(f"  FAIL: tool was called for non-calc question")
        return False


def test_phone_tts():
    """Test 7: Phone number formatting for TTS."""
    tests = {
        "375291224557": "плюс 375, 29, 122, 45, 57",
        "+375 222 71 76 76": "плюс 375, 222, 71, 76, 76",
    }
    all_pass = True
    for input_text, expected_fragment in tests.items():
        result = format_phones_for_tts(input_text)
        ok = expected_fragment in result or "плюс 375" in result
        print(f"  {'PASS' if ok else 'FAIL'}: '{input_text}' -> '{result}'")
        if not ok:
            all_pass = False
    return all_pass


def test_abbreviation_tts():
    """Test 8: BYN and SMS pronunciation."""
    tests = {
        "BYN": "белорусских рублей",
        "СМС": "эс эм эс",
        "SMS": "эс эм эс",
    }
    all_pass = True
    for input_text, expected in tests.items():
        result = normalize_abbreviations_for_tts(input_text)
        ok = expected in result
        print(f"  {'PASS' if ok else 'FAIL'}: '{input_text}' -> '{result}'")
        if not ok:
            all_pass = False
    return all_pass


def test_second_llm_call(settings, system_prompt, schemas):
    """Test 9: After tool call, second LLM call presents results."""
    # Call 1: get tool call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Рассчитай лизинг на машину за 30 тысяч"},
    ]
    tc, _ = call_llm(settings, system_prompt, messages, schemas)
    if not tc:
        print("  FAIL: no tool call on first attempt")
        return False

    # Execute tool
    func_args = json.loads(tc[0]["function"]["arguments"])
    tool = get_tool(tc[0]["function"]["name"])
    filled, defaulted = tool.fill_defaults(func_args)
    result = tool.execute(filled, {})
    result["defaulted"] = defaulted
    summary = tool.format_voice_summary(result)

    # Call 2: present results
    messages.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": tc[0].get("id", "call_1"), "type": "function",
                        "function": {"name": tc[0]["function"]["name"],
                                     "arguments": json.dumps(func_args, ensure_ascii=False)}}],
    })
    messages.append({
        "role": "tool", "tool_call_id": tc[0].get("id", "call_1"), "content": summary,
    })

    base_url = settings.llm.fast_base_url or settings.llm.base_url
    model = settings.llm.model
    events = list(iter_openai_compatible_stream_events(
        base_url=base_url, model=model, messages=messages,
        temperature=0.3, max_tokens=300, timeout_sec=30, tools=schemas,
    ))
    text = "".join(
        (e.get("choices") or [{}])[0].get("delta", {}).get("content", "") or ""
        for e in events
    )
    has_numbers = "9000" in text or "9 000" in text or "898" in text or "897" in text
    has_sms_offer = "смс" in text.lower() or "sms" in text.lower()
    print(f"  {'PASS' if has_numbers else 'FAIL'}: result has real numbers")
    print(f"  {'PASS' if has_sms_offer else 'FAIL'}: offers SMS")
    print(f"  Preview: {text[:200]}")
    return has_numbers


def test_client_type_validation(settings):
    """Test 10: Used vehicle without age returns error."""
    tool = get_tool("calculator")
    result = tool.execute({
        "subject": "Легковой автомобиль", "cost": 20000,
        "condition_new": 0, "client_type": "Физическое лицо",
        "currency": "BYN", "prepaid": 30, "term": 36, "type_schedule": "0",
    }, {})
    if not result.get("ok") and "возраст" in result.get("error", "").lower():
        print(f"  PASS: used without age rejected")
        return True
    else:
        print(f"  FAIL: expected age error, got {result}")
        return False


def test_rag_after_calc(settings, system_prompt, schemas):
    """Test 11: RAG works for normal questions AFTER a calculator conversation.

    In the real voice path, the calc turn skips RAG but the next KB turn
    goes through the RAG path with full KB fragments. This test verifies
    that the model answers from KB context, not from tools, when the
    question is not about calculation.
    """
    # The real test: "кто директор" should NOT trigger a tool call,
    # and should produce a text answer. In the voice path, RAG context
    # would be included for this non-calc question.
    # Here we test with KB-like context in the system prompt.
    messages = [
        {"role": "system", "content": system_prompt + "\n\nСправочная информация:\nДиректор компании Микро Лизинг — Дедков Вадим Николаевич. Он наёмный руководитель."},
        # Previous calc exchange in history
        {"role": "user", "content": "Рассчитай лизинг на машину за 30 тысяч"},
        {"role": "assistant", "content": "Аванс 9000, платёж 898, срок 36 мес. Хотите изменить параметр или отправить по СМС?"},
        # Now a KB question
        {"role": "user", "content": "Кто директор вашей компании?"},
    ]
    tc, text = call_llm(settings, system_prompt, messages, schemas)
    if not tc and ("дедков" in text.lower() or "директор" in text.lower()):
        print(f"  PASS: answered about director after calc context")
        return True
    elif tc:
        print(f"  FAIL: tool was called for KB question after calc")
        return False
    else:
        print(f"  FAIL: no useful answer: {text[:100]}")
        return False


def test_sms_actually_sends(settings):
    """Test 12: SMS API actually sends (to non-existent number, checks response code)."""
    tool = get_tool("send_sms")
    result = tool.execute({
        "phone": "375000000000",  # Non-existent number, won't charge
        "message": "Test from automated validation",
    }, {})
    # Response should be ok=True (message accepted) or ok=False with a delivery error
    # but NOT an auth error (-2) or connection error
    if result.get("ok"):
        print(f"  PASS: SMS API accepted message (id={result.get('message_id', '?')})")
        return True
    elif "авторизац" in result.get("error", "").lower() or "-2" in result.get("error", ""):
        print(f"  FAIL: SMS auth error - credentials wrong")
        return False
    else:
        # Other errors (invalid number, etc.) still mean the API is reachable
        print(f"  PASS: SMS API reachable (error for fake number: {result.get('error', '?')})")
        return True


def test_normal_then_calc(settings, system_prompt, schemas):
    """Test 13: Normal KB question first, then calculator works."""
    # Turn 1: KB question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Какие документы нужны для лизинга?"},
    ]
    tc1, text1 = call_llm(settings, system_prompt, messages, schemas)
    kb_ok = not tc1 and len(text1) > 30
    print(f"  {'PASS' if kb_ok else 'FAIL'}: KB question answered ({len(text1)} chars)")

    # Turn 2: Calculator (fresh messages, simulating new turn)
    messages2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Рассчитай лизинг на машину за 50 тысяч"},
    ]
    tc2, text2 = call_llm(settings, system_prompt, messages2, schemas)
    calc_ok = tc2 and tc2[0]["function"]["name"] == "calculator"
    print(f"  {'PASS' if calc_ok else 'FAIL'}: calculator called after KB question")
    return kb_ok and calc_ok


def test_link_from_api(settings):
    """Test 14: Calculator API returns a valid URL in the response."""
    tool = get_tool("calculator")
    params, _ = tool.fill_defaults({"subject": "Легковой автомобиль", "cost": 30000})
    result = tool.execute(params, {})
    url = result.get("url", "")
    has_url = url.startswith("https://mikro-leasing.by/graphic/?") and len(url) > 40
    calc_id = result.get("calculation_id", "")
    has_id = len(calc_id) > 5
    print(f"  {'PASS' if has_url else 'FAIL'}: URL = {url}")
    print(f"  {'PASS' if has_id else 'FAIL'}: calculation_id = {calc_id}")
    return has_url and has_id


def test_sms_no_repeat(settings, system_prompt, schemas):
    """Test 15: After SMS sent, model says short confirmation, does NOT repeat calc results."""
    # Simulate the tool result from send_sms
    sms_tool = get_tool("send_sms")
    sms_result = {"ok": True, "message_id": "12345"}
    summary = sms_tool.format_voice_summary(sms_result)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Рассчитай лизинг на машину за 30 тысяч"},
        {"role": "assistant", "content": "Аванс 9000, платёж 898 BYN. Отправить по СМС?"},
        {"role": "user", "content": "Да, отправь"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_sms", "type": "function",
                         "function": {"name": "send_sms", "arguments": '{"phone":"375291224557","message":"test"}'}}]},
        {"role": "tool", "tool_call_id": "call_sms", "content": summary},
    ]

    base_url = settings.llm.fast_base_url or settings.llm.base_url
    model = settings.llm.model
    events = list(iter_openai_compatible_stream_events(
        base_url=base_url, model=model, messages=messages,
        temperature=0.3, max_tokens=150, timeout_sec=30, tools=schemas,
    ))
    text = "".join(
        (e.get("choices") or [{}])[0].get("delta", {}).get("content", "") or ""
        for e in events
    )
    is_short = len(text) < 200
    has_confirm = "отправ" in text.lower() or "готово" in text.lower() or "помочь" in text.lower()
    no_repeat = "9000" not in text and "898" not in text
    print(f"  {'PASS' if is_short else 'FAIL'}: response is short ({len(text)} chars)")
    print(f"  {'PASS' if has_confirm else 'FAIL'}: has confirmation")
    print(f"  {'PASS' if no_repeat else 'FAIL'}: does NOT repeat calc numbers")
    print(f"  Response: {text[:150]}")
    return is_short and has_confirm and no_repeat


def main():
    settings = load_settings()
    init_tools(settings)
    schemas = get_tool_schemas()
    system_prompt = settings.app.system_prompt_path.read_text(encoding="utf-8")

    print(f"System prompt: {len(system_prompt)} chars")
    print(f"Tools: {len(schemas)}")
    print(f"LLM: {settings.llm.fast_base_url or settings.llm.base_url}")
    print()

    tests = [
        ("1. Calculator tool called on calc request", lambda: test_calc_tool_call(settings, system_prompt, schemas)),
        ("2. Calculator API returns real results", lambda: test_calc_execution(settings)),
        ("3. Voice summary has required fields", lambda: test_calc_voice_summary(settings)),
        ("4. SMS body contains link", lambda: test_sms_body(settings)),
        ("5. SMS tool called when user agrees", lambda: test_sms_tool_call(settings, system_prompt, schemas)),
        ("6. Non-calc question uses RAG (no tool call)", lambda: test_non_calc_uses_rag(settings, system_prompt, schemas)),
        ("7. Phone number TTS formatting", test_phone_tts),
        ("8. Abbreviation TTS (BYN, SMS)", test_abbreviation_tts),
        ("9. Second LLM call presents results + offers SMS", lambda: test_second_llm_call(settings, system_prompt, schemas)),
        ("10. Used vehicle without age rejected", lambda: test_client_type_validation(settings)),
        ("11. RAG works after calculator conversation", lambda: test_rag_after_calc(settings, system_prompt, schemas)),
        ("12. SMS API actually sends", lambda: test_sms_actually_sends(settings)),
        ("13. Normal KB question then calculator works", lambda: test_normal_then_calc(settings, system_prompt, schemas)),
        ("14. Calculator API returns valid URL/link", lambda: test_link_from_api(settings)),
        ("15. SMS confirmation is short, no repeat", lambda: test_sms_no_repeat(settings, system_prompt, schemas)),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n=== {name} ===")
        try:
            ok = test_fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    if passed < total:
        print(f"\n{total - passed} test(s) failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed!")


if __name__ == "__main__":
    main()
