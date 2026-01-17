from backend.memory import build_memory_block


def test_memory_block_limits_turns():
    transcript = [
        {"role": "user", "text": "u1"},
        {"role": "assistant", "text": "a1"},
        {"role": "user", "text": "u2"},
        {"role": "assistant", "text": "a2"},
        {"role": "user", "text": "u3"},
        {"role": "assistant", "text": "a3"},
    ]

    block = build_memory_block(transcript, max_turns=2)

    assert "Клиент: u1" not in block
    assert "Агент: a1" not in block
    assert "Клиент: u2" in block
    assert "Агент: a2" in block
    assert "Клиент: u3" in block
    assert "Агент: a3" in block


def test_memory_block_empty_when_disabled():
    assert build_memory_block([], max_turns=4) == ""
    assert build_memory_block([{"role": "user", "text": "u1"}], max_turns=0) == ""
