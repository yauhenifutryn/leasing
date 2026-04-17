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


def test_build_memory_block_no_interrupt_instruction_on_normal_turns():
    transcript = [
        {"role": "user", "text": "Привет"},
        {"role": "assistant", "text": "Здравствуйте, чем могу помочь?"},
        {"role": "user", "text": "Хочу лизинг"},
        {"role": "assistant", "text": "Подскажите стоимость."},
    ]
    block = build_memory_block(transcript, max_turns=10)
    assert "ВАЖНО: последний ответ был прерван" not in block
    assert "Клиент: Хочу лизинг" in block
    assert "Агент: Подскажите стоимость." in block


def test_build_memory_block_injects_instruction_when_last_asst_interrupted():
    transcript = [
        {"role": "user", "text": "Рассчитай"},
        {"role": "assistant", "text": "Аванс составляет 57 тысяч [прервано клиентом]"},
    ]
    block = build_memory_block(transcript, max_turns=10)
    assert "ВАЖНО: последний ответ был прерван" in block
    # Instruction appears AFTER history lines
    assert block.index("Аванс составляет") < block.index("ВАЖНО:")


def test_build_memory_block_no_instruction_when_interrupt_is_old():
    # Interrupt exists but user has sent a newer turn AND assistant replied after,
    # so the LAST assistant message is not the interrupted one.
    transcript = [
        {"role": "user", "text": "Рассчитай"},
        {"role": "assistant", "text": "Аванс составляет 57 тысяч [прервано клиентом]"},
        {"role": "user", "text": "Продолжи"},
        {"role": "assistant", "text": "Ежемесячный платёж 5003 рубля."},
    ]
    block = build_memory_block(transcript, max_turns=10)
    assert "ВАЖНО: последний ответ был прерван" not in block


def test_build_memory_block_empty_transcript_unchanged():
    assert build_memory_block([], max_turns=10) == ""


def test_build_memory_block_zero_turns_unchanged():
    assert build_memory_block([{"role": "user", "text": "x"}], max_turns=0) == ""
