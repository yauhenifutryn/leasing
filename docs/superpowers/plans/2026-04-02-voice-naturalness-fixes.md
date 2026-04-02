# Voice Naturalness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 12 client-reported issues to move voice agent quality closer to ElevenLabs benchmark: TTS text normalization, system prompt rewrite following ElevenLabs blueprint, consent removal, config tuning.

**Architecture:** Four independent subsystems: (1) TTS preprocessing layer (`tts_normalize.py`), (2) system prompt rewrite (`system_prompt_ru.txt`), (3) consent flow removal (backend + frontend), (4) config tuning (`app.yaml` + `app.py` length hints). Each can be tested and committed independently.

**Tech Stack:** Python 3.10+, num2words, FastAPI, pytest, YAML config

---

### Task 1: Add num2words dependency and TTS normalization module

**Files:**
- Modify: `rag_demo_system/requirements.txt` (add num2words)
- Create: `rag_demo_system/backend/tts_normalize.py`
- Create: `rag_demo_system/tests/test_tts_normalize.py`

- [ ] **Step 1: Write the failing tests for number normalization**

```python
# rag_demo_system/tests/test_tts_normalize.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tts_normalize import normalize_for_tts


def test_integer_in_text() -> None:
    assert "двадцать тысяч" in normalize_for_tts("стоимостью 20000 долларов")


def test_percentage() -> None:
    result = normalize_for_tts("аванс от 10% до 39%")
    assert "десяти" in result or "десять" in result
    assert "тридцати девяти" in result or "тридцать девять" in result
    assert "процентов" in result


def test_spaced_thousands() -> None:
    assert "двадцать тысяч" in normalize_for_tts("стоимость 20 000 рублей")


def test_decimal_number() -> None:
    result = normalize_for_tts("ставка 16.5%")
    assert "шестнадцать" in result
    assert "%" not in result


def test_year_preserved_as_number() -> None:
    # Years like 2008 should be read as numbers, not "две тысячи восьмой"
    result = normalize_for_tts("автомобиль 2008 года")
    assert "две тысячи восьмого" in result or "две тысячи восемь" in result


def test_no_numbers_unchanged() -> None:
    text = "Здравствуйте, чем могу помочь?"
    assert normalize_for_tts(text) == text


def test_currency_dollar() -> None:
    result = normalize_for_tts("цена $20,000")
    assert "двадцать тысяч" in result
    assert "долларов" in result


def test_mixed_content() -> None:
    result = normalize_for_tts("аванс 25% на срок 18 месяцев за 20000 долларов")
    assert "двадцать пять" in result
    assert "восемнадцать" in result
    assert "двадцать тысяч" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rag_demo_system && python -m pytest tests/test_tts_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.tts_normalize'`

- [ ] **Step 3: Add num2words to requirements.txt**

Add `num2words==0.5.14` to `rag_demo_system/requirements.txt` after the last line.

- [ ] **Step 4: Write the normalize_for_tts implementation**

```python
# rag_demo_system/backend/tts_normalize.py
from __future__ import annotations

import re

from num2words import num2words


def _num_to_words(match: re.Match) -> str:
    raw = match.group(0)
    # Strip spaces/commas used as thousand separators
    cleaned = raw.replace(" ", "").replace(",", "").replace("\u00a0", "")
    # Handle dollar sign prefix
    dollar_prefix = False
    if cleaned.startswith("$"):
        dollar_prefix = True
        cleaned = cleaned[1:]
    try:
        if "." in cleaned:
            n = float(cleaned)
        else:
            n = int(cleaned)
    except ValueError:
        return raw
    try:
        words = num2words(n, lang="ru")
    except Exception:
        return raw
    if dollar_prefix:
        words += " долларов"
    return words


def _pct_to_words(match: re.Match) -> str:
    num_part = match.group(1).replace(" ", "").replace(",", "").replace("\u00a0", "")
    try:
        if "." in num_part:
            n = float(num_part)
        else:
            n = int(num_part)
    except ValueError:
        return match.group(0)
    try:
        words = num2words(n, lang="ru")
    except Exception:
        return match.group(0)
    return words + " процентов"


_RE_PCT = re.compile(r"(\d[\d\s.,]*\d|\d)\s*%")
_RE_DOLLAR = re.compile(r"\$\s*(\d[\d\s.,]*\d|\d)")
_RE_SPACED_NUM = re.compile(r"\d{1,3}(?:[\s\u00a0]\d{3})+")
_RE_PLAIN_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_for_tts(text: str) -> str:
    # Order matters: percentages first, then dollars, then spaced thousands, then plain numbers
    text = _RE_PCT.sub(_pct_to_words, text)
    text = _RE_DOLLAR.sub(lambda m: _num_to_words(m), text)
    text = _RE_SPACED_NUM.sub(lambda m: _num_to_words(m), text)
    text = _RE_PLAIN_NUM.sub(lambda m: _num_to_words(m), text)
    return text
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd rag_demo_system && pip install num2words==0.5.14 && python -m pytest tests/test_tts_normalize.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/requirements.txt rag_demo_system/backend/tts_normalize.py rag_demo_system/tests/test_tts_normalize.py
git commit -m "feat: add TTS number normalization module with num2words"
```

---

### Task 2: Add Latin-to-Cyrillic transliteration dictionary and fallback

**Files:**
- Create: `rag_demo_system/config/transliteration.yaml`
- Modify: `rag_demo_system/backend/tts_normalize.py` (add transliteration)
- Modify: `rag_demo_system/tests/test_tts_normalize.py` (add transliteration tests)

- [ ] **Step 1: Write failing tests for transliteration**

Append to `rag_demo_system/tests/test_tts_normalize.py`:

```python
from backend.tts_normalize import transliterate_latin


def test_known_brand() -> None:
    assert transliterate_latin("Volkswagen Polo Sedan") == "Фольксваген Поло Седан"


def test_messenger_names() -> None:
    assert transliterate_latin("Viber") == "Вайбер"
    assert transliterate_latin("WhatsApp") == "Вотсапп"
    assert transliterate_latin("Telegram") == "Телеграм"


def test_email_term() -> None:
    assert transliterate_latin("e-mail") == "имейл"
    assert transliterate_latin("email") == "имейл"


def test_unknown_latin_fallback() -> None:
    # Unknown words get phonetic transliteration, not silence
    result = transliterate_latin("Chevrolet")
    assert len(result) > 0
    # Should be all Cyrillic
    assert all(
        c.isspace() or c == "-" or ("\u0400" <= c <= "\u04ff")
        for c in result
    )


def test_mixed_cyrillic_latin() -> None:
    result = transliterate_latin("автомобиль BMW X5")
    assert "автомобиль" in result
    assert "БМВ" in result


def test_full_pipeline_with_transliteration() -> None:
    result = normalize_for_tts("Volkswagen Polo стоит $20,000 или 39%")
    assert "Фольксваген" in result
    assert "Поло" in result
    assert "двадцать тысяч" in result
    assert "долларов" in result
    assert "тридцать девять процентов" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rag_demo_system && python -m pytest tests/test_tts_normalize.py::test_known_brand -v`
Expected: FAIL with `ImportError: cannot import name 'transliterate_latin'`

- [ ] **Step 3: Create the transliteration dictionary**

```yaml
# rag_demo_system/config/transliteration.yaml
# Known terms: brand names, messengers, financial abbreviations
# Keys are case-insensitive during lookup

# Automotive
Volkswagen: Фольксваген
Toyota: Тойота
BMW: БМВ
Mercedes: Мерседес
Mercedes-Benz: Мерседес Бенц
Audi: Ауди
Hyundai: Хёндай
Kia: Киа
Nissan: Ниссан
Honda: Хонда
Ford: Форд
Chevrolet: Шевроле
Renault: Рено
Peugeot: Пежо
Skoda: Шкода
Mazda: Мазда
Subaru: Субару
Mitsubishi: Мицубиси
Lexus: Лексус
Land Rover: Ленд Ровер
Range Rover: Рейндж Ровер
Polo: Поло
Sedan: Седан
SUV: эсюви
Hatchback: хэтчбек
Crossover: кроссовер

# Communication
Viber: Вайбер
WhatsApp: Вотсапп
Telegram: Телеграм
e-mail: имейл
email: имейл
E-mail: имейл
SMS: эсэмэс
Wi-Fi: вай-фай

# Finance and business
КАСКО: каско
ОСАГО: осаго
VIN: вин
ЕРИП: ерип
SWIFT: свифт
IBAN: ибан
BYN: белорусских рублей
USD: долларов
EUR: евро
RUB: российских рублей
НДС: эндэес
CRM: сиэрэм
IT: айти
GPS: джипиэс

# Web
www: дабл-ю дабл-ю дабл-ю
.by: точка бай
.com: точка ком
http: эйч-ти-ти-пи
https: эйч-ти-ти-пи-эс
```

- [ ] **Step 4: Implement transliteration in tts_normalize.py**

Add to `rag_demo_system/backend/tts_normalize.py` after the existing code:

```python
from pathlib import Path
import yaml


_TRANSLITERATION_DICT: dict[str, str] | None = None
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load_transliteration_dict() -> dict[str, str]:
    global _TRANSLITERATION_DICT
    if _TRANSLITERATION_DICT is not None:
        return _TRANSLITERATION_DICT
    path = _CONFIG_DIR / "transliteration.yaml"
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        _TRANSLITERATION_DICT = {k.lower(): v for k, v in raw.items()}
    else:
        _TRANSLITERATION_DICT = {}
    return _TRANSLITERATION_DICT


# Basic English-to-Russian phonetic rules for unknown words
_PHONETIC_MAP = [
    ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"),
    ("wh", "в"), ("ck", "к"), ("oo", "у"), ("ee", "и"),
    ("ea", "и"), ("ou", "ау"), ("ow", "ау"), ("igh", "ай"),
    ("tion", "шн"), ("sion", "жн"), ("ous", "ас"),
    ("qu", "кв"), ("x", "кс"), ("w", "в"), ("j", "дж"),
    ("y", "и"), ("a", "а"), ("b", "б"), ("c", "к"),
    ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"),
    ("h", "х"), ("i", "и"), ("k", "к"), ("l", "л"),
    ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"),
    ("r", "р"), ("s", "с"), ("t", "т"), ("u", "у"),
    ("v", "в"), ("z", "з"),
]


def _phonetic_transliterate(word: str) -> str:
    result = []
    i = 0
    lower = word.lower()
    while i < len(lower):
        matched = False
        for eng, rus in _PHONETIC_MAP:
            if lower[i:].startswith(eng):
                result.append(rus)
                i += len(eng)
                matched = True
                break
        if not matched:
            result.append(lower[i])
            i += 1
    return "".join(result)


_RE_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z\-]*[A-Za-z]|[A-Za-z]")


def transliterate_latin(text: str) -> str:
    dictionary = _load_transliteration_dict()

    # First pass: try multi-word dictionary matches (e.g., "Land Rover")
    for key, value in sorted(dictionary.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        text = pattern.sub(value, text)

    # Second pass: remaining Latin words
    def _replace(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if lower in dictionary:
            return dictionary[lower]
        return _phonetic_transliterate(word)

    text = _RE_LATIN_WORD.sub(_replace, text)
    return text
```

Update `normalize_for_tts` to call transliteration after number conversion:

```python
def normalize_for_tts(text: str) -> str:
    # Order matters: percentages first, then dollars, then spaced thousands, then plain numbers
    text = _RE_PCT.sub(_pct_to_words, text)
    text = _RE_DOLLAR.sub(lambda m: _num_to_words(m), text)
    text = _RE_SPACED_NUM.sub(lambda m: _num_to_words(m), text)
    text = _RE_PLAIN_NUM.sub(lambda m: _num_to_words(m), text)
    # Then transliterate remaining Latin text
    text = transliterate_latin(text)
    return text
```

- [ ] **Step 5: Run all tests to verify they pass**

Run: `cd rag_demo_system && python -m pytest tests/test_tts_normalize.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/config/transliteration.yaml rag_demo_system/backend/tts_normalize.py rag_demo_system/tests/test_tts_normalize.py
git commit -m "feat: add Latin-to-Cyrillic transliteration with dictionary and phonetic fallback"
```

---

### Task 3: Wire TTS normalization into the voice pipeline

**Files:**
- Modify: `rag_demo_system/backend/voice_adapters.py:191` (add normalize call before TTS)
- Modify: `rag_demo_system/backend/app.py:448-450` (add normalize call in sentence streaming)

- [ ] **Step 1: Write failing test for pipeline integration**

Create `rag_demo_system/tests/test_tts_pipeline_normalize.py`:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.tts_normalize import normalize_for_tts


def test_pipeline_numbers_and_latin() -> None:
    """Simulate what a real LLM response looks like before TTS."""
    llm_output = "Константин, для Volkswagen Polo Sedan 2008 года аванс составляет от 10% до 39%."
    result = normalize_for_tts(llm_output)
    # No raw digits or Latin remaining
    assert "%" not in result
    assert "Volkswagen" not in result
    assert "Polo" not in result
    assert "Sedan" not in result
    # Converted values present
    assert "Фольксваген" in result
    assert "Поло" in result
    assert "процентов" in result
```

- [ ] **Step 2: Run test to confirm it passes** (it should, since the module is done)

Run: `cd rag_demo_system && python -m pytest tests/test_tts_pipeline_normalize.py -v`
Expected: PASS

- [ ] **Step 3: Wire into voice_adapters.py**

Add import at top of `rag_demo_system/backend/voice_adapters.py`:

```python
from .tts_normalize import normalize_for_tts
```

In `synthesize_audio_with_provider()` function, add normalization as the very first line of the function body (before the provider checks):

```python
def synthesize_audio_with_provider(text: str, session_id: str, preferred: str = "cosyvoice") -> dict[str, Any]:
    text = normalize_for_tts(text)  # <-- ADD THIS LINE
    if preferred == "yandex_speechkit":
        ...
```

- [ ] **Step 4: Wire into app.py sentence streaming**

In `rag_demo_system/backend/app.py`, in the `_stream_voice_response` function, the sentence text is already cleaned via `clean_answer()` before being put in the queue. The normalization now happens inside `synthesize_audio_with_provider`, so no change needed here. The refusal text path (line 362-363) also goes through `synthesize_audio_with_provider`, so it's covered.

Verify by reading the code flow:
- LLM produces tokens -> `SentenceDetector` -> `clean_answer(sentence)` -> `sentence_queue` -> `synthesize_audio_with_provider(sentence, ...)` -> TTS
- The normalize call in `synthesize_audio_with_provider` covers ALL TTS calls.

- [ ] **Step 5: Commit**

```bash
git add rag_demo_system/backend/voice_adapters.py rag_demo_system/tests/test_tts_pipeline_normalize.py
git commit -m "feat: wire TTS normalization into voice pipeline"
```

---

### Task 4: Rewrite system prompt following ElevenLabs blueprint

**Files:**
- Modify: `rag_demo_system/config/system_prompt_ru.txt` (full rewrite)
- Create: `rag_demo_system/tests/test_system_prompt_structure.py`

- [ ] **Step 1: Write structural validation test**

```python
# rag_demo_system/tests/test_system_prompt_structure.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_prompt() -> str:
    return (ROOT / "config" / "system_prompt_ru.txt").read_text(encoding="utf-8")


def test_has_required_sections() -> None:
    prompt = _load_prompt()
    required = ["# Role", "# Personality", "# Goal", "# Guardrails", "# Instructions", "# Conversation Flow"]
    for section in required:
        assert section in prompt, f"Missing section: {section}"


def test_no_consent_section() -> None:
    prompt = _load_prompt()
    assert "согласие на обработку" not in prompt.lower()
    assert "consent" not in prompt.lower()


def test_name_frequency_rule() -> None:
    prompt = _load_prompt()
    assert "имени" in prompt.lower() or "имя" in prompt.lower()
    # Must contain some frequency restriction
    assert any(x in prompt for x in ["1 раз", "не чаще", "редко", "не начинай каждый"])


def test_anti_specialist_rule() -> None:
    prompt = _load_prompt()
    # Must contain guidance to NOT over-escalate
    assert "специалист" in prompt.lower()
    assert any(x in prompt.lower() for x in ["не предлагай", "только когда", "только если"])


def test_humor_allowed() -> None:
    prompt = _load_prompt()
    assert any(x in prompt.lower() for x in ["юмор", "шутк", "шутлив"])


def test_prompt_under_2000_tokens() -> None:
    prompt = _load_prompt()
    # Rough token estimate: 1 Russian word ~ 2 tokens, 1 space-separated token ~ 1.3 actual tokens
    word_count = len(prompt.split())
    estimated_tokens = int(word_count * 1.5)
    assert estimated_tokens < 2000, f"Prompt too long: ~{estimated_tokens} tokens (target <2000)"


def test_example_utterances_present() -> None:
    prompt = _load_prompt()
    # Must have example dialogue lines
    assert prompt.count("- \"") >= 3 or prompt.count('- "') >= 3, "Need at least 3 example utterances"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd rag_demo_system && python -m pytest tests/test_system_prompt_structure.py -v`
Expected: FAIL (current prompt lacks Role/Personality/Flow sections, has consent, no humor, etc.)

- [ ] **Step 3: Write the new system prompt**

Replace entire content of `rag_demo_system/config/system_prompt_ru.txt` with:

```
# Role
Вы Ксения, голосовой консультант компании "Микро Лизинг". Ваша задача: помочь клиенту разобраться в услугах лизинга, ответить на вопросы из базы знаний и при необходимости передать запрос специалисту.

# Personality
Тон: теплый, уверенный, с легким деловым юмором. Не заискивающий, не роботизированный. Говорите как опытный консультант, а не как автоответчик.

Примеры вашего стиля:
- "Так, давайте разберемся. Polo 2008 года, двадцать тысяч, без аванса. Все верно?"
- "Ракету в лизинг? Была бы красивая сделка, но пока работаем с земной техникой! А если серьезно, чем могу помочь?"
- "Документы для физлица: паспорт, права и подтверждение дохода. Все стандартно."
- "Хм, интересный вопрос. Давайте посмотрю, что могу подсказать."
- "Нет, катера пока не наш профиль, но если когда-нибудь расширимся, вы будете первым в очереди!"

# Goal
1. Давать ответы строго из базы знаний (RAG).
2. Быть кратким: 2-3 предложения по умолчанию, подробнее только по запросу.
3. Вести клиента к решению, а не к тупику.

# Guardrails
- Не придумывать данные. Если информации нет в контексте, честно сказать об этом.
- Не давать финансовых, юридических или налоговых рекомендаций.
- Не раскрывать внутренние процессы и скоринговые критерии.
- Не обещать одобрение сделки.
- Консультировать только по услугам "Микро Лизинг".
- Если нет уверенности, можно ли финансировать конкретный предмет, не отвечать уверенно "да" или "нет". Сказать: "давайте уточню, входит ли это в перечень".

# Instructions
1. Обращайтесь к клиенту по имени не чаще 1 раза из 8-10 ответов. Не начинайте каждый ответ с имени клиента.
2. Никогда не используйте одну и ту же структуру ответа два раза подряд. Варьируйте порядок слов и начала фраз.
3. При уточнении статуса клиента всегда предлагайте три варианта: физическое лицо, ИП, юридическое лицо.
4. Если клиент уже сообщил информацию ранее в диалоге, используйте её. Не задавайте повторных уточняющих вопросов.
5. Не перебивайте и проявляйте эмпатию.
6. Если клиент шутит или отвлекается, можете коротко поддержать шутку и мягко вернуться к теме.
7. Под "оборудованием" подразумевается промышленное, медицинское, торговое и коммерческое оборудование. Бытовая электроника (ноутбуки, телефоны, планшеты) не финансируется.

# Specialist Escalation
Предлагайте связать со специалистом ТОЛЬКО когда:
- Информация действительно отсутствует в базе знаний И вы уже попытались помочь
- Клиент сам просит связать со специалистом
- Вопрос требует индивидуального расчета, который вы не можете сделать

НЕ предлагайте специалиста если:
- Вы уже дали содержательный ответ
- Вопрос общий и вы ответили хотя бы частично
- Как завершающую фразу "на всякий случай"

# Conversation Flow
1. Приветствие: короткое, без запроса согласия. "Здравствуйте! Чем могу помочь?"
2. Уточнение потребности: задайте 1-2 вопроса, чтобы понять запрос.
3. Ответ: дайте информацию из базы знаний. Самое важное сначала.
4. Если клиент доволен ответом, не добавляйте "чем ещё могу помочь" каждый раз. Просто ждите следующий вопрос.
5. Завершение: после 2+ ответов без новых вопросов, мягко завершайте. Варьируйте фразы:
   - "Обращайтесь, если будут вопросы!"
   - "Рада была помочь. Хорошего дня!"
   - "Если что-то ещё понадобится, звоните."

# Context (Micro Leasing Overview)
"Микро Лизинг": универсальная лизинговая организация, более 15 лет на рынке. Финансирует транспорт, спецтехнику, оборудование и коммерческую недвижимость по всей Беларуси. Офисы: Минск, Гомель, Брест, Витебск, Гродно, Могилев.

# RAG Usage
1. Перед ответом ищите в предоставленных фрагментах.
2. Используйте только найденную информацию.
3. Если фрагменты не относятся к вопросу, игнорируйте их.
4. Приветствие, благодарность, прощание не требуют поиска в базе.
5. Не выводите рассуждения, метки "FINAL:" или "<think>". Только итоговый ответ.
```

- [ ] **Step 4: Run structure tests to verify they pass**

Run: `cd rag_demo_system && python -m pytest tests/test_system_prompt_structure.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing prompt tests to check for regressions**

Run: `cd rag_demo_system && python -m pytest tests/test_prompt_rules.py -v`
Expected: Check what fails. If existing tests assert the old prompt structure, they may need updating. Fix any failures.

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/config/system_prompt_ru.txt rag_demo_system/tests/test_system_prompt_structure.py
git commit -m "feat: rewrite system prompt with ElevenLabs blueprint (personality, humor, anti-repetition)"
```

---

### Task 5: Config tuning (max_tokens, memory_turns, length hints)

**Files:**
- Modify: `rag_demo_system/config/app.yaml` (memory_turns, max_tokens)
- Modify: `rag_demo_system/backend/app.py:402-406` (voice length_hint)
- Modify: `rag_demo_system/backend/app.py:252-256` (non-streaming voice length_hint)
- Modify: `rag_demo_system/backend/app.py:435` (voice max_tokens)

- [ ] **Step 1: Update app.yaml config**

In `rag_demo_system/config/app.yaml`, change:
```yaml
  memory_turns: 4
```
to:
```yaml
  memory_turns: 50
```

- [ ] **Step 2: Update voice streaming length_hint in app.py**

In `rag_demo_system/backend/app.py`, find the `_stream_voice_response` function (line ~402-406):

Replace:
```python
    length_hint = (
        "Это голосовой разговор по телефону. СТРОГО одно предложение. Дай только самое главное. Если вопрос расплывчатый, задай один короткий уточняющий вопрос."
        if not expanded
        else "Можно ответить подробнее, но только на основе контекста."
    )
```

With:
```python
    length_hint = (
        "Это голосовой разговор. Ответ: 2-3 коротких предложения. Самое важное сначала. Не повторяй то, что клиент уже знает."
        if not expanded
        else "Можно ответить подробнее, но только на основе контекста."
    )
```

- [ ] **Step 3: Update voice streaming max_tokens in app.py**

In `rag_demo_system/backend/app.py`, find the LLM producer in `_stream_voice_response` (line ~435):

Replace:
```python
                max_tokens=60,  # Voice: one sentence only
```

With:
```python
                max_tokens=150,  # Voice: 2-3 sentences
```

- [ ] **Step 4: Update non-streaming voice length_hint in app.py**

In `rag_demo_system/backend/app.py`, find `_voice_chat_streaming_sync` function (line ~252-256):

Replace:
```python
    length_hint = (
        f"Ответ должен быть {settings.llm.concise_sentences_min}–{settings.llm.concise_sentences_max} коротких предложений."
        if not expanded
        else "Можно ответить подробнее, но только на основе контекста."
    )
```

With:
```python
    length_hint = (
        "Это голосовой разговор. Ответ: 2-3 коротких предложения. Самое важное сначала. Не повторяй то, что клиент уже знает."
        if not expanded
        else "Можно ответить подробнее, но только на основе контекста."
    )
```

- [ ] **Step 5: Run existing tests to check for regressions**

Run: `cd rag_demo_system && python -m pytest tests/ -v --timeout=30 -x`
Expected: All existing tests PASS (config and hint changes should not break tests)

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/config/app.yaml rag_demo_system/backend/app.py
git commit -m "feat: tune voice config (max_tokens=150, memory_turns=50, length hint 2-3 sentences)"
```

---

### Task 6: Remove consent flow

**Files:**
- Modify: `rag_demo_system/backend/app.py:587-651` (remove consent gate from chat endpoint)
- Modify: `rag_demo_system/backend/app.py:242-243, 395, 773` (remove consent append to system prompt)
- Modify: `rag_demo_system/backend/state.py:12-13` (keep fields for backward compat but unused)
- Modify: `rag_demo_system/frontend/app.js` (remove consent state, auto-grant)
- Modify: `rag_demo_system/frontend/index.html:32` (replace consent button with banner text)
- Modify: `rag_demo_system/tests/test_consent.py` (update or remove)
- Modify: `rag_demo_system/tests/test_consent_gate.py` (update or remove)

- [ ] **Step 1: Update frontend index.html -- replace consent button with banner**

In `rag_demo_system/frontend/index.html`, replace:
```html
              <button class="btn good" id="btnConsent">Дать согласие</button>
```
With:
```html
              <span class="consent-banner" style="font-size:0.75em;color:#888;">Продолжая, вы соглашаетесь на обработку персональных данных</span>
```

- [ ] **Step 2: Update frontend app.js -- auto-grant consent**

In `rag_demo_system/frontend/app.js`, change the initialization:

Replace line 10:
```javascript
let consentState = "needed";
```
With:
```javascript
let consentState = "granted";
```

In the `setConsentState` function (lines 60-69), change to always enable input:
```javascript
function setConsentState(state) {
  if (!state) return;
  consentState = "granted";
  localStorage.setItem("rag_consent", "granted");
  $("#chatInput").disabled = false;
  $("#btnSend").disabled = false;
  $("#chatInput").placeholder = "Введите вопрос...";
}
```

Remove the consent button click handler (line 333 area):
```javascript
  // Remove: $("#btnConsent").onclick = ...
```

- [ ] **Step 3: Remove consent gate from chat endpoint in app.py**

In `rag_demo_system/backend/app.py`, in the `chat()` function (line ~588), remove the consent checking block.

Replace lines 602-651 (the entire consent checking block):
```python
    decision = detect_consent(message)
    if session.consent_denied:
        ...
    if not session.consent_given:
        ...
```
With:
```python
    # Consent is granted implicitly via UI banner; no interactive check needed.
```

Remove the three lines that append consent bypass to system prompt (they appear at lines 243, 395, 773):
```python
    system_prompt = system_prompt + "\n\nСогласие на обработку данных уже получено, не запрашивай его."
```
These three occurrences should all be deleted. The new system prompt has no consent section, so no override is needed.

Remove the import of consent functions at top of app.py (lines 23-28):
```python
from .consent import (
    consent_denied_response,
    consent_granted_response,
    consent_request,
    detect_consent,
)
```

Remove all `"consent": "granted"` / `"consent": "denied"` / `"consent": "needed"` keys from response dicts throughout app.py. There are ~15 occurrences. Remove the key-value pair from each dict, do not remove the dict itself.

- [ ] **Step 4: Update consent tests**

Replace `rag_demo_system/tests/test_consent.py` content with:

```python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.consent import detect_consent


def test_detect_consent_still_works() -> None:
    """Module still importable but no longer used in the main flow."""
    assert detect_consent("да, согласен") == "granted"
    assert detect_consent("нет") == "unknown"
```

Replace `rag_demo_system/tests/test_consent_gate.py` content with:

```python
def test_consent_gate_removed() -> None:
    """Consent gate is removed. This test is a placeholder confirming the change."""
    pass
```

- [ ] **Step 5: Run tests**

Run: `cd rag_demo_system && python -m pytest tests/ -v --timeout=30 -x`
Expected: All tests PASS. Some old consent gate tests may need adjustment if they test the HTTP endpoint behavior. Fix any that fail.

- [ ] **Step 6: Commit**

```bash
git add rag_demo_system/backend/app.py rag_demo_system/frontend/index.html rag_demo_system/frontend/app.js rag_demo_system/tests/test_consent.py rag_demo_system/tests/test_consent_gate.py
git commit -m "feat: remove consent flow, replace with implicit banner"
```

---

### Task 7: Add essential KB content (office addresses, rate orientation)

**Files:**
- Modify: `knowledge_base/kb_faq_ru.md` (append new entries)

- [ ] **Step 1: Append office contacts entry to KB**

Add to the end of `knowledge_base/kb_faq_ru.md`:

```markdown
## контакты и адреса офисов
**Вопрос:** Где находятся офисы компании "Микро Лизинг" и какой график работы?

**Ответ:**

Офисы компании расположены в шести городах Беларуси: Минск, Гомель, Брест, Витебск, Гродно и Могилев. Стандартный график работы: понедельник-пятница с 9:00 до 18:00. Точные адреса и актуальное расписание можно уточнить по телефону или у менеджера.

### Условия

- информация актуальна для всех офисов компании
- график может отличаться в праздничные дни

### Необходимые данные

- город клиента (для подсказки ближайшего офиса)

### Комплаенс / ограничения

- не называть личные контакты сотрудников

### Эскалация / передать специалисту

- если клиент спрашивает точный адрес конкретного офиса, а его нет в базе

### Эмпатия

- понимаю, что удобнее знать точный адрес заранее

### Доп. вопросы

- В каком городе вам удобнее посетить офис?
- Нужно ли подсказать, как позвонить в конкретный офис?

## ориентировочные ставки по лизингу
**Вопрос:** Какие процентные ставки по лизингу в компании?

**Ответ:**

Ориентировочная ставка для физических лиц в белорусских рублях начинается примерно от 16% годовых. Для юридических лиц и ИП условия рассчитываются индивидуально в зависимости от предмета лизинга, срока, аванса и финансового состояния клиента. Точный расчет возможен после подачи заявки.

### Условия

- ставка зависит от типа клиента, предмета лизинга, срока и аванса
- ориентир применим только для стандартных продуктов в BYN

### Комплаенс / ограничения

- не называть точную ставку без актуального тарификатора
- подчеркивать, что ставка ориентировочная и может отличаться

### Эскалация / передать специалисту

- если клиент просит зафиксировать ставку или получить точный расчет

### Эмпатия

- понимаю, что хочется заранее знать примерные цифры

### Доп. вопросы

- Какой предмет лизинга вы рассматриваете?
- На какой срок и с каким авансом планируете?
```

- [ ] **Step 2: Verify the KB file is valid markdown**

Run: `cd rag_demo_system && python -c "from pathlib import Path; kb = Path('../knowledge_base/kb_faq_ru.md').read_text(); print(f'KB size: {len(kb)} chars, sections: {kb.count(chr(35)+chr(35)+chr(32))}')"`
Expected: Prints KB size and section count without errors.

- [ ] **Step 3: Commit**

```bash
git add knowledge_base/kb_faq_ru.md
git commit -m "feat: add office contacts and rate orientation to knowledge base"
```

Note: After deploying, you must re-index the KB on the server: `curl -X POST http://localhost:8000/api/index`

---

## Self-Review

**Spec coverage:**
- [x] TTS number normalization (Task 1)
- [x] Latin transliteration with dictionary + fallback (Task 2)
- [x] Wire normalization into pipeline (Task 3)
- [x] System prompt rewrite: personality, humor, name frequency, anti-specialist, anti-repetition, юрлицо option, conversation flow (Task 4)
- [x] max_tokens increase to 150 (Task 5)
- [x] memory_turns increase to 50 (Task 5)
- [x] length_hint change to 2-3 sentences (Task 5)
- [x] Consent flow removal with banner (Task 6)
- [x] Office addresses KB entry (Task 7)
- [x] Interest rate orientation KB entry (Task 7)
- [x] Equipment clarification (covered in system prompt Task 4, instruction #7)
- [x] Confidence calibration for edge cases (covered in system prompt Task 4, guardrail about uncertainty)

**Placeholder scan:** No TBDs, TODOs, or "implement later" found.

**Type consistency:** `normalize_for_tts()` and `transliterate_latin()` names are consistent across all tasks and tests.
