# Spec 5: Whisper Initial Prompt Tuning

**Cluster:** E — STT
**Depends on:** —
**Blocks:** —

## Context

Transcript 779154b4 shows Whisper misrecognizing the bot's name "Ксения"
multiple times as "Сеня", "Синяя", "Синяна", "Алексей". Clients' request for
a linear payment schedule ("линейный график") is also at risk — "линейный"
is not in the current prompt's vocabulary. KB search confirms:
- "Ксения" appears 0 times in KB and 0 times in prompt.
- "линейный" / "дифференцированный" appear 0 times in KB and 0 times in prompt.
- "нагрузка" (financial term client used) appears 15 times in KB but 0 in
  prompt.

Current prompt (`services/whisper_server.py:15-35`) is ~972 characters, which
translates to ~275 tokens in Whisper's tokenizer. Whisper's `initial_prompt`
is capped at 224 tokens — anything beyond is silently discarded from the
beginning. The current prompt's preamble ("Клиент звонит в компанию...") is
likely wasted.

## Problem

1. Bot name missing from prompt → Whisper guesses similar-sounding names.
2. Domain terms used on-call absent from biasing (graph types, financial
   vocabulary, Belarusian client slang).
3. Current prompt exceeds 224-token budget; valuable content at the start is
   truncated.

## Goals

- Replace `_DEFAULT_INITIAL_PROMPT` with a tightened string under 224 tokens.
- Prioritize high-ROI items at the end (they are guaranteed to be included).
- Place the bot's name "Ксения" 2-3 times, including final position.
- Add graph-type vocabulary, client-type slang, and financial terms.
- Drop rarely-seen car brand aliases to free token budget.
- No post-STT correction dictionary (explicitly rejected by user).

## Non-goals

- Fine-tuning Whisper on domain transcripts (deferred to V2, per
  `project_stt_v2_roadmap.md`)
- Switching STT engine (deferred)
- Post-STT correction dictionary (rejected)
- Per-user dynamic prompts (future optimization)

## Design

### New `_DEFAULT_INITIAL_PROMPT`

```
Микро Лизинг, лизинг в Беларуси. Помощница Ксения.
Города: Минск, Гомель, Брест, Витебск, Гродно, Могилёв.
Марки: Volkswagen Фольксваген, Toyota Тойота, BMW бэха,
Mercedes мерс, Audi аудюха, Hyundai Хёндай, Kia Киа,
Skoda Шкода, Lada Лада ВАЗ, ГАЗ ГАЗель, МАЗ,
Geely Джили, Chery Чери, Haval Хавал, BYD.
Документы: паспорт, водительские права, УНП, ИНН, VIN,
НДС, КАСКО, ОСАГО, новый, б/у.
Валюты: белорусский рубль, доллар, евро, российский рубль.
Предметы: легковой автомобиль, грузовой автомобиль,
спецтехника, оборудование, недвижимость, прочий транспорт,
тягач, полуприцеп, погрузчик, автобус.
Клиенты: физическое лицо, физлицо, физик, ИП, ипэшник,
индивидуальный предприниматель, юридическое лицо, юрлицо, юрик.
Термины: аванс, срок лизинга, ежемесячный платёж,
выкупной платёж, график платежей, нагрузка, переплата,
удорожание, общая сумма, итого, лизингодатель, лизингополучатель.
Графики: аннуитетный, аннуитет, линейный, дифференцированный.
Голосовая помощница Ксения, Ксения.
```

Character count: ~680. Estimated tokens: ~195. Safely under the 224-token cap.

### What was removed and why

| Removed | Rationale |
|---|---|
| Subaru, Lexus, Peugeot, Citroen, Volvo, Land Rover, Porsche, Chevrolet, Mitsubishi, Honda, Mazda, Ford, Opel, Renault, Nissan | Rare in Belarusian leasing market, <10 KB hits combined |
| Jetour, Omoda, Exeed, Changan, JAC | Less common Chinese brands, BYD/Geely/Chery/Haval cover the bulk |
| "Клиент звонит в компанию..." preamble | Narrative framing is wasted tokens |
| "десять тысяч, двадцать тысяч..." sum examples | Numbers are not a source of transcription errors; Whisper's number tokenization is good |
| "Термины:" section header | Whisper biases on vocabulary, not labels |

### What was added

| Added | Purpose |
|---|---|
| Ксения × 3 (including final) | Maximum bias; final position is guaranteed in window |
| линейный, аннуитетный, дифференцированный | Address transcript 18:17-21 failure |
| нагрузка, переплата, общая сумма, итого | Financial vocabulary client used |
| физик, юрик, ипэшник | Belarusian colloquial forms |
| Separated "Графики:" category | Explicit semantic grouping |

### Token budget verification

Pre-deploy: use `transformers`-provided Whisper tokenizer to count tokens in
the new prompt. Acceptance: `len(tokenizer.encode(prompt)) < 224`.

```python
from faster_whisper.tokenizer import Tokenizer
# Or alternative: openai-whisper tokenizer
# Count tokens, assert < 224, print result
```

Add this as a unit test to prevent regression.

## Files to change

- `rag_demo_system/services/whisper_server.py` (the `_DEFAULT_INITIAL_PROMPT` string)
- `rag_demo_system/tests/test_whisper_prompt.py` (new — token budget regression test)

## Testing

**Unit — token budget**
```python
def test_prompt_fits_token_budget():
    from services.whisper_server import _DEFAULT_INITIAL_PROMPT
    from faster_whisper.tokenizer import Tokenizer
    tokenizer = Tokenizer(...)
    tokens = tokenizer.encode(_DEFAULT_INITIAL_PROMPT)
    assert len(tokens) < 224, f"Prompt is {len(tokens)} tokens, cap is 224"
```

**Integration — domain term recognition**

Use a small set of recorded sample phrases (collect 10 short WAVs covering
the critical vocabulary):
1. "Ксения, я физлицо" → assert transcription contains "Ксения" and "физлицо".
2. "Хочу линейный график" → contains "линейный" and "график".
3. "Какая нагрузка по платежам" → contains "нагрузка".
4. "Аванс двадцать процентов, срок восемьдесят четыре месяца" →
   numbers parsed correctly.
5. Russian brand recognition: "БМВ бэха" → contains "BMW" or "бэха".

These can be recorded offline or generated via Silero TTS for regression.

**Regression — no ground lost**
- Re-transcribe 10 prior call segments using the new prompt; assert no term
  previously recognized is now worse.

## Metrics

Post-deploy:
- `whisper_asr_error_rate_on_domain_terms` — manual spot-check on first 20
  post-deploy calls. Target: name recognition correct ≥ 80% (up from ~20%).
- Log all transcriptions containing "Ксения" variants and graph-type words to
  `.state/stt_audit/{session_id}.jsonl` for V2 fine-tuning dataset.

## Risks

| Risk | Mitigation |
|---|---|
| Ксения-biasing corrupts other names (e.g. "Саша" → "Ксения") | Low: Whisper's language model is stronger than prompt bias on common names |
| Dropped brand triggers misrecognition on edge cases | Latin-script brand names (Subaru, Volvo) remain in Whisper's training data; only cyrillic aliases (Субару, Вольво) are lost |
| Token count miscounted | Unit test catches before deploy |

## Rollback

Single-file revert of `services/whisper_server.py`. No env dependency.

## V2 Queue (not in this spec)

See `project_stt_v2_roadmap.md` memory:
- Fine-tune `large-v3` via LoRA on 100+ hand-corrected call transcripts.
- Evaluate NeMo Canary-1B with hotword boosting.
- Re-evaluate SenseVoice.
