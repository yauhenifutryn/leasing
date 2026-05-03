"""Prompt-content guards.

Active prompt is `config/system_prompt_ru_v2.txt` (see config/app.yaml).
The legacy v1 file is kept around for chat-only paths but the voice path
must not regress on Batch 3 client-feedback rules.
"""
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
PROMPT_V1 = CONFIG_DIR / "system_prompt_ru.txt"
PROMPT_V2 = CONFIG_DIR / "system_prompt_ru_v2.txt"


def _v2_text() -> str:
    return PROMPT_V2.read_text(encoding="utf-8")


def test_system_prompt_forbids_reasoning_output():
    prompt_text = PROMPT_V1.read_text(encoding="utf-8")
    # v1 prompt uses singular imperative "Не выводи рассуждения".
    assert "Не выводи рассуждения" in prompt_text


def test_system_prompt_requires_final_marker() -> None:
    prompt_text = PROMPT_V1.read_text(encoding="utf-8")
    assert "FINAL:" in prompt_text


# ── Bug 24: schedule type framing — output side uses lay phrasing ─────
# Banking jargon "аннуитет"/"линейный" are still accepted on input
# (utterance grounding) but the bot must SPEAK only the lay phrasing.

def test_v2_prompt_does_not_speak_annuity_or_linear() -> None:
    text = _v2_text().lower()
    assert "аннуитет" not in text, (
        "Bug 24: prompt must not instruct the bot to say 'аннуитет' — "
        "use 'равными платежами' instead."
    )
    assert "линейн" not in text, (
        "Bug 24: prompt must not instruct the bot to say 'линейный' — "
        "use 'с уменьшением суммы к концу срока' instead."
    )


def test_v2_prompt_uses_lay_schedule_phrasing() -> None:
    text = _v2_text()
    assert "равными платежами" in text
    assert "с уменьшением суммы к концу срока" in text


# ── Bug 13: gendered persona — neutral self-claims, feminine grammar ──
# Source: live call 15:34:53 — bot replied "у меня есть пол, я женщина".
# Right behavior: stay neutral about personhood claims, but keep feminine
# grammatical agreement on first-person verbs/adjectives.

def test_v2_prompt_blocks_gendered_self_claims() -> None:
    text = _v2_text().lower()
    # No personhood assertion that the bot IS a woman.
    assert "вы — женщина" not in text
    assert "вы женщина" not in text
    # Old "if asked your gender, you ARE a woman" rule must be gone.
    assert "если клиент спрашивает ваш пол, вы женщина" not in text


def test_v2_prompt_keeps_feminine_grammar_forms() -> None:
    text = _v2_text()
    # The feminine-form instructions stay — they're about grammar, not
    # personhood. рада / готова / смогла / могла are listed in the prompt
    # as the required first-person forms.
    for form in ("рада", "готова", "смогла", "могла"):
        assert form in text, f"Bug 13: feminine grammar form '{form}' must stay"


# ── Bug 16: Belarus-only scope ─────────────────────────────────────────
# Live call ccf0139a 14:37:12: caller asked "Что по лизингу в России?"
# and the bot answered with generic Russian-context info. Active prompt
# must restrict the consultation scope to Belarus and gracefully
# redirect questions about other countries.

def test_v2_prompt_restricts_scope_to_belarus() -> None:
    text = _v2_text().lower()
    assert "только по лизингу в беларуси" in text, (
        "Bug 16: prompt must explicitly scope consultation to Belarus."
    )


def test_v2_prompt_handles_other_country_questions() -> None:
    text = _v2_text().lower()
    # The prompt must mention at least one non-Belarus country in the
    # redirect rule so the bot has an explicit pattern for the case.
    assert "россии" in text or "россия" in text or "казахстан" in text


# ── Bug 17: online submission only ─────────────────────────────────────
# Live call ccf0139a 14:41:40: bot offered "отправить документы почтой".
# Real flow is online via личный кабинет на mikro-leasing.by. The prompt
# must positively name the online channel and ban postal/courier/email.

def test_v2_prompt_requires_online_only_submission() -> None:
    text = _v2_text().lower()
    assert "только онлайн" in text, (
        "Bug 17: prompt must positively assert online-only submission."
    )
    assert "личный кабинет" in text
    assert "mikro-leasing.by" in text


def test_v2_prompt_bans_offline_submission_channels() -> None:
    text = _v2_text().lower()
    # The ban list must explicitly name post / courier / email so the
    # bot can't slip into "отправьте документы по почте" again.
    assert "почт" in text  # почтой / по почте / почту
    assert "курьер" in text
    assert "email" in text or "имейл" in text


def test_v2_prompt_default_to_online_for_documents() -> None:
    # Bug 17 (live calls b5d70d6a + 099bfb78 + bf527f44 2026-05-03):
    # ambiguous "отправить документы почтой?" turns must default to the
    # online-submission answer. The unconditional hard ban from Patch F
    # was wrong — KB legitimately documents postal-delivery for the
    # post-buyout-vehicle-docs flow. This rule is the soft-default
    # version: online unless the user names a trigger.
    text = _v2_text().lower()
    # Header for the default-online rule.
    assert "по умолчанию онлайн" in text
    # The canonical default reply must be locked.
    assert "документы подаются онлайн через личный кабинет" in text
    # Channel ban (default behavior): no postal/courier/email by default.
    assert "не предлагай по умолчанию почту" in text


def test_v2_prompt_lists_explicit_postal_triggers() -> None:
    # Postal delivery is allowed ONLY when the caller explicitly names
    # one of these triggers in the current turn — anchors the LLM on
    # surface signals so the rule is reproducible.
    text = _v2_text().lower()
    assert "триггер исключения" in text
    # All canonical triggers should appear at least once.
    for trigger in ("гаи", "после выкупа", "пакет документов на машину"):
        assert trigger in text, f"postal-trigger phrase '{trigger}' missing from prompt"


def test_v2_prompt_keeps_online_submission_rule() -> None:
    text = _v2_text().lower()
    assert "только онлайн" in text
    assert "личный кабинет" in text
    assert "mikro-leasing.by" in text


# ── Bug 20: over-escalation to specialist ──────────────────────────────
# Client review: "часто отправляет к специалисту". The bot's stock
# fallback for any uncertain question is "уточню у специалиста". The
# prompt half of the fix tightens the fallback wording — prefer "I
# don't have that detail right now" + a concrete actionable next step
# (online submission link, calculator, contact phone) over a stock
# escalation. KB enrichment is a parallel track (deferred).

def test_v2_prompt_documents_preferred_fallback_phrase() -> None:
    text = _v2_text().lower()
    assert "пока нет точной информации" in text, (
        "Bug 20: prompt must lock in the preferred non-escalating "
        "fallback phrase 'пока нет точной информации'."
    )


def test_v2_prompt_lists_actionable_alternatives_to_escalation() -> None:
    text = _v2_text().lower()
    # The fallback rule must name at least one concrete next step the
    # bot can offer instead of escalating: online submission, calculator,
    # or the contact phone.
    has_online = "онлайн" in text and ("личный кабинет" in text or "mikro-leasing.by" in text)
    has_calc = "калькулятор" in text
    has_phone = "+375 17 322 77 00" in text or "+375" in text
    assert has_online or has_calc or has_phone


# ── Bug 20 follow-up: unknown abbreviations rule ──────────────────────
# Live call b5d70d6a 2026-05-03 19:47: caller said "АПДН"; bot replied
# "не встречала... уточню у специалиста" — escalated immediately. The
# preferred behavior is to acknowledge the gap, offer a similar known
# term, ask the user to rephrase, and only escalate as a last resort.

def test_v2_prompt_handles_unknown_abbreviations_without_escalating() -> None:
    text = _v2_text().lower()
    # Header + key phrasing for the unknown-abbreviation rule.
    assert "незнакомые аббревиатуры" in text or "незнакомых аббревиатур" in text
    # Must explicitly say "do NOT escalate by default".
    assert "не переключайте" in text or "не переключать" in text


def test_v2_prompt_unknown_abbrev_offers_similar_known_term() -> None:
    text = _v2_text().lower()
    # The rule must explicitly tell the LLM to offer похожий термин
    # from the leasing domain (ПДН, ИНН, УНП) so the bot can guess.
    assert "возможно, вы имели в виду" in text
    # At least one canonical similar abbreviation listed inline.
    assert "пдн" in text or "инн" in text or "унп" in text


# ── Bug 23: encourage immediate document submission ────────────────────
# Client wants the bot to actively upsell: "если подадите документы
# прямо сейчас онлайн, решение придёт в течение одного рабочего дня".
# Lives in the prompt because the deterministic post-calc renderer is
# now terse (Bug 25); per the handover, the upsell goes prompt-side, not
# in profile_prompts.render_calc_result.

def test_v2_prompt_carries_immediate_submission_upsell() -> None:
    text = _v2_text().lower()
    # Time-promise phrase + online channel reference, both required so
    # the bot can't ship one half without the other.
    has_time = "одного рабочего дня" in text or "один рабочий день" in text
    assert has_time, "Bug 23: prompt must include the 'one business day' time promise."
    assert "подадите документы" in text or "оформите заявку онлайн" in text


def test_v2_prompt_upsell_lives_in_post_calc_context() -> None:
    text = _v2_text()
    # Locate the upsell line and verify it sits near a post-calc anchor
    # so the LLM associates it with the right moment in the flow.
    assert "одного рабочего дня" in text.lower() or "один рабочий день" in text.lower()
    # The line must mention "после расчёта" / "после успешного расчёта"
    # OR live inside the existing "После успешного расчёта" block.
    upsell_idx = text.lower().find("одного рабочего дня")
    if upsell_idx < 0:
        upsell_idx = text.lower().find("один рабочий день")
    # Window of ~600 chars around the upsell must mention расчёт / SMS
    # — otherwise it'll fire at the wrong moment in the conversation.
    window = text[max(0, upsell_idx - 600): upsell_idx + 600].lower()
    assert "расчёт" in window or "расчета" in window or "смс" in window or "сейчас" in window


# ── Bug 26: TTS abbreviation pronunciation block ───────────────────────
# Live call 5746bfec (post-KB-swap, 2026-05-03): bot read "ПДН" as
# "пэ-дэ-эн-эн" (extra trailing "эн") and "РФ" as "эр-эф-эф" — Silero
# letter-spelling is robotic and adds extra phonemes when given Cyrillic
# uppercase tokens it doesn't have explicit pronunciation rules for.
# The fix is to pre-spell the abbreviations in the prompt so the LLM
# emits the phonetic form to TTS.

def test_v2_prompt_has_abbreviation_pronunciation_block() -> None:
    text = _v2_text()
    # Header marker for the block.
    assert "ПРОИЗНОШЕНИЕ" in text.upper(), (
        "Bug 26: prompt must have a 'ПРОИЗНОШЕНИЕ АБРЕВИАТУР' block."
    )


def test_v2_prompt_letter_form_acronyms() -> None:
    text = _v2_text().lower()
    # Letter-by-letter Russian: ПДН / РФ / ВНЖ.
    assert "пэ-дэ-эн" in text
    assert "эр-эф" in text
    assert "вэ-эн-жэ" in text


def test_v2_prompt_word_form_acronyms() -> None:
    text = _v2_text().lower()
    # Word-form acronyms: КАСКО / ОСАГО — read as words, not letter-by-letter.
    assert "каско" in text
    assert "осаго" in text


# ── Bug 19: working-hours-aware specialist handoff (prompt half) ──────

def test_v2_prompt_references_working_hours_marker() -> None:
    text = _v2_text()
    # The prompt must instruct the LLM to read the runtime
    # "[Время сейчас: ...]" tag injected by app.py before promising
    # specialist transfers.
    assert "Время сейчас" in text
    assert "РАБОЧЕЕ" in text and "НЕРАБОЧЕЕ" in text


def test_v2_prompt_carries_offhours_redirect_phrase() -> None:
    text = _v2_text()
    # Locks in the canonical off-hours redirect wording so any future
    # rephrasing has to update both the helper marker and the prompt
    # rule together.
    assert "перезвонят в рабочее время" in text


def test_v2_prompt_currency_pronunciation() -> None:
    text = _v2_text().lower()
    # Currency rules: BYN / USD / EUR / RUB must NOT be spelled
    # letter-by-letter. The block lists the canonical full-name
    # pronunciation for each.
    assert "белорусские рубли" in text
    assert "доллары" in text
    assert "евро" in text
    # Explicit ban on letter-by-letter currencies.
    has_ban = (
        "не произноси" in text
        or "не озвучивай" in text
        or "запрещено произносить" in text
    )
    assert has_ban, (
        "Bug 26: prompt must explicitly ban letter-by-letter "
        "currency pronunciation (BYN/USD/EUR)."
    )
