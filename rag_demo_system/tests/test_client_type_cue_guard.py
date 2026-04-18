"""Fix 26: `filter_patches` must reject classifier-inferred `client_type`
when the user utterance contains no explicit client-type keyword.

Before: classifier defaults `client_type="Физическое лицо"` from prompt
examples on early turns, orchestrator silently labels the caller, caller
never asked. Sometimes the assumption is wrong ("truck + individual" is
the invalid combo the API rejects).

After: patch only passes when the utterance contains physлицо / юрлицо /
ИП / компания / организация / etc. Otherwise orchestrator falls through
to its clarification path and asks the client explicitly.
"""

from __future__ import annotations

from backend.profile_hygiene import filter_patches, utterance_has_client_type_cue


def test_cue_detection_positive_cases():
    assert utterance_has_client_type_cue("я физлицо")
    assert utterance_has_client_type_cue("Я физическое лицо")
    assert utterance_has_client_type_cue("мы юрлицо, ООО Ромашка")
    assert utterance_has_client_type_cue("от компании")
    assert utterance_has_client_type_cue("на организацию")
    assert utterance_has_client_type_cue("ИП")
    assert utterance_has_client_type_cue("я ИП, Иванов Иван")
    assert utterance_has_client_type_cue("самозанятый")
    assert utterance_has_client_type_cue("индивидуальный предприниматель")


def test_cue_detection_negative_cases():
    # Requests for leasing — no client-type words — must NOT signal a cue.
    assert not utterance_has_client_type_cue("хочу машину в лизинг")
    assert not utterance_has_client_type_cue("лизинг на грузовой автомобиль")
    assert not utterance_has_client_type_cue("80000 рублей, новый")
    assert not utterance_has_client_type_cue("36 месяцев")
    assert not utterance_has_client_type_cue("")
    assert not utterance_has_client_type_cue("да, всё верно")


def test_filter_patches_drops_client_type_without_cue():
    """Classifier inferred Физическое лицо from 'хочу машину' — no cue — drop."""
    patches = {
        "client_type": "Физическое лицо",
        "subject": "Легковой автомобиль",
    }
    out = filter_patches(patches, utterance="Хочу взять машину в лизинг")
    assert "client_type" not in out, out
    # Sibling patches unaffected.
    assert out.get("subject") == "Легковой автомобиль"


def test_filter_patches_keeps_client_type_with_explicit_cue():
    patches = {"client_type": "Физическое лицо"}
    out = filter_patches(patches, utterance="я физическое лицо, хочу машину")
    assert out.get("client_type") == "Физическое лицо"


def test_filter_patches_keeps_client_type_with_yurlico_cue():
    patches = {"client_type": "Юридическое лицо"}
    out = filter_patches(patches, utterance="мы ООО, берем грузовой в лизинг")
    assert out.get("client_type") == "Юридическое лицо"


def test_filter_patches_drops_client_type_on_truck_utterance():
    """The canonical failure case: bot infers Физическое лицо from 'грузовой'
    despite 'грузовой' actually implying юрлицо. With Fix 26, classifier can
    still emit the wrong patch but hygiene rejects it for lack of cue.
    """
    patches = {
        "client_type": "Физическое лицо",
        "subject": "Грузовой автомобиль",
    }
    out = filter_patches(patches, utterance="хочу грузовой в лизинг на 36 месяцев")
    assert "client_type" not in out
    assert out.get("subject") == "Грузовой автомобиль"


def test_filter_patches_keeps_client_type_in_slot_fill_reply():
    """Single-word slot-fill reply to 'вы физлицо или юрлицо?' — still valid."""
    patches = {"client_type": "Физическое лицо"}
    out = filter_patches(patches, utterance="физлицо")
    assert out.get("client_type") == "Физическое лицо"
