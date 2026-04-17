from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.profile_hygiene import filter_patches


def test_drops_patches_from_noise_short_utterance():
    # "ОООООО" is 1 non-digit token of questionable content
    assert filter_patches({"client_type": "Юридическое лицо"}, "ОООООО") == {}


def test_drops_bot_name_as_user_name():
    patches = {"name": "Ксения"}
    out = filter_patches(patches, "Ксения, ладно, а что такое нагрузка?", bot_name="Ксения")
    assert "name" not in out


def test_ipeshnik_preserved_as_ip():
    # Classifier emits "ИП" — filter should preserve it
    out = filter_patches({"client_type": "ИП"}, "я ипэшник", bot_name="Ксения")
    assert out.get("client_type") == "ИП"


def test_currency_bare_rubli_in_belarus_is_byn():
    out = filter_patches({"currency": "RUB"}, "ну давай в рублях")
    assert out.get("currency") == "BYN"


def test_currency_russian_rubles_is_rub():
    out = filter_patches({"currency": "RUB"}, "в российских рублях")
    assert out.get("currency") == "RUB"


def test_prepaid_out_of_range_is_dropped():
    out = filter_patches({"prepaid_pct": 50}, "аванс пятьдесят процентов")
    assert "prepaid_pct" not in out


def test_prepaid_in_range_is_kept():
    out = filter_patches({"prepaid_pct": 20}, "аванс двадцать процентов")
    assert out.get("prepaid_pct") == 20


def test_term_out_of_range_is_dropped():
    out = filter_patches({"term_months": 300}, "триста месяцев")
    assert "term_months" not in out
