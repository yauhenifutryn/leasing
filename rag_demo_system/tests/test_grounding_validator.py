from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.grounding_validator import extract_high_risk_facts, check_grounded, replace_ungrounded


def test_extracts_belarus_phone():
    facts = extract_high_risk_facts("Позвоните +375 17 322 77 00 в офис.")
    assert any(f["type"] == "phone" and "+375" in f["value"] for f in facts)


def test_extracts_street_address():
    facts = extract_high_risk_facts("Офис: проспект Победителей, 57.")
    assert any(f["type"] == "street_address" and "Победителей" in f["value"] for f in facts)


def test_extracts_street_abbrev():
    facts = extract_high_risk_facts("Офис: ул. Немига, 24.")
    assert any(f["type"] == "street_address" and "Немига" in f["value"] for f in facts)


def test_extracts_russian_tri_name():
    facts = extract_high_risk_facts("Директор — Вадим Николаевич Дедков.")
    assert any(f["type"] == "personal_name" and "Вадим Николаевич Дедков" in f["value"] for f in facts)


def test_check_grounded_true_when_in_chunks():
    chunks = ["Наш офис находится по адресу проспект Победителей, 57, Минск."]
    assert check_grounded({"type": "street_address", "value": "проспект Победителей, 57"}, chunks) is True


def test_check_grounded_false_when_absent():
    chunks = ["Наш офис находится по адресу проспект Победителей, 57."]
    assert check_grounded({"type": "street_address", "value": "ул. Немига, 24"}, chunks) is False


def test_replace_ungrounded_address():
    response = "В Минске офис находится по адресу: Минск, ул. Немига, 24."
    chunks = ["Наш офис: проспект Победителей, 57."]
    out = replace_ungrounded(response, chunks)
    assert "Немига" not in out
    assert "уточните у специалиста" in out.lower() or "+375" in out


def test_replace_leaves_grounded_facts():
    response = "Офис: проспект Победителей, 57."
    chunks = ["проспект Победителей, 57"]
    out = replace_ungrounded(response, chunks)
    assert "Победителей" in out


def test_strips_ungrounded_typical_percent():
    # "обычно 10%" in response but not in chunks -> stripped
    resp = "Аванс от 0% до 40%. На обычных условиях обычно от 10%."
    chunks = ["По условиям калькулятора аванс может быть от 0% до 40% от стоимости."]
    out = replace_ungrounded(resp, chunks)
    assert "10" not in out
    assert "обычно" not in out.lower() or "обычных условиях" not in out.lower()


def test_keeps_grounded_percent_when_anchor_in_chunk():
    # "обычно 30%" with "обычно 30%" in a chunk -> keep
    resp = "Для ИП обычно 30%."
    chunks = ["Для индивидуальных предпринимателей обычно 30% аванс по калькулятору."]
    out = replace_ungrounded(resp, chunks)
    assert "30" in out


def test_keeps_non_anchored_percent():
    # Plain "40%" with no anchor word -> not checked by typical_percent rule
    resp = "Аванс до 40%."
    chunks = ["Аванс до 40%."]
    out = replace_ungrounded(resp, chunks)
    assert "40" in out
