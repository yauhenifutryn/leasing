from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.consent import detect_consent


def test_detect_consent_granted_yes() -> None:
    assert detect_consent("да") == "granted"


def test_detect_consent_granted_confirm() -> None:
    assert detect_consent("подтверждаю") == "granted"


def test_detect_consent_denied() -> None:
    assert detect_consent("не согласен") == "denied"
