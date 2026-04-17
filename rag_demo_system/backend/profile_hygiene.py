"""Guard rails on classifier profile_patches before merging into ClientProfile.

Drops or normalizes patches that would corrupt the profile from noise
utterances, bot-name echoes, malformed enum values, or out-of-MVP-range numbers.
"""

from __future__ import annotations

from typing import Any

MVP_PREPAID_RANGE = (0.0, 40.0)
MVP_TERM_RANGE = (12, 84)


def _normalize_client_type(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    if s in {"физлицо", "физик", "физ. лицо", "физическое лицо", "физическое"}:
        return "Физическое лицо"
    if s in {"ип", "ипэшник", "индивидуальный предприниматель", "самозанятый"}:
        return "ИП"
    if s in {"юрлицо", "юридическое лицо", "ооо", "оао", "зао", "организация", "компания", "юридическое"}:
        return "Юридическое лицо"
    return None


def _normalize_currency(patch_value: Any, utterance: str) -> str | None:
    if not isinstance(patch_value, str):
        return None
    s = patch_value.strip().upper()
    if s in {"BYN", "BLR"}:
        return "BYN"
    if s == "USD":
        return "USD"
    if s == "EUR":
        return "EUR"
    if s == "RUB":
        low = (utterance or "").lower()
        if "росси" in low or "российск" in low:
            return "RUB"
        # Bare "рубли" in Belarus context -> BYN
        return "BYN"
    return None


def filter_patches(
    patches: dict[str, Any],
    utterance: str,
    bot_name: str = "Ксения",
) -> dict[str, Any]:
    """Run classifier-emitted patches through hygiene checks. Returns filtered dict."""
    if not patches:
        return {}

    # Drop everything when utterance is noise (fewer than 2 non-digit tokens).
    tokens = [t for t in (utterance or "").strip().split() if not t.isdigit()]
    if len(tokens) < 2:
        return {}

    out: dict[str, Any] = dict(patches)

    # Drop bot name echoed as user name.
    if isinstance(out.get("name"), str) and out["name"].strip().lower() == bot_name.lower():
        out.pop("name")

    # Normalize / validate client_type.
    if "client_type" in out:
        ct = _normalize_client_type(out["client_type"])
        if ct is None:
            out.pop("client_type")
        else:
            out["client_type"] = ct

    # Normalize currency.
    if "currency" in out:
        cur = _normalize_currency(out["currency"], utterance)
        if cur is None:
            out.pop("currency")
        else:
            out["currency"] = cur

    # Prepaid range.
    if "prepaid_pct" in out:
        try:
            p = float(out["prepaid_pct"])
            if not (MVP_PREPAID_RANGE[0] <= p <= MVP_PREPAID_RANGE[1]):
                out.pop("prepaid_pct")
        except (TypeError, ValueError):
            out.pop("prepaid_pct")

    # Term range.
    if "term_months" in out:
        try:
            t = int(out["term_months"])
            if not (MVP_TERM_RANGE[0] <= t <= MVP_TERM_RANGE[1]):
                out.pop("term_months")
        except (TypeError, ValueError):
            out.pop("term_months")

    return out
