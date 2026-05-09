"""Per-currency NBRB rate lookups.

The original `_get_usd_byn_rate()` helper hard-coded the USD path. To
support EUR/RUB calc paths we need a generalized `_get_nbrb_rate(currency)`
that:

  - hits the right NBRB endpoint per currency code
    (`/exrates/rates/{CODE}?parammode=2`),
  - caches per-currency independently (an EUR fetch must not poison the
    USD cache),
  - keeps the existing fallback chain (live → cached → settings → 3.0)
    on a per-currency basis,
  - returns 1.0 for BYN (BYN→BYN is identity, no API call).

Backward compat: `_get_usd_byn_rate()` and `_fetch_nbrb_usd_byn_rate()`
remain callable and behave exactly as before (USD currency).
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backend.profile_prompts as pp  # noqa: E402

# Capture real fetch BEFORE conftest's autouse stub.
_REAL_FETCH = getattr(pp, "_fetch_nbrb_rate", None) or pp._fetch_nbrb_usd_byn_rate


@pytest.fixture(autouse=True)
def _restore_real_fetch(monkeypatch):
    """Override conftest's stub so this file exercises the real fetch."""
    if hasattr(pp, "_fetch_nbrb_rate"):
        monkeypatch.setattr(pp, "_fetch_nbrb_rate", _REAL_FETCH)
    monkeypatch.setattr(pp, "_fetch_nbrb_usd_byn_rate", lambda: _REAL_FETCH("USD"))
    yield


def _reset_cache():
    pp._USD_BYN_RATE_CACHE = None
    pp._USD_BYN_RATE_CACHE_TS = None
    if hasattr(pp, "_NBRB_RATE_CACHE"):
        pp._NBRB_RATE_CACHE.clear()


class _FakeResp:
    def __init__(self, payload: dict):
        self._buf = BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._buf.read()


def test_fetch_eur_hits_eur_endpoint():
    _reset_cache()
    captured = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp({"Cur_Scale": 1, "Cur_OfficialRate": 3.55})

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        rate = pp._fetch_nbrb_rate("EUR")
    assert rate == 3.55
    assert "EUR" in captured["url"]
    assert "USD" not in captured["url"]


def test_fetch_rub_divides_by_scale_100():
    _reset_cache()
    payload = {"Cur_Scale": 100, "Cur_OfficialRate": 3.42}
    with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
        rate = pp._fetch_nbrb_rate("RUB")
    assert rate == pytest.approx(0.0342)


def test_fetch_byn_returns_one_without_api_call():
    """BYN→BYN is identity. Skip the API entirely."""
    _reset_cache()
    with patch("urllib.request.urlopen", side_effect=AssertionError("must not call")):
        rate = pp._get_nbrb_rate("BYN")
    assert rate == 1.0


def test_get_rate_caches_per_currency_independently():
    """Fetching EUR must not leak into the USD cache (or vice versa)."""
    _reset_cache()
    eur_payload = {"Cur_Scale": 1, "Cur_OfficialRate": 3.55}
    usd_payload = {"Cur_Scale": 1, "Cur_OfficialRate": 3.05}

    call_log = []

    def _fake_urlopen(req, timeout=None):
        call_log.append(req.full_url)
        if "EUR" in req.full_url:
            return _FakeResp(eur_payload)
        if "USD" in req.full_url:
            return _FakeResp(usd_payload)
        raise ValueError(f"unexpected url: {req.full_url}")

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        eur1 = pp._get_nbrb_rate("EUR")
        usd1 = pp._get_nbrb_rate("USD")
        eur2 = pp._get_nbrb_rate("EUR")  # should hit cache
        usd2 = pp._get_nbrb_rate("USD")  # should hit cache

    assert eur1 == 3.55
    assert usd1 == 3.05
    assert eur2 == 3.55
    assert usd2 == 3.05
    # Two real fetches (one per currency), the next two hit cache.
    assert len(call_log) == 2


def test_get_rate_unknown_currency_falls_back_gracefully():
    """An unknown code returns a sensible float, never raises."""
    _reset_cache()
    with patch("urllib.request.urlopen", side_effect=OSError("404")):
        rate = pp._get_nbrb_rate("XYZ")
    assert isinstance(rate, float)
    assert rate > 0


def test_legacy_get_usd_byn_rate_still_returns_usd_rate():
    """Backward compat: the old helper must keep returning USD rate."""
    _reset_cache()
    payload = {"Cur_Scale": 1, "Cur_OfficialRate": 3.0851}
    with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
        rate = pp._get_usd_byn_rate()
    assert rate == 3.0851
