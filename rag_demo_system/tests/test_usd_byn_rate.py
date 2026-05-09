"""USD/BYN rate sourcing — pulls live from NBRB public API with a
TTL-cached fallback to the static config rate.

The static 3.0 stub embedded in classifier readbacks ("по курсу 3 к 1")
looks unprofessional in client demos. The National Bank of Belarus
publishes the official daily rate at:

  https://api.nbrb.by/exrates/rates/USD?parammode=2

These tests lock in:
  - a successful NBRB fetch parses Cur_OfficialRate / Cur_Scale,
  - network/parse failures fall back to the previous cached value,
  - cold-start failures fall back to the constant,
  - cache holds for the TTL so we don't hammer NBRB on every USD turn.
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

# Capture the real NBRB fetchers BEFORE conftest's autouse fixture
# stubs them. After the per-currency refactor the legacy
# `_fetch_nbrb_usd_byn_rate` is a thin wrapper around `_fetch_nbrb_rate`,
# so we need to restore both for the USD-cache path to actually fetch.
_REAL_FETCH = pp._fetch_nbrb_usd_byn_rate
_REAL_FETCH_BY_CCY = pp._fetch_nbrb_rate


@pytest.fixture(autouse=True)
def _restore_real_fetch(monkeypatch):
    """Override conftest's stub so this file exercises the real fetch
    (with urllib mocked at the network layer in each test)."""
    monkeypatch.setattr(pp, "_fetch_nbrb_usd_byn_rate", _REAL_FETCH)
    monkeypatch.setattr(pp, "_fetch_nbrb_rate", _REAL_FETCH_BY_CCY)
    yield


def _reset_cache():
    pp._USD_BYN_RATE_CACHE = None
    pp._USD_BYN_RATE_CACHE_TS = None
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


def test_nbrb_fetch_parses_official_rate_with_scale_1():
    _reset_cache()
    payload = {
        "Cur_ID": 145,
        "Cur_Abbreviation": "USD",
        "Cur_Scale": 1,
        "Cur_OfficialRate": 3.0851,
    }
    with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
        rate = pp._fetch_nbrb_usd_byn_rate()
    assert rate == 3.0851


def test_nbrb_fetch_divides_by_scale():
    # Some currencies publish per-100; USD is per-1, but lock in the math.
    _reset_cache()
    payload = {"Cur_Scale": 100, "Cur_OfficialRate": 308.51}
    with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
        rate = pp._fetch_nbrb_usd_byn_rate()
    assert rate == pytest.approx(3.0851)


def test_nbrb_fetch_returns_none_on_network_error():
    _reset_cache()
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        rate = pp._fetch_nbrb_usd_byn_rate()
    assert rate is None


def test_nbrb_fetch_returns_none_on_malformed_json():
    _reset_cache()

    class _Bad:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"<html>not json</html>"

    with patch("urllib.request.urlopen", return_value=_Bad()):
        rate = pp._fetch_nbrb_usd_byn_rate()
    assert rate is None


def test_get_rate_uses_live_nbrb_when_available():
    _reset_cache()
    payload = {"Cur_Scale": 1, "Cur_OfficialRate": 3.1234}
    with patch("urllib.request.urlopen", return_value=_FakeResp(payload)):
        rate = pp._get_usd_byn_rate()
    assert rate == 3.1234


def test_get_rate_falls_back_to_settings_on_cold_start_when_nbrb_down():
    _reset_cache()
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        rate = pp._get_usd_byn_rate()
    # Whatever settings say (default 3.0 in test env). Must NOT raise.
    assert isinstance(rate, float)
    assert rate > 0


def test_get_rate_caches_within_ttl():
    _reset_cache()
    payload = {"Cur_Scale": 1, "Cur_OfficialRate": 3.05}
    with patch("urllib.request.urlopen", return_value=_FakeResp(payload)) as mock_open:
        first = pp._get_usd_byn_rate()
        second = pp._get_usd_byn_rate()
    assert first == 3.05
    assert second == 3.05
    # One real fetch, the second call hits the cache.
    assert mock_open.call_count == 1
