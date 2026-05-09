"""Test-suite fixtures.

Autouse: prevent the USD/BYN rate helper from hitting NBRB's public API
during unit tests. Without this, any test that runs through the
USD→BYN conversion path becomes (a) slow, (b) flaky on CI, and
(c) coupled to whatever today's official rate happens to be — which
breaks deterministic-amount assertions like
`80000 USD × 3.0 == 240000 BYN`.

Tests that specifically want to verify the NBRB fetch behavior
(test_usd_byn_rate.py) patch `urllib.request.urlopen` themselves and
override this fixture's effect by re-patching `_fetch_nbrb_usd_byn_rate`
back to the real implementation locally.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_live_nbrb(monkeypatch):
    import backend.profile_prompts as pp

    pp._USD_BYN_RATE_CACHE = None
    pp._USD_BYN_RATE_CACHE_TS = None
    pp._NBRB_RATE_CACHE.clear()

    monkeypatch.setattr(pp, "_fetch_nbrb_usd_byn_rate", lambda: None)
    monkeypatch.setattr(pp, "_fetch_nbrb_rate", lambda _currency: None)
    yield
    pp._USD_BYN_RATE_CACHE = None
    pp._USD_BYN_RATE_CACHE_TS = None
    pp._NBRB_RATE_CACHE.clear()
