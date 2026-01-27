from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import router as router_mod
from backend.router import route_non_rag
from backend.llm import LLMResponse


def test_router_fallback_greeting_without_llm():
    decision = route_non_rag("привет")
    assert decision is not None
    assert decision.kind == "greeting"
    assert "Здравствуйте" in decision.response


def test_router_fallback_identity_without_llm():
    decision = route_non_rag("Как вас зовут?")
    assert decision is not None
    assert decision.kind == "identity"
    assert "Микро Лизинг" in decision.response


def test_router_does_not_treat_questions_as_greeting():
    decision = route_non_rag("привет, какие условия лизинга?")
    assert decision is None


def test_router_llm_response_used_when_available(monkeypatch):
    def fake_call(*args, **kwargs):
        return LLMResponse(text="LLM GREET", raw={})

    monkeypatch.setattr(router_mod, "call_openai_compatible", fake_call)

    decision = route_non_rag("привет", base_url="http://127.0.0.1:1", model="x")
    assert decision is not None
    assert decision.kind == "greeting"
    assert decision.response == "LLM GREET"


def test_router_fallback_when_llm_unavailable(monkeypatch):
    def fake_call(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(router_mod, "call_openai_compatible", fake_call)

    decision = route_non_rag("Как вас зовут?", base_url="http://127.0.0.1:1", model="x")
    assert decision is not None
    assert decision.kind == "identity"
    assert "Микро Лизинг" in decision.response


def test_router_uncertain_with_llm_down_returns_none():
    decision = route_non_rag("Какая завтра погода?", base_url="http://127.0.0.1:1", model="x")
    assert decision is None
