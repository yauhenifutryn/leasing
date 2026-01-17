from backend.router import route_non_rag


def test_router_no_llm_returns_none():
    assert route_non_rag("Здравствуйте") is None
    assert route_non_rag("Как вас зовут?") is None
    assert route_non_rag("Что вы думаете о политике?") is None


def test_router_heuristics_when_llm_available():
    decision = route_non_rag("привет", base_url="http://127.0.0.1:1", model="x")
    assert decision is not None
    assert decision.kind == "greeting"

    decision = route_non_rag("здравсствуйт", base_url="http://127.0.0.1:1", model="x")
    assert decision is not None
    assert decision.kind == "greeting"

    decision = route_non_rag("Как вас зовут?", base_url="http://127.0.0.1:1", model="x")
    assert decision is not None
    assert decision.kind == "identity"
