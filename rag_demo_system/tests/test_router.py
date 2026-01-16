from backend.router import route_non_rag


def test_router_no_llm_returns_none():
    assert route_non_rag("Здравствуйте") is None
    assert route_non_rag("Как вас зовут?") is None
    assert route_non_rag("Что вы думаете о политике?") is None
