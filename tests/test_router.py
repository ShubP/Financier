"""Router intent-classification tests (offline, mock provider)."""

from src.agents.router_agent import RouterAgent
from src.llm.mock_provider import MockProvider


def _router() -> RouterAgent:
    llm = MockProvider(router_model="r", default_model="d")
    return RouterAgent(llm)


def test_routes_portfolio() -> None:
    out = _router().route({"query": "Analyze my portfolio allocation"})
    assert out["intent"] == "portfolio"


def test_routes_market() -> None:
    out = _router().route({"query": "What's the price of AAPL today?"})
    assert out["intent"] == "market"


def test_routes_goal() -> None:
    out = _router().route(
        {"query": "How much to save to retire in 20 years?"}
    )
    assert out["intent"] == "goal"


def test_routes_qa_default() -> None:
    out = _router().route({"query": "Explain what an index fund is"})
    assert out["intent"] == "qa"
