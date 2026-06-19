"""End-to-end workflow test (offline, mock provider)."""

from src.llm.mock_provider import MockProvider
from src.workflow.graph import FinanceAssistant


def _assistant() -> FinanceAssistant:
    llm = MockProvider(router_model="r", default_model="d")
    return FinanceAssistant(llm=llm)


def test_qa_turn_has_response_and_disclaimer() -> None:
    result = _assistant().ask("What is a bond?")
    response = result.get("response", "")
    assert response
    # finalize appends the educational disclaimer
    assert "education" in response.lower()
    trace = result.get("agent_trace", [])
    assert any("qa" in step for step in trace)
