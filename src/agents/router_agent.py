"""Router agent: classify a query into one specialist intent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..core.logging_utils import get_logger
from ..core.state import Intent
from ..llm.base import LLMProvider

logger = get_logger(__name__)


class RouterDecision(BaseModel):
    """Structured routing output: which agent and why."""

    intent: Intent
    reason: str = ""


_SYSTEM = (
    "You route a user's finance message to ONE specialist agent. "
    "Choose the single best intent:\n"
    "- qa: general finance concepts/education (what is, how does, "
    "explain, compare).\n"
    "- portfolio: analyze the user's own holdings, allocation, "
    "diversification, or risk.\n"
    "- market: live price/quote/trend for specific tickers or "
    "indices.\n"
    "- goal: savings/retirement planning, projections, 'how much' "
    "or 'how long' questions.\n"
    "- smalltalk: greetings, thanks, or questions about you."
)


class RouterAgent:
    """LLM-based intent classifier (keyword-based on the mock)."""

    name = "router"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def route(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        try:
            decision = self.llm.structured(RouterDecision, _SYSTEM, query)
            intent: str = decision.intent
            reason = decision.reason
        except Exception as exc:
            logger.warning("Router failed, defaulting to qa: %s", exc)
            intent, reason = "qa", "router error fallback"
        trace = state.get("agent_trace", []) + [f"router -> {intent}"]
        return {
            "intent": intent,
            "route_reason": reason,
            "agent_trace": trace,
        }
