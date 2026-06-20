"""Router agent: classify a query into one specialist intent."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from ..core.logging_utils import get_logger
from ..core.state import Intent
from ..llm.base import LLMProvider

logger = get_logger(__name__)


class RouterDecision(BaseModel):
    """Structured routing output (kept for compatibility/tests)."""

    intent: Intent
    reason: str = ""


_INTENTS = ("portfolio", "market", "goal", "smalltalk", "qa")

# Keyword fallback used when the model doesn't return a clean label
# (and the basis for routing on the offline mock provider).
_KEYWORDS: list[tuple[str, list[str]]] = [
    (
        "portfolio",
        [
            "portfolio", "holding", "allocation", "diversif",
            "rebalanc", "asset mix", "my stocks", "my investments",
        ],
    ),
    (
        "market",
        [
            "price", "quote", "ticker", "market", "trend",
            "trading at", "stock price", "how is", "$",
        ],
    ),
    (
        "goal",
        [
            "goal", "retire", "save", "saving", "target",
            "projection", "afford", "how much", "year",
        ],
    ),
    ("smalltalk", ["hi", "hello", "hey", "thanks", "thank you"]),
]

_SYSTEM = (
    "Classify the user's finance message into exactly ONE intent. "
    "Reply with ONLY the intent word, nothing else.\n"
    "Intents:\n"
    "- qa: general finance concepts/education (what is, how does, "
    "explain, compare)\n"
    "- portfolio: analyze the user's own holdings, allocation, "
    "diversification, or risk\n"
    "- market: live price/quote/trend for tickers or indices\n"
    "- goal: savings/retirement planning, projections, 'how much' "
    "or 'how long'\n"
    "- smalltalk: greetings, thanks, or questions about you"
)


def _keyword_intent(query: str) -> str:
    text = query.lower()
    for intent, keywords in _KEYWORDS:
        if any(k in text for k in keywords):
            return intent
    return "qa"


class RouterAgent:
    """Classify intent via a one-word LLM label, with a keyword
    fallback that also drives routing on the offline mock provider."""

    name = "router"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def route(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        intent = ""
        try:
            reply = self.llm.classify(_SYSTEM, query)
            # Only trust a concise label: scan the first few words.
            words = re.findall(r"[a-z]+", reply.lower())
            for word in words[:4]:
                if word in _INTENTS:
                    intent = word
                    break
        except Exception as exc:
            logger.warning("Router classify failed: %s", exc)
        if not intent:
            intent = _keyword_intent(query)
        trace = state.get("agent_trace", []) + [f"router -> {intent}"]
        return {
            "intent": intent,
            "route_reason": "",
            "agent_trace": trace,
        }
