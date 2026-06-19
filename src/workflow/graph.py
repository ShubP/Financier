"""Multi-agent workflow built on a LangGraph StateGraph.

Flow: router -> one specialist (qa | market | portfolio | goal |
smalltalk) -> finalize. The router classifies intent; each specialist
writes its contribution into shared state; finalize guarantees a
response and appends the educational disclaimer.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from ..agents import (
    GoalAgent,
    MarketAgent,
    PortfolioAgent,
    QAAgent,
    RouterAgent,
)
from ..core.config import load_config
from ..core.logging_utils import get_logger
from ..core.state import AgentState, new_state
from ..data.market_data import get_market_data
from ..llm.factory import build_provider
from ..rag.knowledge_base import get_knowledge_base

logger = get_logger(__name__)

_SPECIALISTS = {"qa", "market", "portfolio", "goal", "smalltalk"}

_SMALLTALK = (
    "Hi! I'm Financier, your AI finance study buddy. I can explain "
    "investing concepts, analyze a portfolio's allocation and risk, "
    "pull live market quotes, and project savings goals. What would "
    "you like to explore?"
)


class FinanceAssistant:
    """Builds and runs the compiled multi-agent graph."""

    def __init__(
        self,
        llm: Any | None = None,
        kb: Any | None = None,
        market: Any | None = None,
    ) -> None:
        cfg = load_config()
        self.llm = llm or build_provider(cfg)
        self.kb = kb if kb is not None else get_knowledge_base(cfg)
        self.market = market or get_market_data(cfg)
        self.disclaimer = str(cfg.get("app.disclaimer", ""))

        self.router = RouterAgent(self.llm)
        self.qa = QAAgent(self.llm, self.kb)
        self.market_agent = MarketAgent(self.llm, self.market)
        self.portfolio = PortfolioAgent(self.llm, self.market)
        self.goal = GoalAgent(self.llm)
        self._graph = self._build_graph()

    @property
    def provider_name(self) -> str:
        return self.llm.name

    # --- graph wiring ---
    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("router", lambda s: self.router.route(s))
        g.add_node("qa", lambda s: self.qa.run(s))
        g.add_node("market", lambda s: self.market_agent.run(s))
        g.add_node("portfolio", lambda s: self.portfolio.run(s))
        g.add_node("goal", lambda s: self.goal.run(s))
        g.add_node("smalltalk", self._smalltalk)
        g.add_node("finalize", self._finalize)

        g.set_entry_point("router")
        g.add_conditional_edges(
            "router",
            self._route,
            {
                "qa": "qa",
                "market": "market",
                "portfolio": "portfolio",
                "goal": "goal",
                "smalltalk": "smalltalk",
            },
        )
        for node in ("qa", "market", "portfolio", "goal", "smalltalk"):
            g.add_edge(node, "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    @staticmethod
    def _route(state: dict[str, Any]) -> str:
        intent = state.get("intent", "qa")
        return intent if intent in _SPECIALISTS else "qa"

    def _smalltalk(self, state: dict[str, Any]) -> dict[str, Any]:
        trace = state.get("agent_trace", []) + ["smalltalk"]
        return {
            "response": _SMALLTALK,
            "agent_trace": trace,
            "sources": [],
        }

    def _finalize(self, state: dict[str, Any]) -> dict[str, Any]:
        response = state.get("response") or (
            "Sorry, I couldn't generate a response. Please rephrase."
        )
        if self.disclaimer and state.get("intent") != "smalltalk":
            response = f"{response}\n\n---\n*{self.disclaimer}*"
        return {"response": response}

    # --- public entry point ---
    def ask(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> AgentState:
        """Run one user turn through the graph and return the state."""
        state = new_state(query, history, profile)
        try:
            return self._graph.invoke(state)
        except Exception as exc:  # never crash the UI on a bad turn
            logger.exception("Workflow failed: %s", exc)
            fallback = (
                "Something went wrong while processing that. "
                "Please try again."
            )
            return {**state, "response": fallback, "error": str(exc)}


_INSTANCE: FinanceAssistant | None = None


def get_assistant() -> FinanceAssistant:
    """Return a process-wide :class:`FinanceAssistant` singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = FinanceAssistant()
    return _INSTANCE
