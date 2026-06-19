"""Portfolio analysis agent: allocation, diversification, risk."""

from __future__ import annotations

from typing import Any

from ..data.market_data import MarketData
from ..finance.portfolio import (
    Holding,
    PortfolioAnalysis,
    analyze_portfolio,
)
from ..llm.base import LLMProvider
from .base_agent import history_messages

_SYSTEM = (
    "You are Financier. Using the pre-computed analysis, explain the "
    "user's portfolio in plain language: total value, how it is "
    "allocated across asset classes, how diversified it is (a 0-100 "
    "score), its overall risk level, and the 1-3 most useful "
    "suggestions from the notes. Educational only — never tell the "
    "user to buy or sell specific securities."
)


def _summary(a: PortfolioAnalysis) -> str:
    alloc = ", ".join(
        f"{k} {v:.0f}%" for k, v in a.allocation_by_class.items()
    )
    lines = [
        f"Total value: ${a.total_value:,.2f}",
        f"Holdings: {a.num_holdings}",
        f"Allocation: {alloc or 'n/a'}",
        f"Diversification score: {a.diversification_score}/100",
        f"Largest position: {a.concentration_top_pct:.0f}% of total",
        f"Risk: {a.risk_level} (score {a.risk_score}/100)",
    ]
    if a.total_gain is not None:
        lines.append(
            f"Gain/loss: ${a.total_gain:,.2f} "
            f"({a.total_gain_pct:+.1f}%)"
        )
    if a.notes:
        lines.append("Notes: " + " ".join(a.notes))
    return "\n".join(lines)


class PortfolioAgent:
    """Analyzes the user's holdings from their stored profile."""

    name = "portfolio"

    def __init__(self, llm: LLMProvider, market: MarketData) -> None:
        self.llm = llm
        self.market = market

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        trace = state.get("agent_trace", []) + ["portfolio"]
        raw = state.get("profile", {}).get("holdings", [])
        if not raw:
            return {
                "response": (
                    "I don't see any holdings yet. Add them in the "
                    "Portfolio tab (symbol, shares or value, asset "
                    "class) and I'll analyze allocation, "
                    "diversification, and risk."
                ),
                "agent_trace": trace,
                "sources": [],
            }
        holdings = [Holding.model_validate(h) for h in raw]
        analysis = analyze_portfolio(holdings, market=self.market)
        system = (
            f"{_SYSTEM}\n\nComputed analysis:\n{_summary(analysis)}"
        )
        query = state.get("query") or "Analyze my portfolio."
        answer = self.llm.reason(system, history_messages(state, query))
        return {
            "response": answer,
            "analysis": analysis.model_dump(),
            "sources": ["Your portfolio (computed locally)"],
            "agent_trace": trace,
        }
