"""Market analysis agent: live quotes and recent trends."""

from __future__ import annotations

import re
from typing import Any

from ..data.market_data import MarketData
from ..llm.base import LLMProvider
from .base_agent import history_messages

_NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "s&p 500": "^GSPC",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "bitcoin": "BTC-USD",
}

# Uppercase tokens that are not tickers.
_STOPWORDS = {
    "A", "I", "THE", "AND", "OR", "MY", "IS", "IT", "ETF", "CEO",
    "IPO", "USD", "DCA", "IRA", "API", "AI", "US", "OK", "FAQ",
    "ROI", "CD", "ESG", "P", "E",
}


def extract_symbols(text: str) -> list[str]:
    """Pull likely ticker symbols out of free text."""
    out: list[str] = []
    for m in re.findall(r"\$([A-Za-z.\-]{1,6})", text):
        out.append(m.upper())
    low = text.lower()
    for name, ticker in _NAME_TO_TICKER.items():
        if name in low:
            out.append(ticker)
    for token in re.findall(r"\b[A-Z]{2,5}\b", text):
        if token not in _STOPWORDS:
            out.append(token)
    seen: set[str] = set()
    unique: list[str] = []
    for sym in out:
        if sym not in seen:
            seen.add(sym)
            unique.append(sym)
    return unique[:5]


_SYSTEM = (
    "You are Financier. Summarize the provided live market data for the "
    "user in plain language: the current price, today's move, and the "
    "recent trend if relevant. Be brief and factual. Do NOT predict "
    "future prices or recommend buying or selling."
)


class MarketAgent:
    """Fetches quotes/history and explains them."""

    name = "market"

    def __init__(self, llm: LLMProvider, market: MarketData) -> None:
        self.llm = llm
        self.market = market

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state.get("query", "")
        symbols = extract_symbols(query)
        if not symbols:
            holdings = state.get("profile", {}).get("holdings", [])
            symbols = [
                h.get("symbol") for h in holdings if h.get("symbol")
            ][:3]
        trace = state.get("agent_trace", []) + ["market"]
        if not symbols:
            return {
                "response": (
                    "Tell me a ticker symbol (e.g., AAPL or $MSFT) "
                    "and I'll pull the latest price and recent trend."
                ),
                "agent_trace": trace,
                "sources": [],
            }

        quotes = [self.market.get_quote(s).model_dump() for s in symbols]
        history = {symbols[0]: self.market.get_history(symbols[0])}
        lines = "\n".join(
            f"{q['symbol']}: {q['price']} {q['currency']} "
            f"({q['change_pct']:+.2f}% today; source {q['source']})"
            for q in quotes
        )
        system = f"{_SYSTEM}\n\nLive data:\n{lines}"
        answer = self.llm.reason(system, history_messages(state, query))
        srcs = sorted({q["source"] for q in quotes})
        return {
            "response": answer,
            "market": {"quotes": quotes, "history": history},
            "sources": [f"Live market data ({', '.join(srcs)})"],
            "agent_trace": trace,
        }
