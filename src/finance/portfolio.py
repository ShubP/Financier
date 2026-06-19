"""Portfolio analytics: allocation, diversification, risk, gains.

Pure functions over a list of holdings. Prices may be supplied
directly or resolved live via the market-data layer. The Portfolio
Analysis agent turns these numbers into plain-English guidance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Relative risk weight per asset class (0..100) for a blended score.
_CLASS_RISK: dict[str, float] = {
    "equity": 80.0,
    "stock": 80.0,
    "etf": 70.0,
    "reit": 70.0,
    "crypto": 100.0,
    "commodity": 60.0,
    "bond": 30.0,
    "cash": 5.0,
    "other": 50.0,
}

# Upper bound (exclusive) -> label. Anything above the last is "Aggressive".
_RISK_BANDS: list[tuple[float, str]] = [
    (35.0, "Conservative"),
    (65.0, "Moderate"),
]


class Holding(BaseModel):
    """One position. Provide ``value`` OR (``shares`` and a price)."""

    symbol: str
    shares: float = 0.0
    price: float | None = None  # resolved live if omitted
    value: float | None = None  # overrides shares * price
    asset_class: str = "equity"
    cost_basis: float | None = None  # total amount invested (optional)


class HoldingResult(BaseModel):
    symbol: str
    asset_class: str
    value: float
    weight_pct: float
    price: float | None = None
    gain: float | None = None
    gain_pct: float | None = None


class PortfolioAnalysis(BaseModel):
    total_value: float
    num_holdings: int
    holdings: list[HoldingResult]
    allocation_by_class: dict[str, float]
    diversification_score: float  # 0..100 (higher = better spread)
    concentration_top_pct: float
    risk_score: float  # 0..100
    risk_level: str
    total_cost: float | None = None
    total_gain: float | None = None
    total_gain_pct: float | None = None
    notes: list[str] = []


def _resolve_value(
    holding: Holding, market: Any | None
) -> tuple[float, float | None]:
    """Return (market_value, unit_price) for a holding."""
    if holding.value is not None:
        return float(holding.value), holding.price
    price = holding.price
    if price is None and market is not None:
        try:
            price = market.get_quote(holding.symbol).price
        except Exception:
            price = None
    if price is not None and holding.shares:
        return float(holding.shares) * float(price), float(price)
    return 0.0, price


def _risk_level(score: float) -> str:
    for bound, label in _RISK_BANDS:
        if score < bound:
            return label
    return "Aggressive"


def analyze_portfolio(
    holdings: list[Holding], market: Any | None = None
) -> PortfolioAnalysis:
    """Compute allocation, diversification, risk, and gains."""
    resolved: list[tuple[Holding, float, float | None]] = []
    for h in holdings:
        value, price = _resolve_value(h, market)
        resolved.append((h, value, price))

    total = sum(v for _, v, _ in resolved)
    results: list[HoldingResult] = []
    class_totals: dict[str, float] = {}
    weight_fracs: list[float] = []

    for h, value, price in resolved:
        weight = (value / total) if total else 0.0
        weight_fracs.append(weight)
        ac = h.asset_class.lower().strip() or "other"
        class_totals[ac] = class_totals.get(ac, 0.0) + value

        gain = gain_pct = None
        if h.cost_basis is not None and h.cost_basis > 0:
            gain = round(value - h.cost_basis, 2)
            gain_pct = round((gain / h.cost_basis) * 100.0, 2)

        results.append(
            HoldingResult(
                symbol=h.symbol.upper(),
                asset_class=ac,
                value=round(value, 2),
                weight_pct=round(weight * 100.0, 2),
                price=round(price, 2) if price is not None else None,
                gain=gain,
                gain_pct=gain_pct,
            )
        )

    allocation = {
        ac: round((amt / total) * 100.0, 2) if total else 0.0
        for ac, amt in class_totals.items()
    }
    hhi = sum(w * w for w in weight_fracs)
    diversification = round(max(0.0, (1.0 - hhi)) * 100.0, 1)
    concentration = (
        round(max(weight_fracs) * 100.0, 1) if weight_fracs else 0.0
    )
    risk_score = round(
        sum(
            w * _CLASS_RISK.get(r.asset_class, 50.0)
            for w, r in zip(weight_fracs, results)
        ),
        1,
    )

    costs = [h.cost_basis for h in holdings if h.cost_basis is not None]
    total_cost = round(sum(costs), 2) if costs else None
    total_gain = (
        round(total - total_cost, 2) if total_cost is not None else None
    )
    total_gain_pct = (
        round((total_gain / total_cost) * 100.0, 2)
        if total_cost
        else None
    )

    return PortfolioAnalysis(
        total_value=round(total, 2),
        num_holdings=len(holdings),
        holdings=results,
        allocation_by_class=allocation,
        diversification_score=diversification,
        concentration_top_pct=concentration,
        risk_score=risk_score,
        risk_level=_risk_level(risk_score),
        total_cost=total_cost,
        total_gain=total_gain,
        total_gain_pct=total_gain_pct,
        notes=_build_notes(
            results, allocation, concentration, risk_score
        ),
    )


def _build_notes(
    results: list[HoldingResult],
    allocation: dict[str, float],
    concentration: float,
    risk_score: float,
) -> list[str]:
    notes: list[str] = []
    if results and concentration > 40:
        top = max(results, key=lambda r: r.weight_pct)
        notes.append(
            f"{top.symbol} is {concentration:.0f}% of the portfolio — "
            "that is concentrated; consider trimming or diversifying."
        )
    bonds = allocation.get("bond", 0.0) + allocation.get("cash", 0.0)
    if bonds < 10 and risk_score > 65:
        notes.append(
            "Little to no bonds/cash — the mix is equity-heavy and "
            "more volatile. A bond/cash sleeve can cushion drawdowns."
        )
    if allocation.get("cash", 0.0) > 30:
        notes.append(
            "A large cash position may lag inflation over time; "
            "consider investing some for long-term goals."
        )
    if len(results) < 3 and results:
        notes.append(
            "Holding only a few positions raises single-name risk; "
            "broad index funds spread it out cheaply."
        )
    return notes
