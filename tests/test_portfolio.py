"""Portfolio analytics tests."""

from src.finance.portfolio import Holding, analyze_portfolio


def test_basic_allocation_and_risk() -> None:
    holdings = [
        Holding(symbol="VTI", value=6000, asset_class="equity"),
        Holding(symbol="BND", value=4000, asset_class="bond"),
    ]
    a = analyze_portfolio(holdings)
    assert a.total_value == 10000
    assert a.allocation_by_class["equity"] == 60.0
    assert a.allocation_by_class["bond"] == 40.0
    # 0.6 * 80 + 0.4 * 30 = 60
    assert abs(a.risk_score - 60.0) < 0.5
    assert a.risk_level == "Moderate"
    assert a.diversification_score > 0


def test_concentration_note() -> None:
    holdings = [
        Holding(symbol="TSLA", value=9000, asset_class="equity"),
        Holding(symbol="BND", value=1000, asset_class="bond"),
    ]
    a = analyze_portfolio(holdings)
    assert a.concentration_top_pct >= 90
    assert any("concentrated" in n.lower() for n in a.notes)


def test_gain_computation() -> None:
    holdings = [
        Holding(
            symbol="X",
            value=1200,
            cost_basis=1000,
            asset_class="equity",
        )
    ]
    a = analyze_portfolio(holdings)
    assert a.total_gain == 200
    assert a.total_gain_pct == 20.0
