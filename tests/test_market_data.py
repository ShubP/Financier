"""Market-data tests (mock source, offline)."""

from src.data.market_data import MarketData, Quote


def test_mock_quote_deterministic() -> None:
    md = MarketData()
    q1 = md._mock_quote("AAPL")
    q2 = md._mock_quote("AAPL")
    assert isinstance(q1, Quote)
    assert q1.price == q2.price
    assert q1.price > 0
    assert q1.source == "mock"


def test_mock_history_length() -> None:
    md = MarketData()
    hist = md._mock_history("AAPL", "6mo")
    assert len(hist) == 126
    assert hist[0]["close"] > 0


def test_cache_roundtrip() -> None:
    md = MarketData()
    md._store("k", "v")
    assert md._cached("k") == "v"
