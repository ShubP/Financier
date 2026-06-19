"""Real-time market data with graceful fallbacks.

Source order: yfinance (no key) -> Alpha Vantage (optional key) ->
deterministic mock data. Quotes and history are cached in-memory with
a configurable TTL to respect rate limits. The mock source keeps the
whole app functional offline and in tests.
"""

from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from ..core.config import Config, env, load_config
from ..core.logging_utils import get_logger

logger = get_logger(__name__)


class Quote(BaseModel):
    """A point-in-time price quote for one symbol."""

    symbol: str
    name: str
    price: float
    previous_close: float
    change: float
    change_pct: float
    currency: str = "USD"
    source: str = "mock"  # yfinance | alpha_vantage | mock
    as_of: str = ""


class MarketData:
    """Fetch quotes/history with caching and a source fallback chain."""

    def __init__(self, cfg: Config | None = None) -> None:
        cfg = cfg or load_config()
        self._primary = str(cfg.get("market_data.primary", "yfinance"))
        ttl_min = int(cfg.get("market_data.cache_ttl_minutes", 30))
        self._ttl = ttl_min * 60
        self._av_fallback = bool(
            cfg.get("market_data.alpha_vantage_fallback", True)
        )
        self._cache: dict[str, tuple[float, Any]] = {}

    # --- public API ---
    def get_quote(self, symbol: str) -> Quote:
        symbol = symbol.upper().strip()
        key = f"quote:{symbol}"
        hit = self._cached(key)
        if hit is not None:
            return hit

        quote: Quote | None = None
        if self._primary == "yfinance":
            quote = self._yf_quote(symbol)
            if quote is None and self._av_fallback:
                quote = self._av_quote(symbol)
        elif self._primary == "alpha_vantage":
            quote = self._av_quote(symbol)
        if quote is None:
            quote = self._mock_quote(symbol)

        self._store(key, quote)
        return quote

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s.upper().strip(): self.get_quote(s) for s in symbols}

    def get_history(
        self,
        symbol: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> list[dict[str, Any]]:
        symbol = symbol.upper().strip()
        key = f"hist:{symbol}:{period}:{interval}"
        hit = self._cached(key)
        if hit is not None:
            return hit

        hist: list[dict[str, Any]] | None = None
        if self._primary == "yfinance":
            hist = self._yf_history(symbol, period, interval)
        if not hist:
            hist = self._mock_history(symbol, period)

        self._store(key, hist)
        return hist

    # --- cache ---
    def _cached(self, key: str) -> Any:
        item = self._cache.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self._ttl:
            self._cache.pop(key, None)
            return None
        return val

    def _store(self, key: str, val: Any) -> None:
        self._cache[key] = (time.time(), val)

    # --- yfinance source ---
    def _yf_quote(self, symbol: str) -> Quote | None:
        try:
            import yfinance as yf
        except Exception:
            return None
        try:
            info = yf.Ticker(symbol).fast_info
            price = float(info.last_price)
            prev = float(info.previous_close)
            currency = str(getattr(info, "currency", "USD") or "USD")
        except Exception as exc:
            logger.warning("yfinance quote %s failed: %s", symbol, exc)
            return None
        if not price or math.isnan(price):
            return None
        return _build_quote(symbol, price, prev, currency, "yfinance")

    def _yf_history(
        self, symbol: str, period: str, interval: str
    ) -> list[dict[str, Any]] | None:
        try:
            import yfinance as yf
        except Exception:
            return None
        try:
            df = yf.Ticker(symbol).history(
                period=period, interval=interval
            )
        except Exception as exc:
            logger.warning("yfinance hist %s failed: %s", symbol, exc)
            return None
        if df is None or df.empty:
            return None
        out: list[dict[str, Any]] = []
        for idx, row in df.iterrows():
            out.append(
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 2),
                }
            )
        return out

    # --- Alpha Vantage source ---
    def _av_quote(self, symbol: str) -> Quote | None:
        key = env("ALPHA_VANTAGE_API_KEY")
        if not key:
            return None
        import requests

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": key,
        }
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=10,
            )
            data = resp.json().get("Global Quote", {})
            price = float(data["05. price"])
            prev = float(data["08. previous close"])
        except Exception as exc:
            logger.warning("alpha vantage %s failed: %s", symbol, exc)
            return None
        return _build_quote(symbol, price, prev, "USD", "alpha_vantage")

    # --- deterministic mock source ---
    def _mock_quote(self, symbol: str) -> Quote:
        base = _seed_price(symbol)
        drift = ((_seed_int(symbol + "drift") % 400) - 200) / 100.0
        prev = round(base, 2)
        price = round(base * (1 + drift / 100.0), 2)
        return _build_quote(symbol, price, prev, "USD", "mock")

    def _mock_history(
        self, symbol: str, period: str
    ) -> list[dict[str, Any]]:
        n = _period_to_days(period)
        seed = _seed_int(symbol)
        price = _seed_price(symbol)
        today = datetime.now(timezone.utc).date()
        out: list[dict[str, Any]] = []
        for k in range(n):
            day = today - timedelta(days=(n - 1 - k))
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            step = ((seed % 201) - 100) / 5000.0  # ~ +/- 2%
            price = max(1.0, price * (1 + step))
            out.append({"date": day.isoformat(), "close": round(price, 2)})
        return out


# --- module-level helpers ---
def _build_quote(
    symbol: str, price: float, prev: float, currency: str, source: str
) -> Quote:
    change = price - prev
    pct = (change / prev * 100.0) if prev else 0.0
    return Quote(
        symbol=symbol,
        name=symbol,
        price=round(price, 2),
        previous_close=round(prev, 2),
        change=round(change, 2),
        change_pct=round(pct, 2),
        currency=currency,
        source=source,
        as_of=_now_iso(),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _seed_int(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _seed_price(symbol: str) -> float:
    # Stable pseudo price in roughly [20, 500).
    return 20.0 + (_seed_int(symbol) % 48000) / 100.0


def _period_to_days(period: str) -> int:
    return {
        "1mo": 21,
        "3mo": 63,
        "6mo": 126,
        "1y": 252,
        "2y": 504,
    }.get(period, 126)


_INSTANCE: MarketData | None = None


def get_market_data(cfg: Config | None = None) -> MarketData:
    """Return a process-wide :class:`MarketData` singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MarketData(cfg)
    return _INSTANCE
