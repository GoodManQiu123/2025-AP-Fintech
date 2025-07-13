"""Adaptive threshold strategy: buy near local low, sell near local high."""
from __future__ import annotations

from collections import deque
from typing import Deque

from core.strategy_base import Strategy
from core.types import MarketData, Signal


class AdaptiveThresholdStrategy(Strategy):
    """Buy when price dips x% below recent max; sell when y% above recent min."""

    def __init__(
        self,
        lookback: int = 20,
        buy_pct: float = 0.02,
        sell_pct: float = 0.02,
    ) -> None:
        self._lookback = lookback
        self._buy_pct = buy_pct
        self._sell_pct = sell_pct
        self._prices: Deque[float] = deque(maxlen=lookback)
        self._holding = False
        self._entry_price: float | None = None

    # ------------------------------------------------------------------ API --
    def generate_signal(self, bar: MarketData) -> Signal:
        price = bar.price
        if price is None:
            return Signal.HOLD

        self._prices.append(price)

        if len(self._prices) < self._lookback:
            return Signal.HOLD  # not enough data yet

        recent_max = max(self._prices)
        recent_min = min(self._prices)

        if not self._holding and price < recent_min * (1 + self._buy_pct):
            self._holding = True
            self._entry_price = price
            return Signal.BUY

        if self._holding and price > recent_max * (1 - self._sell_pct):
            self._holding = False
            self._entry_price = None
            return Signal.SELL

        return Signal.HOLD


def build() -> Strategy:
    """Factory used by engine dynamic import."""
    # 30-bar look-back, buy 2 % below recent min, sell 2 % below recent max
    return AdaptiveThresholdStrategy(lookback=30, buy_pct=0.02, sell_pct=0.02)
