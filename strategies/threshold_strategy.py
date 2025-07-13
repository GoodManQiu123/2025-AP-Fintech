"""A minimal rule-based demo strategy."""
from __future__ import annotations

from core.strategy_base import Strategy
from core.types import MarketData, Signal


class ThresholdStrategy(Strategy):
    """Buy low (< buy_thr) and sell high (> sell_thr)."""

    def __init__(self, buy_thr: float, sell_thr: float) -> None:
        self._buy_thr = buy_thr
        self._sell_thr = sell_thr
        self._holding = False

    def generate_signal(self, tick: MarketData) -> Signal:
        price = tick.price
        if not self._holding and price < self._buy_thr:
            self._holding = True
            return Signal.BUY
        if self._holding and price > self._sell_thr:
            self._holding = False
            return Signal.SELL
        return Signal.HOLD


def build() -> Strategy:  # factory for dynamic import
    return ThresholdStrategy(buy_thr=99.8, sell_thr=100.8)
