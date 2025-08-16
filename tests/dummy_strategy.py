"""A minimal stateful strategy used only for test/integration.

Behavior:
- Buy 1 unit on the first bar that has a price.
- Sell all on a later bar if the price is strictly above the entry.
- Otherwise HOLD.
"""
from __future__ import annotations

from core.strategy_base import Strategy  # type: ignore
from core.types import MarketData, Signal  # type: ignore


class _Mini(Strategy):
    def __init__(self) -> None:
        self._in = False
        self._entry = None
        self.last_units = 0

    def generate_signal(self, bar: MarketData) -> Signal:
        if bar.price is None:
            return Signal.HOLD
        if not self._in:
            self._in = True
            self._entry = bar.price
            self.last_units = 1
            return Signal.BUY
        if bar.price > (self._entry or 0.0):
            self._in = False
            self.last_units = 1
            return Signal.SELL
        self.last_units = 0
        return Signal.HOLD


def build() -> Strategy:
    return _Mini()
