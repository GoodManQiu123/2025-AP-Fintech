"""Adaptive threshold trading strategy.

This module implements a simple, stateful strategy that aims to:
- Buy when the current price is *near* the rolling minimum (i.e., within
  ``buy_pct`` *above* the recent low).
- Sell when the current price is *near* the rolling maximum (i.e., within
  ``sell_pct`` *below* the recent high).

Design notes:
    * A fixed-size window (deque) maintains the most recent ``lookback`` prices.
    * Signals are only generated once enough data has been accumulated to fill
      the window.
    * The strategy is stateful (tracks whether it's currently holding).

Assumptions:
    * ``MarketData.price`` may be ``None``; in that case the strategy holds.
    * External engine is responsible for position sizing and order execution.

The code follows the Google Python Style Guide for docstrings and formatting.
"""
from __future__ import annotations

from collections import deque
from typing import Deque

from core.strategy_base import Strategy
from core.types import MarketData, Signal


class AdaptiveThresholdStrategy(Strategy):
    """Buy near rolling low, sell near rolling high.

    This strategy enters a long position when the current price is within
    ``buy_pct`` *above* the rolling minimum (i.e., close to the local low),
    and exits (sells) when the current price is within ``sell_pct`` *below*
    the rolling maximum (i.e., close to the local high).

    Args:
        lookback: Number of most recent bars to consider for rolling extrema.
        buy_pct: Fractional threshold (e.g., 0.02 for 2%) above the rolling
            minimum that qualifies as "near the low" for a BUY signal.
        sell_pct: Fractional threshold (e.g., 0.02 for 2%) below the rolling
            maximum that qualifies as "near the high" for a SELL signal.
    """

    def __init__(
        self,
        lookback: int = 20,
        buy_pct: float = 0.02,
        sell_pct: float = 0.02,
    ) -> None:
        self._lookback: int = lookback
        self._buy_pct: float = buy_pct
        self._sell_pct: float = sell_pct

        # Rolling window of recent prices, capped at `lookback` elements.
        self._price_window: Deque[float] = deque(maxlen=lookback)

        # Internal position state.
        self._in_position: bool = False
        self._entry_price: float | None = None  # Kept for potential analytics.

    # ------------------------------------------------------------------ API --
    def generate_signal(self, bar: MarketData) -> Signal:
        """Generate the next trading signal based on the latest market data.

        This method mutates internal state by updating the rolling price window.
        A signal other than HOLD is only emitted when the rolling window is
        fully populated.

        Args:
            bar: Latest market data point. Only ``bar.price`` is used here.

        Returns:
            A member of ``Signal``:
                * Signal.BUY  — if not holding and price is near the rolling low.
                * Signal.SELL — if holding and price is near the rolling high.
                * Signal.HOLD — otherwise, or when data is insufficient/invalid.
        """
        price = bar.price
        if price is None:
            return Signal.HOLD

        # Update rolling window with the latest price.
        self._price_window.append(price)

        # Wait until the window is fully populated.
        if len(self._price_window) < self._lookback:
            return Signal.HOLD

        recent_max = max(self._price_window)
        recent_min = min(self._price_window)

        # BUY when price is within `buy_pct` above the rolling minimum.
        if not self._in_position and price < recent_min * (1 + self._buy_pct):
            self._in_position = True
            self._entry_price = price
            return Signal.BUY

        # SELL when price is within `sell_pct` below the rolling maximum.
        if self._in_position and price > recent_max * (1 - self._sell_pct):
            self._in_position = False
            self._entry_price = None
            return Signal.SELL

        return Signal.HOLD


def build() -> Strategy:
    """Factory function for dynamic import by the strategy engine.

    Returns:
        A configured ``AdaptiveThresholdStrategy`` instance using:
            * lookback = 30 bars
            * buy_pct  = 0.02  (BUY when price is within +2% of rolling min)
            * sell_pct = 0.02  (SELL when price is within -2% of rolling max)
    """
    return AdaptiveThresholdStrategy(lookback=30, buy_pct=0.02, sell_pct=0.02)
