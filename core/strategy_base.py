# strategy_base.py
"""Abstract strategy interface.

Defines the base ``Strategy`` contract that all trading strategies must follow.
Implementations must provide ``generate_signal`` and may optionally override
``observe`` (for warm-up) and ``update`` (for adaptive behaviors).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import MarketData, Signal


class Strategy(ABC):
    """Base class for all trading strategies."""

    # ---------------- mandatory -------------------------------------------
    @abstractmethod
    def generate_signal(self, bar: MarketData) -> Signal:
        """Return BUY / SELL / HOLD for the given market bar."""
        raise NotImplementedError

    # ---------------- optional --------------------------------------------
    def observe(self, bar: MarketData) -> None:  # noqa: D401
        """Warm-up hook: collect data without returning a signal."""
        return

    def update(self, **kwargs) -> None:  # noqa: D401
        """Optional: update internal params from performance feedback."""
        return
