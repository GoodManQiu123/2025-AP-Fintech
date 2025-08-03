"""Abstract strategy interface."""
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

    # Extension point for feedback (not used yet)
    def update(self, **kwargs) -> None:  # noqa: D401
        """Optional: update internal params from performance feedback."""
        return
