"""Abstract strategy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import MarketData, Signal


class Strategy(ABC):
    """Base class for all trading strategies."""

    @abstractmethod
    def generate_signal(self, tick: MarketData) -> Signal:
        """Return BUY/SELL/HOLD for the given market tick."""
        raise NotImplementedError

    # Extension point for future learning / feedback
    def update(self, **kwargs) -> None:  # noqa: D401  (simple verb)
        """Optional: update internal params from performance feedback."""
        return
