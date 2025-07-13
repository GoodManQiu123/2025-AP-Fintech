"""Simple long-only portfolio simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.types import MarketData, Signal


@dataclass(slots=True)
class Trade:
    """Round-trip trade record."""
    entry_price: float
    exit_price: float
    profit: float


class Portfolio:
    """Tracks cash, position, and realised PnL (long-only, 1 unit)."""

    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self._cash: float = starting_cash
        self._units: int = 0
        self._entry_price: float = 0.0
        self.trades: List[Trade] = []

    # --------------------------------------------------------------------- api
    def execute(self, signal: Signal, bar: MarketData) -> None:
        """Update portfolio state in response to a strategy signal."""
        price = bar.price
        if price is None:
            return

        if signal is Signal.BUY and self._units == 0:
            self._units = 1
            self._entry_price = price
            self._cash -= price
        elif signal is Signal.SELL and self._units == 1:
            self._units = 0
            self._cash += price
            self.trades.append(
                Trade(
                    entry_price=self._entry_price,
                    exit_price=price,
                    profit=price - self._entry_price,
                )
            )
            self._entry_price = 0.0

    # ------------------------------------------------------------------ report
    @property
    def realised_pnl(self) -> float:
        return sum(t.profit for t in self.trades)

    def summary(self) -> str:
        return (
            f"Cash: {self._cash:.2f} | "
            f"Open units: {self._units} | "
            f"Realised PnL: {self.realised_pnl:.2f}"
        )
