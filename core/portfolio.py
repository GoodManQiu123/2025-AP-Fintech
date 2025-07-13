"""Minimal long-only portfolio simulator."""
from __future__ import annotations

from dataclasses import dataclass

from core.types import MarketData, Signal


@dataclass
class Trade:
    """Record of a completed round-trip trade."""
    entry_price: float
    exit_price: float
    profit: float


class Portfolio:
    """Tracks cash, position, and realised PnL."""

    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self.cash = starting_cash
        self.units = 0.0
        self.entry_price = 0.0
        self.trades: list[Trade] = []

    def execute(self, signal: Signal, tick: MarketData) -> None:
        """Act on strategy signal."""
        if signal == Signal.BUY and self.units == 0:
            self.units = 1.0
            self.entry_price = tick.price
            self.cash -= tick.price
        elif signal == Signal.SELL and self.units == 1:
            profit = tick.price - self.entry_price
            self.cash += tick.price
            self.trades.append(
                Trade(self.entry_price, tick.price, profit)
            )
            self.units = 0.0
            self.entry_price = 0.0
        # HOLD or mismatched signal → no action

    # --- Reporting helpers ----------------------------------------------------

    @property
    def realised_pnl(self) -> float:
        return sum(t.profit for t in self.trades)

    def summary(self) -> str:
        return (
            f"Cash: {self.cash:.2f}, "
            f"Open units: {self.units}, "
            f"Realised PnL: {self.realised_pnl:.2f}"
        )
