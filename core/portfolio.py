"""Enhanced long‑only portfolio with rich performance metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from core.types import MarketData, Signal


@dataclass(slots=True)
class Trade:
    """Completed round‑trip trade record."""
    entry_price: float
    exit_price: float
    profit: float


class Portfolio:
    """Tracks cash, position, and computes detailed performance stats."""

    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self._start_cash: float = starting_cash
        self._cash: float = starting_cash
        self._units: int = 0
        self._entry_price: float = 0.0
        self.trades: List[Trade] = []
        self._equity_curve: List[float] = [starting_cash]  # record after each trade

    # ------------------------------------------------------------------ core
    def execute(self, signal: Signal, bar: MarketData) -> None:
        """Execute trade according to signal (long‑only, 1 unit)."""
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
            profit = price - self._entry_price
            self.trades.append(Trade(self._entry_price, price, profit))
            self._entry_price = 0.0
            self._equity_curve.append(self._cash)  # equity changes only when flat

    # ---------------------------------------------------------------- metrics
    @property
    def realised_pnl(self) -> float:
        return sum(t.profit for t in self.trades)

    @property
    def win_trades(self) -> int:
        return sum(1 for t in self.trades if t.profit > 0)

    @property
    def loss_trades(self) -> int:
        return sum(1 for t in self.trades if t.profit < 0)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown based on equity curve (positive number)."""
        peak = self._equity_curve[0]
        max_dd = 0.0
        for equity in self._equity_curve:
            peak = max(peak, equity)
            drawdown = peak - equity
            max_dd = max(max_dd, drawdown)
        return max_dd

    # ---------------------------------------------------------------- report
    def summary(self) -> str:
        total_trades = len(self.trades)
        win_rate = (self.win_trades / total_trades * 100) if total_trades else 0
        avg_pnl = self.realised_pnl / total_trades if total_trades else 0
        roi_pct = (self.realised_pnl / self._start_cash * 100) if self._start_cash else 0

        max_profit = max((t.profit for t in self.trades), default=0.0)
        max_loss = min((t.profit for t in self.trades), default=0.0)

        lines = [
            "========== Portfolio Summary ==========",
            f"Start cash        : {self._start_cash:,.2f}",
            f"End cash          : {self._cash:,.2f}",
            f"Open units        : {self._units}",
            f"Realised PnL      : {self.realised_pnl:,.2f}",
            f"ROI %             : {roi_pct:,.2f} %",
            "",
            f"Total trades      : {total_trades}",
            f"  Wins / Losses   : {self.win_trades} / {self.loss_trades}",
            f"  Win rate        : {win_rate:,.2f} %",
            f"  Avg PnL per trade: {avg_pnl:,.2f}",
            f"  Max profit      : {max_profit:,.2f}",
            f"  Max loss        : {max_loss:,.2f}",
            "",
            f"Max drawdown      : {self.max_drawdown:,.2f}",
            "========================================",
        ]
        return "\n".join(lines)
