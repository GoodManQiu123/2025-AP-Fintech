"""Enhanced portfolio with detailed trade analytics."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List

from core.types import MarketData, Signal


@dataclass(slots=True)
class Trade:
    """Round‑trip trade record."""
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_price: float
    exit_price: float
    units: int
    profit: float


class Portfolio:
    """Long‑only, position‑size = `units` (default 1)."""

    def __init__(self, starting_cash: float = 10_000.0, units: int = 1) -> None:
        self._start_cash = starting_cash
        self._cash = starting_cash
        self._units = 0
        self._entry_price = 0.0
        self._entry_time: dt.datetime | None = None
        self._trade_units = units
        self.trades: List[Trade] = []
        self._equity_curve: List[float] = [starting_cash]

    # ───────────────────────────────────────────────────────── trade execution
    def execute(self, signal: Signal, bar: MarketData) -> None:
        """Execute a BUY or SELL signal based on current position state."""
        price = bar.price
        if price is None:
            return

        ts = dt.datetime.fromisoformat(bar.time)

        if signal is Signal.BUY and self._units == 0:
            self._units = self._trade_units
            self._entry_price = price
            self._entry_time = ts
            self._cash -= price * self._trade_units

        elif signal is Signal.SELL and self._units > 0:
            self._units = 0
            profit = (price - self._entry_price) * self._trade_units
            self._cash += price * self._trade_units
            self.trades.append(
                Trade(
                    entry_time=self._entry_time,  # type: ignore[arg-type]
                    exit_time=ts,
                    entry_price=self._entry_price,
                    exit_price=price,
                    units=self._trade_units,
                    profit=profit,
                )
            )
            self._entry_price = 0.0
            self._entry_time = None
            self._equity_curve.append(self._cash)

    # ───────────────────────────────────────────────────────── performance
    @property
    def realised_pnl(self) -> float:
        return sum(t.profit for t in self.trades)

    @property
    def win_trades(self) -> int:
        return sum(t.profit > 0 for t in self.trades)

    @property
    def loss_trades(self) -> int:
        return sum(t.profit < 0 for t in self.trades)

    @property
    def max_drawdown(self) -> float:
        peak = self._equity_curve[0]
        mdd = 0.0
        for equity in self._equity_curve:
            peak = max(peak, equity)
            mdd = max(mdd, peak - equity)
        return mdd

    # ───────────────────────────────────────────────────────── report
    def summary(self) -> str:
        total = len(self.trades)
        roi_pct = self.realised_pnl / self._start_cash * 100
        win_rate = self.win_trades / total * 100 if total else 0
        gross_profit = sum(max(t.profit, 0) for t in self.trades)
        gross_loss = sum(min(t.profit, 0) for t in self.trades)
        max_gain = max((t.profit for t in self.trades), default=0.0)
        max_loss = min((t.profit for t in self.trades), default=0.0)

        lines = [
            "========== Portfolio Summary ==========",
            f"Start cash       : {self._start_cash:,.2f}",
            f"End cash         : {self._cash:,.2f}",
            f"Realised PnL     : {self.realised_pnl:,.2f}",
            f"ROI %            : {roi_pct:,.2f} %",
            f"Open units       : {self._units}",
            "",
            f"Total trades     : {total}",
            f"Winning / Losing : {self.win_trades} / {self.loss_trades}",
            f"Win rate         : {win_rate:,.2f} %",
            f"Gross profit     : {gross_profit:,.2f}",
            f"Gross loss       : {gross_loss:,.2f}",
            f"Max single gain  : {max_gain:,.2f}",
            f"Max single loss  : {max_loss:,.2f}",
            "",
            f"Max drawdown     : {self.max_drawdown:,.2f}",
            "========================================",
        ]
        return "\n".join(lines)
