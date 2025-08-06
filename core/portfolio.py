"""Portfolio with multi-unit support, analytics, and trade plotting."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from core.types import MarketData, Signal


@dataclass(slots=True)
class Trade:
    """Round-trip trade record (may cover >1 unit)."""
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_price: float
    exit_price: float
    units: int
    profit: float


class Portfolio:
    """Long-only portfolio supporting variable unit size."""

    # ───────────────────────────── init / execute ──────────────────────────
    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self._start_cash = starting_cash
        self._cash = starting_cash
        self._units = 0
        self._entry_price = 0.0
        self._entry_time: Optional[dt.datetime] = None
        self.trades: List[Trade] = []
        self._equity_curve: List[float] = [starting_cash]

    def execute(self, signal: Signal, bar: MarketData, units: int = 1) -> None:
        """Execute BUY / SELL up to `units`; auto-adjust for cash / inventory."""
        price = bar.price
        if price is None or units <= 0:
            return

        ts = dt.datetime.fromisoformat(bar.time)

        if signal is Signal.BUY:
            max_affordable = int(self._cash // price)
            qty = min(units, max_affordable)
            if qty == 0:
                return
            if self._units == 0:
                self._entry_price = price
                self._entry_time = ts
            else:  # 加仓，更新平均持仓成本
                self._entry_price = (
                    self._entry_price * self._units + price * qty
                ) / (self._units + qty)
            self._cash -= price * qty
            self._units += qty

        elif signal is Signal.SELL:
            qty = min(units, self._units)
            if qty == 0:
                return
            self._cash += price * qty
            self._units -= qty
            if self._units == 0 and self._entry_time:
                profit = (price - self._entry_price) * qty
                self.trades.append(
                    Trade(
                        entry_time=self._entry_time,
                        exit_time=ts,
                        entry_price=self._entry_price,
                        exit_price=price,
                        units=qty,
                        profit=profit,
                    )
                )
                self._entry_time = None
                self._entry_price = 0.0
                self._equity_curve.append(self._cash)

    # ───────────────────────────── analytics ───────────────────────────────
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

    # ───────────────────────────── reporting ───────────────────────────────
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
            f"Open units       : {self._units}",
            f"Realised PnL     : {self.realised_pnl:,.2f}",
            f"ROI %            : {roi_pct:,.2f} %",
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

    def trade_logs(self) -> str:
        """Return formatted trade log."""
        header = "\n----- Trade Log -----"
        if not self.trades:
            return f"{header}\nNo trades executed."
        rows = [
            f"{t.entry_time.date()} BUY {t.units}@{t.entry_price:.2f} → "
            f"{t.exit_time.date()} SELL @{t.exit_price:.2f}  PnL {t.profit:.2f}"
            for t in self.trades
        ]
        return f"{header}\n" + "\n".join(rows)

    # ───────────────────────────── plotting ────────────────────────────────
    def plot_trades(
        self,
        price_csv: Path,
        entry_dt: Optional[dt.datetime] = None,
        *,
        title: str = "Trade Overlay",
    ) -> None:
        """Draw close-price line plus buy/sell markers."""
        df = pd.read_csv(price_csv, parse_dates=["time"]).set_index("time")
        if entry_dt:
            df = df[df.index >= entry_dt]

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.figure(figsize=(13, 6))
        plt.plot(df["close"], label="Close", linewidth=1.4, color="#1f77b4")

        buys_x, buys_y, sells_x, sells_y = [], [], [], []
        for t in self.trades:
            buys_x.append(t.entry_time)
            buys_y.append(t.entry_price)
            sells_x.append(t.exit_time)
            sells_y.append(t.exit_price)

        plt.scatter(buys_x, buys_y, marker="^", s=80, color="#2ca02c", label="Buy")
        plt.scatter(sells_x, sells_y, marker="v", s=80, color="#d62728", label="Sell")

        plt.title(title, fontsize=14, pad=10)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()
