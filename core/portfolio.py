"""Portfolio with analytics, plotting, and log export."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from core.types import MarketData, Signal


# ─────────────────────────────── data model ────────────────────────────────
@dataclass(slots=True)
class Trade:
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_price: float
    exit_price: float
    units: int
    profit: float


# ───────────────────────────── portfolio class ─────────────────────────────
class Portfolio:
    """Long-only portfolio supporting variable unit size."""

    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self._start_cash = starting_cash
        self._cash = starting_cash
        self._units = 0
        self._entry_price = 0.0
        self._entry_time: Optional[dt.datetime] = None
        self.trades: List[Trade] = []
        self._equity_curve: List[float] = [starting_cash]

    # --------------------------- trade execution ---------------------------
    def execute(self, signal: Signal, bar: MarketData, units: int = 1) -> None:
        """Execute BUY / SELL up to `units`, auto-adjusting for cash / inventory."""
        price = bar.price
        if price is None or units <= 0:
            return
        ts = dt.datetime.fromisoformat(bar.time)

        if signal is Signal.BUY:
            qty = min(units, int(self._cash // price))
            if qty == 0:
                return
            if self._units == 0:
                self._entry_price, self._entry_time = price, ts
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
                self._close_position(exit_price=price, exit_time=ts, qty=qty)

    def _close_position(self, *, exit_price: float, exit_time: dt.datetime, qty: int) -> None:
        """Record closed trade and update equity curve."""
        profit = (exit_price - self._entry_price) * qty
        self.trades.append(
            Trade(
                entry_time=self._entry_time,  # type: ignore[arg-type]
                exit_time=exit_time,
                entry_price=self._entry_price,
                exit_price=exit_price,
                units=qty,
                profit=profit,
            )
        )
        self._entry_time = None
        self._entry_price = 0.0
        self._equity_curve.append(self._cash)

    # --------------------------- analytics ---------------------------------
    @property
    def realised_pnl(self) -> float:
        return sum(t.profit for t in self.trades)

    @property
    def max_drawdown(self) -> float:
        peak = dd = 0.0
        for eq in self._equity_curve:
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        return dd

    # --------------------------- text reports ------------------------------
    def summary(self) -> str:
        total = len(self.trades)
        roi = self.realised_pnl / self._start_cash * 100
        win = sum(t.profit > 0 for t in self.trades)
        loss = total - win
        win_rate = win / total * 100 if total else 0
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
            f"ROI %            : {roi:,.2f} %",
            "",
            f"Total trades     : {total}",
            f"Winning / Losing : {win} / {loss}",
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
        header = "\n----- Trade Log -----"
        if not self.trades:
            return f"{header}\nNo trades executed."
        body = "\n".join(
            f"{t.entry_time.date()} BUY {t.units}@{t.entry_price:.2f} → "
            f"{t.exit_time.date()} SELL @{t.exit_price:.2f}  PnL {t.profit:.2f}"
            for t in self.trades
        )
        return f"{header}\n{body}"

    # --------------------------- plotting & export -------------------------
    def _save_plot(
        self,
        price_csv: Path,
        dst_file: Path,
        entry_dt: Optional[dt.datetime],
        title: str,
    ) -> None:
        df = pd.read_csv(price_csv, parse_dates=["time"]).set_index("time")
        if entry_dt:
            df = df[df.index >= entry_dt]

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.figure(figsize=(13, 6))
        plt.plot(df["close"], label="Close", linewidth=1.3, color="#1f77b4")

        plt.scatter(
            [t.entry_time for t in self.trades],
            [t.entry_price for t in self.trades],
            marker="^",
            s=90,
            color="#2ca02c",
            label="Buy",
            zorder=3,
        )
        plt.scatter(
            [t.exit_time for t in self.trades],
            [t.exit_price for t in self.trades],
            marker="v",
            s=90,
            color="#d62728",
            label="Sell",
            zorder=3,
        )

        plt.title(title, fontsize=14, pad=10)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(dst_file, dpi=120)
        plt.close()

    def export_logs(
        self,
        dst_dir: Path,
        price_csv: Path,
        entry_dt: Optional[dt.datetime],
        asset: str,
    ) -> None:
        """Write summary.log, trade_log.log, and trades.png into dst_dir."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "summary.log").write_text(self.summary(), encoding="utf-8")
        (dst_dir / "trade_log.log").write_text(self.trade_logs(), encoding="utf-8")
        self._save_plot(
            price_csv,
            dst_file=dst_dir / "trades.png",
            entry_dt=entry_dt,
            title=f"{asset} Trade Overlay",
        )
