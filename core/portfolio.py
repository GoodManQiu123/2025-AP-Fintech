"""Portfolio with FIFO lot accounting, analytics, plotting, and log export."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from core.types import MarketData, Signal


# ─────────────────────────────── data models ────────────────────────────────
@dataclass(slots=True)
class PositionLot:
    """One FIFO lot created by a BUY transaction."""
    time: dt.datetime
    price: float
    qty: int


@dataclass(slots=True)
class Trade:
    """A realised trade generated when lots are consumed by SELLs."""
    entry_time: dt.datetime
    exit_time: dt.datetime
    entry_price: float
    exit_price: float
    units: int
    profit: float


# ───────────────────────────── portfolio class ─────────────────────────────
class Portfolio:
    """Long-only portfolio with FIFO lot-based accounting.

    Supports:
      • Multiple incremental BUYs (adds lots)
      • Multiple incremental SELLs (consumes lots FIFO and creates multiple Trade records)
      • Accurate realised PnL for partial closes
    """

    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self._start_cash = starting_cash
        self._cash = starting_cash
        self._lots: List[PositionLot] = []
        self.trades: List[Trade] = []
        # Equity curve of realised cash only (simple but robust)
        self._equity_curve: List[float] = [starting_cash]

    # --------------------------- helper properties ---------------------------
    @property
    def open_units(self) -> int:
        return sum(lot.qty for lot in self._lots)

    @property
    def cash(self) -> float:
        return self._cash

    # --------------------------- trade execution -----------------------------
    def execute(self, signal: Signal, bar: MarketData, units: int = 1) -> None:
        """Execute BUY / SELL up to `units`, with automatic cash/position constraints."""
        price = bar.price
        if price is None or units <= 0:
            return
        ts = dt.datetime.fromisoformat(bar.time)

        if signal is Signal.BUY:
            affordable = int(self._cash // price)
            qty = min(units, max(affordable, 0))
            if qty == 0:
                return
            self._cash -= price * qty
            self._lots.append(PositionLot(time=ts, price=price, qty=qty))
            # realised equity after trade (cash only)
            self._equity_curve.append(self._cash)

        elif signal is Signal.SELL:
            if self.open_units <= 0:
                return
            qty_to_sell = min(units, self.open_units)
            if qty_to_sell <= 0:
                return

            remaining = qty_to_sell
            while remaining > 0 and self._lots:
                lot = self._lots[0]
                close_qty = min(remaining, lot.qty)
                self._cash += price * close_qty
                profit = (price - lot.price) * close_qty
                self.trades.append(
                    Trade(
                        entry_time=lot.time,
                        exit_time=ts,
                        entry_price=lot.price,
                        exit_price=price,
                        units=close_qty,
                        profit=profit,
                    )
                )
                lot.qty -= close_qty
                remaining -= close_qty
                if lot.qty == 0:
                    self._lots.pop(0)

            self._equity_curve.append(self._cash)

    # --------------------------- analytics -----------------------------------
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

    # --------------------------- text reports --------------------------------
    def summary(self) -> str:
        total = len(self.trades)
        roi = (self.realised_pnl / self._start_cash * 100.0) if self._start_cash else 0.0
        win = sum(t.profit > 0 for t in self.trades)
        loss = total - win
        win_rate = (win / total * 100.0) if total else 0.0
        gross_profit = sum(max(t.profit, 0.0) for t in self.trades)
        gross_loss = sum(min(t.profit, 0.0) for t in self.trades)
        max_gain = max((t.profit for t in self.trades), default=0.0)
        max_loss = min((t.profit for t in self.trades), default=0.0)

        lines = [
            "========== Portfolio Summary ==========",
            f"Start cash       : {self._start_cash:,.2f}",
            f"End cash         : {self._cash:,.2f}",
            f"Open units       : {self.open_units}",
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

    # --------------------------- plotting & export ---------------------------
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

        # Mark trade points
        buy_times = [t.entry_time for t in self.trades]
        buy_prices = [t.entry_price for t in self.trades]
        sell_times = [t.exit_time for t in self.trades]
        sell_prices = [t.exit_price for t in self.trades]

        if buy_times:
            plt.scatter(buy_times, buy_prices, marker="^", s=90, color="#2ca02c", label="Buy", zorder=3)
        if sell_times:
            plt.scatter(sell_times, sell_prices, marker="v", s=90, color="#d62728", label="Sell", zorder=3)

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
        """Write summary.log, trade_log.log (includes summary), and trades.png."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        # 合并 summary 到 trade_log
        combined = f"{self.summary()}\n{self.trade_logs()}"
        (dst_dir / "trade.log").write_text(combined, encoding="utf-8")
        self._save_plot(
            price_csv,
            dst_file=dst_dir / "trades.png",
            entry_dt=entry_dt,
            title=f"{asset} Trade Overlay",
        )
