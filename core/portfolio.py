"""Portfolio with FIFO lots, rich analytics, mark-to-market, and export."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

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
    profit: float  # absolute profit in quote currency


# ───────────────────────────── portfolio class ─────────────────────────────
class Portfolio:
    """Long-only portfolio with FIFO lot-based accounting and rich analytics.

    Features:
      • Multiple incremental BUYs (adds lots)
      • Multiple incremental SELLs (consumes lots FIFO, each partial close = one Trade)
      • Correct realised PnL for partial closes
      • Mark-to-market on each bar via `mark_from_bar` (last price/time, equity)
    """

    # ------------------------------- init -----------------------------------
    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self._start_cash = float(starting_cash)
        self._cash = float(starting_cash)

        # Open position lots (FIFO) and realised trade records
        self._lots: List[PositionLot] = []
        self.trades: List[Trade] = []

        # Curves for risk
        self._equity_curve_cash: List[float] = [self._cash]  # cash-only (after trades)
        self._equity_curve_marked: List[float] = [self._cash]  # marked equity each bar

        # Last mark (price/time) for mark-to-market
        self._last_mark_price: Optional[float] = None
        self._last_mark_time: Optional[dt.datetime] = None

        # Exposure stats (time in market)
        self._bars_total: int = 0
        self._bars_in_market: int = 0

    # --------------------------- helper properties ---------------------------
    @property
    def open_units(self) -> int:
        return sum(lot.qty for lot in self._lots)

    @property
    def cash(self) -> float:
        return self._cash

    def _weighted_avg_cost(self) -> float:
        """Average cost for current open position (0 if flat)."""
        units = self.open_units
        if units <= 0:
            return 0.0
        total_cost = sum(l.price * l.qty for l in self._lots)
        return total_cost / units

    # --------------------------- trade execution -----------------------------
    def execute(self, signal: Signal, bar: MarketData, units: int = 1) -> None:
        """Execute BUY / SELL up to `units`, with automatic cash/position constraints.

        BUY:
          - Spend available cash, add FIFO lot(s).
        SELL:
          - Consume lots FIFO, create one Trade per consumed lot (or part-lot).

        Notes:
          - This method updates cash and lots.
          - Equity curves: cash-equity is appended after each trade event.
          - Marked-equity is updated in `mark_from_bar` (called per bar in engine).
        """
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
            # cash equity after trade
            self._equity_curve_cash.append(self._cash)

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

            # cash equity after trade
            self._equity_curve_cash.append(self._cash)

    # ----------------------------- mark-to-market ----------------------------
    def mark_from_bar(self, bar: MarketData) -> None:
        """Mark-to-market with the bar's close (or adj_close). Also track exposure."""
        price = bar.price
        if price is None:
            return
        ts = dt.datetime.fromisoformat(bar.time)

        # bars counting used for exposure
        self._bars_total += 1
        if self.open_units > 0:
            self._bars_in_market += 1

        self._last_mark_price = float(price)
        self._last_mark_time = ts

        equity = self._cash + self.open_units * self._last_mark_price
        self._equity_curve_marked.append(equity)

    # ------------------------------ analytics --------------------------------
    @staticmethod
    def _max_drawdown(series: List[float]) -> float:
        peak = dd = 0.0
        for x in series:
            peak = max(peak, x)
            dd = max(dd, peak - x)
        return dd

    @property
    def realised_pnl(self) -> float:
        return sum(t.profit for t in self.trades)

    def _trade_returns(self) -> List[float]:
        """Per-trade percentage returns; empty if no trades."""
        returns: List[float] = []
        for t in self.trades:
            if t.entry_price > 0:
                returns.append((t.exit_price - t.entry_price) / t.entry_price)
        return returns

    @staticmethod
    def _consecutive_counts(vals: List[bool]) -> Tuple[int, int]:
        """Return (max_consecutive_true, max_consecutive_false)."""
        max_t = max_f = cur_t = cur_f = 0
        for v in vals:
            if v:
                cur_t += 1
                cur_f = 0
            else:
                cur_f += 1
                cur_t = 0
            max_t = max(max_t, cur_t)
            max_f = max(max_f, cur_f)
        return max_t, max_f

    def _holding_days(self) -> Tuple[float, float]:
        """Return (avg_days, median_days) for realised trades."""
        if not self.trades:
            return 0.0, 0.0
        days = [(t.exit_time - t.entry_time).days for t in self.trades]
        avg = sum(days) / len(days)
        med = float(median(days))
        return avg, med

    # --------------------------- text reports --------------------------------
    def summary(self) -> str:
        # basics
        start_cash = self._start_cash
        end_cash = self._cash
        open_units = self.open_units
        mark_price = self._last_mark_price if self._last_mark_price is not None else 0.0
        open_value = open_units * mark_price
        avg_cost = self._weighted_avg_cost()

        unreal = (mark_price - avg_cost) * open_units if open_units > 0 else 0.0
        unreal_pct = (unreal / (avg_cost * open_units) * 100.0) if (open_units > 0 and avg_cost > 0) else 0.0

        total_equity = end_cash + open_value
        equity_roi = ((total_equity - start_cash) / start_cash * 100.0) if start_cash > 0 else 0.0

        realised = self.realised_pnl
        realised_roi = (realised / start_cash * 100.0) if start_cash > 0 else 0.0

        # trade stats
        total = len(self.trades)
        wins = sum(t.profit > 0 for t in self.trades)
        losses = sum(t.profit < 0 for t in self.trades)
        flats = total - wins - losses
        win_rate = (wins / total * 100.0) if total else 0.0

        gross_profit = sum(max(t.profit, 0.0) for t in self.trades)
        gross_loss = sum(min(t.profit, 0.0) for t in self.trades)
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf")

        avg_win = (gross_profit / wins) if wins else 0.0
        avg_loss = (gross_loss / losses) if losses else 0.0
        expectancy = (realised / total) if total else 0.0

        best = max((t.profit for t in self.trades), default=0.0)
        worst = min((t.profit for t in self.trades), default=0.0)

        max_wins, max_losses = self._consecutive_counts([t.profit > 0 for t in self.trades])
        avg_days, med_days = self._holding_days()

        rets = self._trade_returns()
        ret_mean = (sum(rets) / len(rets)) if rets else 0.0
        ret_var = (sum((r - ret_mean) ** 2 for r in rets) / len(rets)) if rets else 0.0
        ret_std = ret_var ** 0.5
        sharpe_like = (ret_mean / ret_std) if ret_std > 0 else 0.0

        # drawdowns & exposure
        dd_cash = self._max_drawdown(self._equity_curve_cash) if self._equity_curve_cash else 0.0
        dd_marked = self._max_drawdown(self._equity_curve_marked) if self._equity_curve_marked else 0.0
        exposure = (self._bars_in_market / self._bars_total * 100.0) if self._bars_total > 0 else 0.0

        lines = [
            "========== Portfolio Summary ==========",
            f"Start cash                 : {start_cash:,.2f}",
            f"End cash (realised)        : {end_cash:,.2f}",
            f"Open units                 : {open_units}",
            f"Mark price (last seen)     : {mark_price:,.2f}",
            f"Open value (marked)        : {open_value:,.2f}",
            f"Avg open cost              : {avg_cost:,.2f}",
            f"Unrealised PnL             : {unreal:,.2f}",
            f"Unrealised PnL %           : {unreal_pct:,.2f} %",
            f"Total equity (C+V)         : {total_equity:,.2f}",
            f"Equity ROI % (vs start)    : {equity_roi:,.2f} %",
            "",
            f"Realised PnL               : {realised:,.2f}",
            f"ROI % (realised only)      : {realised_roi:,.2f} %",
            "",
            f"Trades (total/win/loss/flat): {total} / {wins} / {losses} / {flats}",
            f"Win rate                   : {win_rate:,.2f} %",
            f"Profit factor              : {profit_factor if profit_factor != float('inf') else 'inf'}",
            f"Avg win / Avg loss         : {avg_win:,.2f} / {avg_loss:,.2f}",
            f"Expectancy per trade       : {expectancy:,.2f}",
            f"Best / Worst trade         : {best:,.2f} / {worst:,.2f}",
            f"Max consecutive wins/loss  : {max_wins} / {max_losses}",
            f"Avg / Median hold (days)   : {avg_days:,.2f} / {med_days:,.2f}",
            f"Trade return mean / std    : {ret_mean*100:,.2f}% / {ret_std*100:,.2f}%",
            f"Sharpe-like (per trade)    : {sharpe_like:,.2f}",
            "",
            f"Max drawdown (cash only)   : {dd_cash:,.2f}",
            f"Max drawdown (marked eq.)  : {dd_marked:,.2f}",
            f"Exposure (time in market)  : {exposure:,.2f} %",
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
        """Write trade.log (summary + trade logs) and trades.png into dst_dir."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        combined = f"{self.summary()}\n{self.trade_logs()}"
        (dst_dir / "trade.log").write_text(combined, encoding="utf-8")
        self._save_plot(
            price_csv,
            dst_file=dst_dir / "trades.png",
            entry_dt=entry_dt,
            title=f"{asset} Trade Overlay",
        )
