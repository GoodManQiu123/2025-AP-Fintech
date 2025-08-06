"""Entry-point: CSV feed ➜ strategy ➜ portfolio (includes plotting)."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal


# ───────────────────────── helper functions ──────────────────────────
def _load_strategy(dotted: str):
    mod = importlib.import_module(dotted)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{dotted} must expose build()")
    return mod.build()


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trading back-test.")
    p.add_argument("--asset", default="AAPL")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--strategy", default="strategies.threshold_strategy")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument(
        "--entry-date",
        type=str,
        help="YYYY-MM-DD funding date; earlier bars only feed observe()",
    )
    return p.parse_args()


# ───────────────────────────── main loop ─────────────────────────────
def run() -> None:
    args = _cli()

    csv_path = Path(args.data_dir) / f"{args.asset.upper()}.csv"
    if not csv_path.exists():
        sys.exit(f"Data file not found: {csv_path}")

    entry_dt = (
        dt.datetime.fromisoformat(args.entry_date) if args.entry_date else None
    )

    feed = CSVFeed(csv_path, asset=args.asset)
    strategy = _load_strategy(args.strategy)
    portfolio = Portfolio(starting_cash=args.cash)

    for bar in feed.stream():
        bar_dt = dt.datetime.fromisoformat(bar.time)

        if entry_dt and bar_dt < entry_dt:
            if hasattr(strategy, "observe"):
                strategy.observe(bar)
            continue

        signal = strategy.generate_signal(bar)
        if signal is not Signal.HOLD:
            units = getattr(strategy, "last_units", 1)
            portfolio.execute(signal, bar, units=units)

    # --- textual output ---
    print(portfolio.summary())
    print(portfolio.trade_logs())

    # --- visual output ---
    portfolio.plot_trades(csv_path, entry_dt, title=f"{args.asset} Trade Overlay")


if __name__ == "__main__":
    run()
