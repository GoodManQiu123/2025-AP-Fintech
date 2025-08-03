"""Entry-point: CSV feed ➜ strategy ➜ portfolio simulation."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal


def _load_strategy(path: str):
    mod = importlib.import_module(path)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{path} must expose build()")
    return mod.build()


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trading back-test.")
    p.add_argument("--asset", default="AAPL")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--strategy", default="strategies.threshold_strategy")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--entry-date", type=str, help="YYYY-MM-DD funding date")
    return p.parse_args()


def run() -> None:
    args = _cli()

    csv_path = Path(args.data_dir) / f"{args.asset.upper()}.csv"
    if not csv_path.exists():
        sys.exit(f"Data file not found: {csv_path}")

    entry_dt = dt.datetime.fromisoformat(args.entry_date) if args.entry_date else None

    feed = CSVFeed(csv_path, asset=args.asset)
    strategy = _load_strategy(args.strategy)
    portfolio = Portfolio(starting_cash=args.cash)

    for bar in feed.stream():
        bar_dt = dt.datetime.fromisoformat(bar.time)

        if entry_dt and bar_dt < entry_dt:
            # Warm-up: feed bar to observe() only
            if hasattr(strategy, "observe"):
                strategy.observe(bar)
            continue

        signal = strategy.generate_signal(bar)
        if signal is not Signal.HOLD:
            portfolio.execute(signal, bar)

    # ---------- results ----------------------------------------------------
    print(portfolio.summary())
    print("\n----- Trade Log -----")
    for t in portfolio.trades:
        print(
            f"{t.entry_time.date()} BUY {t.units}@{t.entry_price:.2f} → "
            f"{t.exit_time.date()} SELL @{t.exit_price:.2f}  PnL {t.profit:.2f}"
        )


if __name__ == "__main__":
    run()
