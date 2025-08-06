"""Run back-test, print reports, and save logs/plot."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal


# ---------------------------- helpers ----------------------------
def _load_strategy(dotted_path: str):
    mod = importlib.import_module(dotted_path)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{dotted_path} must expose build() factory")
    return mod.build()


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trading back-test.")
    p.add_argument("--asset", default="AAPL")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--strategy", default="strategies.threshold_strategy")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--entry-date", help="YYYY-MM-DD funding date")
    return p.parse_args()


# ----------------------------- main ------------------------------
def run() -> None:
    args = _parse_cli()
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
            if hasattr(strategy, "observe"):
                strategy.observe(bar)
            continue

        sig = strategy.generate_signal(bar)
        if sig is not Signal.HOLD:
            units = getattr(strategy, "last_units", 1)
            portfolio.execute(sig, bar, units=units)

    # ------------- console output -------------
    print(portfolio.summary())
    print(portfolio.trade_logs())

    # ------------- persistent logs ------------
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"{args.asset.upper()}_{ts}"
    portfolio.export_logs(log_dir, csv_path, entry_dt, asset=args.asset.upper())
    print(f"\nLogs & chart saved to: {log_dir.resolve()}")


if __name__ == "__main__":
    run()
