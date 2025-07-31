"""Entry‑point: feed → strategy → portfolio loop."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal


def _load_strategy(module_path: str):
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "build"):
        raise AttributeError(f"{module_path} must expose build() factory")
    return mod.build()


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trading simulation loop.")
    p.add_argument("--asset", default="AAPL", help="Asset symbol (default AAPL)")
    p.add_argument("--data-dir", default="data", help="CSV directory")
    p.add_argument("--strategy", default="strategies.threshold_strategy")
    p.add_argument("--cash", type=float, default=10_000.0, help="Starting cash")
    p.add_argument(
        "--entry-date",
        type=str,
        help="YYYY-MM-DD: only trade on/after this date, "
        "but earlier data is still fed into strategy for context.",
    )
    return p.parse_args()


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
        signal = strategy.generate_signal(bar)
        if entry_dt and dt.datetime.fromisoformat(bar.time) < entry_dt:
            # Before funding date: ignore trades, but still build context
            continue
        if signal is not Signal.HOLD:
            portfolio.execute(signal, bar)

    # -------- results -------------------------------------------------------
    print(portfolio.summary())
    print("\n----- Trade Log -----")
    for t in portfolio.trades:
        print(
            f"{t.entry_time.date()} BUY {t.units}@{t.entry_price:.2f}  →  "
            f"{t.exit_time.date()} SELL @{t.exit_price:.2f}  PnL {t.profit:.2f}"
        )


if __name__ == "__main__":
    run()
