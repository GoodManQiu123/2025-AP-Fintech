# engine.py
"""Run backtests, print reports, and save logs/plots & conversation JSONs.

This script:
  1) Loads market data from CSV.
  2) Dynamically imports a strategy via a dotted path and instantiates it
     using the strategy's ``build()`` factory.
  3) Streams bars to the strategy to obtain trading signals.
  4) Executes trades through the Portfolio and marks positions to market.
  5) Prints a console summary and exports logs (including strategy chat logs
     if the strategy exposes ``export_chat_logs``).

Note:
    This file intentionally preserves the original behavior and I/O contract.
    Comments and docstrings are in English; formatting follows the Google
    Python Style Guide.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import sys
from pathlib import Path
from typing import Any

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal


def _load_strategy(dotted_path: str) -> Any:
    """Import a strategy module and return an instance via its build() factory.

    Args:
        dotted_path: Dotted import path to the strategy module
            (e.g., "strategies.threshold_strategy").

    Returns:
        A strategy instance constructed by the module's ``build()`` function.

    Raises:
        AttributeError: If the module does not expose a ``build`` factory.
        ModuleNotFoundError: If the module cannot be imported.
    """
    try:
        mod = importlib.import_module(dotted_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to import strategy module: {dotted_path}"
        ) from exc

    if not hasattr(mod, "build"):
        raise AttributeError(f"{dotted_path} must expose build() factory")

    return mod.build()


def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run trading back-test.")
    parser.add_argument("--asset", default="AAPL", help="Asset symbol (e.g., AAPL).")
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing the CSV data files."
    )
    parser.add_argument(
        "--strategy",
        default="strategies.threshold_strategy",
        help="Dotted path to the strategy module exposing build().",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10_000.0,
        help="Starting cash balance for the backtest.",
    )
    parser.add_argument(
        "--entry-date",
        help="ISO date (YYYY-MM-DD). Bars strictly before this date are used "
        "for warm-up (observe) only; trading starts on/after this date.",
    )
    return parser.parse_args()


def run() -> None:
    """Run the backtest loop and export summary and logs."""
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

        # Warm-up phase before entry date (collect context, no trading/marking).
        if entry_dt and bar_dt < entry_dt:
            if hasattr(strategy, "observe"):
                strategy.observe(bar)
            continue

        # Test period: decide & execute first, then mark to market with this bar.
        sig = strategy.generate_signal(bar)
        if sig is not Signal.HOLD:
            units = getattr(strategy, "last_units", 1)
            portfolio.execute(sig, bar, units=units)

        # Mark-to-market with end-of-bar price (keeps open value synced to latest).
        portfolio.mark_from_bar(bar)

    # Console outputs.
    print(portfolio.summary())
    print(portfolio.trade_logs())

    # Persistent logs.
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"{args.asset.upper()}_{ts}"
    portfolio.export_logs(log_dir, csv_path, entry_dt, asset=args.asset.upper())

    # Conversation dumps (if provided by the strategy).
    if hasattr(strategy, "export_chat_logs"):
        try:
            strategy.export_chat_logs(log_dir)
        except Exception as exc:  # noqa: BLE001 - display warning but continue
            print(f"[warn] failed to export conversation logs: {exc}")

    print(f"\nLogs & chart saved to: {log_dir.resolve()}")


if __name__ == "__main__":
    run()
