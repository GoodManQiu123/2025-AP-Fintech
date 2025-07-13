"""Entry-point: feed → strategy → portfolio loop."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal


# ─────────────────────────────────────────────── helpers ──────────────────────
def _load_strategy(dotted_path: str):
    """Import dotted module path and return its build() product."""
    module = importlib.import_module(dotted_path)
    if not hasattr(module, "build"):
        raise AttributeError(f"{dotted_path} must expose a build() factory")
    return module.build()


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trading simulation loop.")
    parser.add_argument(
        "--asset",
        default="AAPL",
        help="Asset symbol, e.g. AAPL / BTCUSDT (default: AAPL)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing CSV files (default: data/)",
    )
    parser.add_argument(
        "--strategy",
        default="strategies.threshold_strategy",
        help="Dotted path to strategy module exposing build() (default threshold)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────── engine ───────────────────────
def run() -> None:
    args = _parse_cli()

    csv_path: Path = Path(args.data_dir) / f"{args.asset.upper()}.csv"
    if not csv_path.exists():
        sys.exit(f"Data file not found: {csv_path}")

    feed = CSVFeed(csv_path, asset=args.asset)
    strategy = _load_strategy(args.strategy)
    portfolio = Portfolio()

    for bar in feed.stream():
        signal = strategy.generate_signal(bar)
        if signal is not Signal.HOLD:
            portfolio.execute(signal, bar)

    # --- simple report -------------------------------------------------------
    print(portfolio.summary())
    for trade in portfolio.trades:
        print(
            f"Trade: entry={trade.entry_price}, exit={trade.exit_price}, "
            f"profit={trade.profit:.2f}"
        )


if __name__ == "__main__":
    run()
