"""Main loop wiring data feed → strategy → portfolio."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from core.data_feed import CSVFeed
from core.portfolio import Portfolio
from core.types import Signal

# -- Select strategy at runtime -----------------------------------------------
STRATEGY_NAME = "strategies.threshold_strategy"  # change to GPT later

# -- Paths --------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_CSV = BASE_DIR / "data" / "sample_prices.csv"


def _load_strategy(strategy_module: str):
    """Dynamic import so users can swap strategies via CLI/ENV later."""
    module = importlib.import_module(strategy_module)
    return module.build()  # each strategy exposes build() factory


def run() -> None:
    feed = CSVFeed(DATA_CSV)
    strategy = _load_strategy(STRATEGY_NAME)
    portfolio = Portfolio()

    for tick in feed.stream():
        signal = strategy.generate_signal(tick)
        if signal != Signal.HOLD:
            portfolio.execute(signal, tick)

    print(portfolio.summary())
    for trade in portfolio.trades:
        print(f"Trade: entry={trade.entry_price}, exit={trade.exit_price}, "
              f"profit={trade.profit:.2f}")


if __name__ == "__main__":  # `python -m core.engine`
    sys.exit(run())
