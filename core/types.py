"""Common primitive aliases, signals and market-data container."""
from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import NewType, Optional

# ──────────────────────────────────────────────────────────── primitives ─────
Asset = NewType("Asset", str)          # e.g. "AAPL", "BTCUSDT"
Timestamp = NewType("Timestamp", str)  # ISO-8601 string
Price = NewType("Price", float)
Volume = NewType("Volume", float | int)

# ───────────────────────────────────────────────────────────── signal ─────────
class Signal(Enum):
    """Discrete trading signals."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()

# ─────────────────────────────────────────────────────── market bar DTO ──────
@dataclasses.dataclass(slots=True, frozen=True)
class MarketData:
    """Unified market bar used throughout the engine."""
    asset: Asset
    time: Timestamp
    open: Optional[Price] = None
    high: Optional[Price] = None
    low: Optional[Price] = None
    close: Optional[Price] = None
    adj_close: Optional[Price] = None
    volume: Optional[Volume] = None

    @property
    def price(self) -> Price:
        """Preferred price value for simple strategies."""
        return self.close or self.adj_close  # type: ignore[return-value]
