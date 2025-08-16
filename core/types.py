# types.py
"""Common primitive aliases, trading signals, and the market-data container."""
from __future__ import annotations

import dataclasses
from enum import Enum, auto
from typing import NewType, Optional

# ───────────────────────────────────────────────────────────── primitives ────
Asset = NewType("Asset", str)          # e.g., "AAPL", "BTCUSDT"
Timestamp = NewType("Timestamp", str)  # ISO-8601 string
Price = NewType("Price", float)
Volume = NewType("Volume", float | int)


# ──────────────────────────────────────────────────────────────── signals ────
class Signal(Enum):
    """Discrete trading signals."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()


# ───────────────────────────────────────────────────────── market bar DTO ────
@dataclasses.dataclass(slots=True, frozen=True)
class MarketData:
    """Unified market bar used throughout the engine.

    Attributes:
        asset: Asset symbol.
        time: ISO-8601 timestamp string.
        open: Open price (optional).
        high: High price (optional).
        low: Low price (optional).
        close: Close price (optional).
        adj_close: Adjusted close price (optional).
        volume: Volume (optional).
    """
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
        """Preferred price value for simple strategies.

        Returns:
            ``close`` if available, otherwise ``adj_close``.
            May be ``None`` if neither is provided.
        """
        # The cast-ignore mirrors the original behavior where callers handle None.
        return self.close or self.adj_close  # type: ignore[return-value]
