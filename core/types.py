"""Common type aliases and enums used across the codebase."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NewType

# --- Trading signal -----------------------------------------------------------


class Signal(Enum):
    """Discrete trading signals output by any strategy."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()


Price = NewType("Price", float)
Timestamp = NewType("Timestamp", str)  # ISO-8601 string for MVP


@dataclass(slots=True, frozen=True)
class MarketData:
    """Lightweight market tick fed into strategies."""
    time: Timestamp
    price: Price
