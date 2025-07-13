"""Market data feed abstractions."""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from core.types import MarketData, Price, Timestamp


class BaseFeed(ABC):
    """Abstract stream of `MarketData` ticks."""

    @abstractmethod
    def stream(self) -> Iterator[MarketData]:
        """Yield market ticks forever (or until EOF)."""
        raise NotImplementedError


class CSVFeed(BaseFeed):
    """CSV-backed feed for quick offline simulation.

    Expected header: time,price
    """

    def __init__(self, csv_path: str | Path) -> None:
        self._csv_path = Path(csv_path)

    def stream(self) -> Iterator[MarketData]:
        with self._csv_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield MarketData(
                    time=Timestamp(row["time"]),
                    price=Price(float(row["price"])),
                )
