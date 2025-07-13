"""CSV → MarketBar feed."""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from core.types import Asset, MarketData, Price, Timestamp, Volume

# ───────────────────────────────────────────────────────── helper consts ─────
_OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "open": ("open",),
    "high": ("high",),
    "low": ("low",),
    "close": ("close",),
    "adj_close": ("adj_close", "adjclose", "adj close"),
    "volume": ("volume", "vol"),
}


# ────────────────────────────────────────────────────────────── base feed ─────
class BaseFeed(ABC):
    """Abstract iterator of :class:`MarketBar`."""

    @abstractmethod
    def stream(self) -> Iterator[MarketData]:
        """Yield successive market bars."""
        raise NotImplementedError


# ───────────────────────────────────────────────────────────── CSV feed ──────
class CSVFeed(BaseFeed):
    """Lightweight CSV adapter (stock / crypto / anything tabular)."""

    def __init__(self, csv_path: str | Path, *, asset: str | None = None) -> None:
        self._path = Path(csv_path)
        self._asset = Asset(asset or self._path.stem.upper())

    # --- private helpers -----------------------------------------------------
    @staticmethod
    def _norm(col: str) -> str:
        return col.lower().replace(" ", "").replace("_", "")

    def _header_map(self, header: list[str]) -> dict[str, str]:
        """Return mapping of canonical → CSV column names."""
        norm2raw = {self._norm(col): col for col in header}
        if "time" not in norm2raw:
            raise KeyError("'time' column missing in CSV")

        mapping = {"time": norm2raw["time"]}
        for canon, aliases in _OPTIONAL_ALIASES.items():
            for alias in (canon, *aliases):
                key = self._norm(alias)
                if key in norm2raw:
                    mapping[canon] = norm2raw[key]
                    break
        return mapping

    def _parse_float(self, raw_val: str | None) -> float | None:
        return float(raw_val) if raw_val else None

    # --- public API ---------------------------------------------------------
    def stream(self) -> Iterator[MarketData]:
        with self._path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            mapping = self._header_map(reader.fieldnames or [])

            for row in reader:
                yield MarketData(
                    asset=self._asset,
                    time=Timestamp(row[mapping["time"]]),
                    open=self._parse_float(row.get(mapping.get("open", ""))),
                    high=self._parse_float(row.get(mapping.get("high", ""))),
                    low=self._parse_float(row.get(mapping.get("low", ""))),
                    close=self._parse_float(row.get(mapping.get("close", ""))),
                    adj_close=self._parse_float(row.get(mapping.get("adj_close", ""))),
                    volume=Volume(self._parse_float(row.get(mapping.get("volume", ""))))
                    if mapping.get("volume") and row.get(mapping["volume"])
                    else None,
                )
