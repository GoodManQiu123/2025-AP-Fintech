# data_downloader.py
"""Yahoo Finance CSV downloader with an extensible provider interface.

This module provides:
    * BaseDownloader: Abstract façade for market-data providers.
    * YahooDownloader: Concrete implementation using `yfinance`.

Primary responsibilities:
    * Fetch OHLCV data for a symbol within an optional date range/interval.
    * Normalize/flatten provider-specific columns to a canonical schema:
        - open, high, low, close, adj_close, volume
    * Persist the result as a clean CSV with columns:
        asset, time, open, high, low, close, adj_close, volume
    * Offer a small CLI for ad-hoc downloads.

Notes:
    * Behavior intentionally mirrors the original implementation. Edits here are
      limited to formatting and documentation; external API and semantics are
      unchanged.
"""
from __future__ import annotations

import abc
import datetime as dt
import sys
from pathlib import Path
from typing import Final, NewType

import pandas as pd

# -----------------------------------------------------------------------------
# Type aliases / constants
# -----------------------------------------------------------------------------
CsvPath = NewType("CsvPath", Path)

_DATE_FMT: Final[str] = "%Y-%m-%d"
_FIELD_CANON: Final[dict[str, str]] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adjclose": "adj_close",
    "volume": "volume",
}
_FIELD_NAMES = set(_FIELD_CANON.keys())


# -----------------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------------
class BaseDownloader(abc.ABC):
    """Abstract façade for market-data providers."""

    @abc.abstractmethod
    def download(
        self,
        symbol: str,
        start: dt.date | None,
        end: dt.date | None,
        interval: str,
        auto_adjust: bool,
        out_csv: CsvPath,
    ) -> CsvPath:
        """Fetch data and write a CSV.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            start: Inclusive start date (or None).
            end: Exclusive end date (or None).
            interval: Data interval (e.g., "1d", "1h").
            auto_adjust: Whether to return adjusted prices from the provider.
            out_csv: Output CSV path.

        Returns:
            The path to the written CSV.
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Concrete Yahoo Finance implementation
# -----------------------------------------------------------------------------
class YahooDownloader(BaseDownloader):
    """Free Yahoo Finance downloader (no API key required)."""

    # ------------------------------- Helpers ---------------------------------
    def _flatten_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with single-level, lower-case columns.

        The function handles both single-level and MultiIndex columns by
        selecting the level that contains the known OHLCV field names.
        """
        if not isinstance(raw.columns, pd.MultiIndex):
            return raw.rename(columns=lambda c: str(c).lower().replace(" ", ""))

        # Identify the level containing the field names (open/close/…).
        level0 = {str(c).lower().replace(" ", "") for c in raw.columns.get_level_values(0)}
        level1 = {str(c).lower().replace(" ", "") for c in raw.columns.get_level_values(1)}

        if _FIELD_NAMES & level0:
            raw.columns = [c[0] for c in raw.columns]
        elif _FIELD_NAMES & level1:
            raw.columns = [c[1] for c in raw.columns]
        else:
            raise KeyError("Could not locate OHLCV field names in MultiIndex columns.")

        raw.columns = [str(c).lower().replace(" ", "") for c in raw.columns]
        return raw

    def _standardise_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to canonical names and drop extras.

        Keeps only the mapping of provider fields to the canonical schema and
        renames them accordingly.
        """
        keep_cols = {c: _FIELD_CANON[c] for c in df.columns if c in _FIELD_CANON}
        if not keep_cols:
            raise KeyError("No OHLCV fields found after flattening.")
        return df[keep_cols.keys()].rename(columns=keep_cols)

    # ------------------------------- Main API --------------------------------
    def download(
        self,
        symbol: str,
        start: dt.date | None = None,
        end: dt.date | None = None,
        interval: str = "1d",
        auto_adjust: bool = False,
        out_csv: str | Path | None = None,
    ) -> CsvPath:
        """Download OHLCV data for `symbol` and write a canonical CSV.

        Args:
            symbol: Ticker symbol.
            start: Inclusive start date; if both start and end are None, a
                one-year lookback is used by default.
            end: Exclusive end date.
            interval: Data interval (e.g., "1d", "1h", "5m").
            auto_adjust: Whether to return adjusted prices.
            out_csv: Output CSV path; defaults to "data/{SYMBOL}.csv".

        Returns:
            Path to the written CSV.

        Raises:
            ValueError: If start date is later than end date.
            RuntimeError: If the provider returns an empty DataFrame.
            KeyError: If expected OHLCV columns cannot be located.
        """
        import yfinance as yf  # Local import keeps dependency optional.

        # Resolve date window.
        if not start and not end:
            start = dt.date.today() - dt.timedelta(days=365)
        if start and end and start > end:
            raise ValueError("--start must be ≤ --end")

        raw = yf.download(
            tickers=symbol,
            start=start.strftime(_DATE_FMT) if start else None,
            end=end.strftime(_DATE_FMT) if end else None,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise RuntimeError(f"No data returned for symbol={symbol!r}")

        tidy = (
            self._standardise_fields(self._flatten_columns(raw))
            .reset_index()
            .rename(columns={"Date": "time"})
        )

        # Insert asset column at position 0.
        tidy.insert(0, "asset", symbol.upper())

        out_path = CsvPath(Path(out_csv or f"data/{symbol.upper()}.csv"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(out_path, index=False)
        return out_path


# -----------------------------------------------------------------------------
# CLI convenience
# -----------------------------------------------------------------------------
def _parse_date(date_str: str | None) -> dt.date | None:
    """Parse 'YYYY-MM-DD' string to a date, or return None for falsy input."""
    return dt.datetime.strptime(date_str, _DATE_FMT).date() if date_str else None


def _cli() -> None:
    """Command-line interface for ad-hoc Yahoo Finance CSV downloads."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Yahoo Finance CSV.")
    parser.add_argument("symbol", help="Ticker symbol, e.g., AAPL")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--days", type=int, help="Look-back window (days)")
    grp.add_argument("--start", type=str, help="YYYY-MM-DD start date")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD end date (default today)")
    parser.add_argument("--interval", default="1d", help="1d, 1h, 5m …")
    parser.add_argument("--auto-adjust", action="store_true", help="Return adjusted prices")
    parser.add_argument("--out", metavar="PATH", help="Output CSV path")
    args = parser.parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if args.days and not start:
        start = dt.date.today() - dt.timedelta(days=args.days)

    downloader = YahooDownloader()
    try:
        path = downloader.download(
            symbol=args.symbol,
            start=start,
            end=end,
            interval=args.interval,
            auto_adjust=args.auto_adjust,
            out_csv=args.out,
        )
        print(f"Data saved → {path}")
    except Exception as err:  # Surface error and exit with non-zero status.
        sys.exit(f"Error: {err}")


if __name__ == "__main__":
    _cli()
