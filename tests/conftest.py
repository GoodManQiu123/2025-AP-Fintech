import csv
from pathlib import Path
from typing import Iterable, Mapping

import pytest


@pytest.fixture
def make_csv(tmp_path):
    """Create a minimal OHLCV CSV and return its path.

    Usage:
        path = make_csv(
            rows=[{"time": "2024-01-01", "close": 10.0}, ...],
            filename="TEST.csv"
        )
    """
    def _make_csv(rows: Iterable[Mapping], filename: str = "TEST.csv") -> Path:
        path = tmp_path / filename
        rows = list(rows)
        if not rows:
            raise ValueError("rows must be non-empty")
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        return path

    return _make_csv
