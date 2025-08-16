from core.data_feed import CSVFeed  # type: ignore
from core.types import MarketData  # type: ignore


def test_csvfeed_parses_aliases(make_csv):
    path = make_csv(
        [
            {
                "time": "2024-01-01",
                "open": "9.0",
                "high": "10.0",
                "low": "8.5",
                "close": "9.5",
                "Adj Close": "9.4",
                "Vol": "12345",
            }
        ],
        filename="ALIAS.csv",
    )

    feed = CSVFeed(path, asset="ALIAS")
    bars = list(feed.stream())
    assert len(bars) == 1
    bar: MarketData = bars[0]
    assert bar.asset == "ALIAS"
    assert bar.open == 9.0
    assert bar.high == 10.0
    assert bar.low == 8.5
    assert bar.close == 9.5
    assert bar.adj_close == 9.4  # alias mapping
    assert float(bar.volume) == 12345.0  # Vol → volume
