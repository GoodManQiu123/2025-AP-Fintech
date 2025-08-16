from core.portfolio import Portfolio  # type: ignore
from core.types import MarketData  # type: ignore

from tests.dummy_strategy import build


def test_minimal_loop_buy_then_sell(make_csv):
    # Prepare a tiny price series that goes up (so strategy will sell).
    rows = [
        {"time": "2024-01-01", "close": 10.0},
        {"time": "2024-01-02", "close": 10.5},
    ]
    # We don't need CSVFeed here; construct bars directly for brevity.
    bars = [MarketData(asset="TEST", time=r["time"], close=r["close"]) for r in rows]

    strat = build()
    pf = Portfolio(starting_cash=100.0)

    for bar in bars:
        sig = strat.generate_signal(bar)
        if sig.name != "HOLD":
            pf.execute(sig, bar, units=getattr(strat, "last_units", 1))
        pf.mark_from_bar(bar)

    # We should have exactly one round-trip trade with positive profit.
    assert len(pf.trades) == 1
    assert pf.trades[0].profit > 0.0
