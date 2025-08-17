import datetime as dt

from core.portfolio import Portfolio  # type: ignore
from core.types import MarketData, Signal  # type: ignore


def _bar(t: str, price: float) -> MarketData:
    return MarketData(asset="TEST", time=t, close=price)


def test_buy_sell_fifo_and_realised_pnl_and_mark():
    pf = Portfolio(start_cash=100.0)

    # BUY 5 @ 10 (cash -> 50 left)
    pf.execute(Signal.BUY, _bar("2024-01-01", 10.0), units=5)
    assert pf.cash == 50.0
    assert pf.open_units == 5

    # SELL 2 @ 12 -> realised +4, cash +24
    pf.execute(Signal.SELL, _bar("2024-01-02", 12.0), units=2)
    assert pf.realised_pnl == 4.0
    assert pf.open_units == 3
    assert pf.cash == 74.0  # 50 + 2*12

    # Mark-to-market @ 11
    pf.mark_from_bar(_bar("2024-01-03", 11.0))
    # summary() should include key sections/lines
    text = pf.summary()
    for needle in [
        "Realised PnL",
        "Unrealised PnL",
        "Max drawdown",
        "Exposure (time in market)",
    ]:
        assert needle in text

    # Trade log line format sanity check
    logs = pf.trade_logs()
    assert "Trade Log" in logs
    assert "@12.00" in logs or "@12" in logs
