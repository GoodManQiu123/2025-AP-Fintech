"""LLM strategy with rich metrics and configurable history/trading window."""
from __future__ import annotations

import json
from collections import deque
from typing import Deque, Literal, Optional

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal


class AIStrategy(Strategy):
    """Generate signals via OpenAI, with SMA/volatility/RSI context."""

    def __init__(
        self,
        *,
        history_days: Optional[int] = None,   # None = entire CSV
        trade_days: Optional[int] = None,     # None = until file end
        metrics_window: int = 20,
        rsi_window: int = 14,
        verbose: bool = False,
    ) -> None:
        self._history_days = history_days
        self._trade_days = trade_days
        self._observed_days = 0
        self._traded_days = 0

        self._hist_prices: Deque[float] = deque()
        self._holding = False

        self._metrics = RollingWindow(metrics_window)
        self._rsi = RollingWindow(rsi_window)

        sys_msg = (
            "You are an autonomous trading agent. "
            'Reply strictly JSON: {"signal":"BUY|SELL|HOLD","comment":"..." }.'
        )
        self._chat = ChatAgent(system_prompt=sys_msg, verbose=verbose)

        self._history_sent = False

    # ---------------------------- helper ------------------------------------
    def _maybe_send_history(self) -> None:
        if self._history_sent:
            return
        if self._history_days is None or len(self._hist_prices) >= self._history_days:
            stats = {
                "period_days": len(self._hist_prices),
                "min": min(self._hist_prices),
                "max": max(self._hist_prices),
                "mean": sum(self._hist_prices) / len(self._hist_prices),
            }
            self._chat.send(
                "Historical metrics for the asset:\n" + json.dumps(stats)
            )
            self._history_sent = True

    # ---------------------------- strategy API ------------------------------
    def generate_signal(self, bar: MarketData) -> Signal:
        if bar.price is None:
            return Signal.HOLD

        self._observed_days += 1
        self._hist_prices.append(bar.price)

        # send history once ready
        self._maybe_send_history()
        if not self._history_sent:
            return Signal.HOLD

        # stop trading if reached limit
        if self._trade_days is not None and self._traded_days >= self._trade_days:
            return Signal.HOLD

        # metrics update
        self._metrics.push(bar.price)
        self._rsi.push(bar.price)
        if not (self._metrics.full and self._rsi.full):
            return Signal.HOLD

        prompt = json.dumps(
            {
                "price": bar.price,
                "sma": self._metrics.sma(),
                "volatility": self._metrics.std(),
                "rsi": self._rsi.rsi(),
                "holding": self._holding,
            }
        )
        reply = self._chat.send(prompt)
        try:
            sig_text: Literal["BUY", "SELL", "HOLD"] = json.loads(reply)["signal"]
        except Exception:
            sig_text = "HOLD"

        signal = Signal[sig_text]
        if signal is Signal.BUY:
            self._holding = True
        elif signal is Signal.SELL:
            self._holding = False
        self._traded_days += 1
        return signal


# ----------------------------- factory --------------------------------------
def build(
    *,
    history_days: Optional[int] = None,
    trade_days: Optional[int] = None,
    verbose: bool = False,
) -> Strategy:
    """Factory with optional overrides."""
    return AIStrategy(
        history_days=history_days,
        trade_days=trade_days,
        verbose=verbose,
    )
