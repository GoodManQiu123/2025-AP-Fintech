"""LLM strategy with separate observe() and robust JSON parsing."""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Deque, Literal, Optional

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal

_JSON_RE = re.compile(r'"?signal"?\s*:\s*"?(\w+)"?', re.I)


class AIStrategy(Strategy):
    """Generate BUY/SELL/HOLD via OpenAI with technical metrics context."""

    def __init__(
        self,
        *,
        history_days: Optional[int] = 60,   # days for history summary
        metrics_window: int = 20,
        rsi_window: int = 14,
        verbose_llm: bool = False,
    ) -> None:
        self._history_days = history_days
        self._hist_prices: Deque[float] = deque()
        self._history_sent = False

        self._holding = False
        self._metrics = RollingWindow(metrics_window)
        self._rsi = RollingWindow(rsi_window)

        sys_prompt = (
            "You are an autonomous trading bot.\n"
            'Reply STRICT JSON: {"signal":"BUY|SELL|HOLD"}'
        )
        self._chat = ChatAgent(system_prompt=sys_prompt, verbose=verbose_llm)

    # --------------------------- observe -----------------------------------
    def observe(self, bar: MarketData) -> None:
        """Collect historical prices; avoid OpenAI calls before entry date."""
        if bar.price is None:
            return
        self._hist_prices.append(bar.price)
        if (
            not self._history_sent
            and self._history_days is not None
            and len(self._hist_prices) >= self._history_days
        ):
            stats = {
                "period_days": len(self._hist_prices),
                "min": min(self._hist_prices),
                "max": max(self._hist_prices),
                "mean": sum(self._hist_prices) / len(self._hist_prices),
            }
            self._chat.send(f"Historical stats:\n{json.dumps(stats)}")
            self._history_sent = True

    # --------------------------- helpers -----------------------------------
    @staticmethod
    def _parse_signal(reply: str) -> Literal["BUY", "SELL", "HOLD"]:
        try:
            return json.loads(reply)["signal"].upper()  # type: ignore[return-value]
        except Exception:
            m = _JSON_RE.search(reply)
            return m.group(1).upper() if m else "HOLD"  # type: ignore[return-value]

    # --------------------------- main API ----------------------------------
    def generate_signal(self, bar: MarketData) -> Signal:
        if bar.price is None:
            return Signal.HOLD

        # ensure at least one history message sent (if not limited by days)
        if not self._history_sent and self._history_days is None:
            stats = {
                "period_bars": len(self._hist_prices),
                "min": min(self._hist_prices, default=bar.price),
                "max": max(self._hist_prices, default=bar.price),
            }
            self._chat.send(f"Historical stats:\n{json.dumps(stats)}")
            self._history_sent = True

        # update metrics windows
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
        sig_text = self._parse_signal(reply)
        signal = Signal[sig_text] if sig_text in Signal.__members__ else Signal.HOLD

        if signal is Signal.BUY:
            self._holding = True
        elif signal is Signal.SELL:
            self._holding = False
        return signal


# --------------------------- factory ---------------------------------------
def build(**kwargs) -> Strategy:
    """Factory for dynamic import (accepts verbose_llm etc.)."""
    return AIStrategy(**kwargs)
