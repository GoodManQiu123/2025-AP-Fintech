"""Advanced LLM strategy supporting multi-unit & cash-aware trading."""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Deque, Literal, Optional

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal

_JSON_RE = re.compile(
    r'"?signal"?\s*:\s*"?(\w+)"?\s*,\s*"?(units)"?\s*:\s*"?([0-9]+)"?', re.I
)


class AIStrategy(Strategy):
    """LLM-driven strategy with rich context and multi-unit capability."""

    def __init__(
        self,
        *,
        start_cash: float = 10_000.0,
        history_days: Optional[int] = 60,
        metrics_window: int = 20,
        rsi_window: int = 14,
        verbose_llm: bool = True,
        max_units: int = 10,
    ) -> None:
        # ---- cash & position management -----------------------------------
        self._cash = start_cash
        self._units = 0
        self._max_units = max_units

        # ---- history warm-up ----------------------------------------------
        self._history_days = history_days
        self._hist_prices: Deque[float] = deque()
        self._history_sent = False

        # ---- technical metrics --------------------------------------------
        self._sma = RollingWindow(metrics_window)
        self._rsi = RollingWindow(rsi_window)

        # ---- Chat agent ----------------------------------------------------
        system_prompt = (
            "You are a professional trading agent.\n"
            "Always output STRICT JSON:\n"
            '{"signal":"BUY|SELL|HOLD","units": <positive integer>} '
            "When HOLD, units can be 0 (no action).\n"
            "When SELL, units must be <= current position_unit.\n"
            "Consider provided indicators and cash/position constraints."
        )
        self._chat = ChatAgent(system_prompt=system_prompt, verbose=verbose_llm)

    # ------------------------- observe phase -------------------------------
    def observe(self, bar: MarketData) -> None:
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

    # ------------------------- helper utils --------------------------------
    @staticmethod
    def _safe_json(reply: str) -> tuple[str, int]:
        """Parse signal & units; fall back to HOLD/0 on error."""
        try:
            obj = json.loads(reply)
            return obj["signal"].upper(), int(obj["units"])
        except Exception:
            m = _JSON_RE.search(reply)
            if m:
                return m.group(1).upper(), int(m.group(3))
            return "HOLD", 0

    # ------------------------- main strategy API ---------------------------
    def generate_signal(self, bar: MarketData) -> Signal:
        if bar.price is None:
            return Signal.HOLD

        # ensure history once if unlimited
        if not self._history_sent and self._history_days is None:
            self.observe(bar)

        # update metrics
        self._sma.push(bar.price)
        self._rsi.push(bar.price)
        if not (self._sma.full and self._rsi.full):
            return Signal.HOLD

        # ---------- build prompt ------------------------------------------
        context = {
            "price": bar.price,
            "sma": self._sma.sma(),
            "volatility": self._sma.std(),
            "rsi": self._rsi.rsi(),
            "cash": self._cash,
            "position_units": self._units,
            "position_value": self._units * bar.price,
        }
        reply = self._chat.send(json.dumps(context))

        sig_text, req_units = self._safe_json(reply)
        if sig_text not in ("BUY", "SELL", "HOLD"):
            sig_text, req_units = "HOLD", 0

        # ---------- enforce constraints -----------------------------------
        if sig_text == "BUY":
            max_affordable = int(self._cash // bar.price)
            req_units = max(1, min(req_units, max_affordable, self._max_units))
            if req_units == 0:
                sig_text = "HOLD"
        elif sig_text == "SELL":
            req_units = min(req_units, self._units)
            if req_units == 0:
                sig_text = "HOLD"

        # ---------- update cash/position ----------------------------------
        if sig_text == "BUY":
            cost = req_units * bar.price
            self._cash -= cost
            self._units += req_units
            return Signal.BUY

        if sig_text == "SELL":
            revenue = req_units * bar.price
            self._cash += revenue
            self._units -= req_units
            return Signal.SELL

        return Signal.HOLD


# ---------------- factory for dynamic import ------------------------------
def build(**kwargs) -> Strategy:
    return AIStrategy(**kwargs)
