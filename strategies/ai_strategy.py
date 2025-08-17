"""LLM-driven trading strategy with multi-unit, cash-aware execution.

This module defines an ``AIStrategy`` that delegates decision-making to a large
language model (LLM) while enforcing risk and position constraints locally.
The LLM receives a compact JSON context (latest price, technical indicators,
cash, and position state) and must respond with strict JSON indicating the
desired signal and the number of units.

Key behavior:
* History warm-up: optionally compute and send a one-off summary of recent
  prices to the chat agent for additional context.
* Metrics gating: no trading signal is emitted until rolling indicators are
  fully populated.
* Constraint enforcement: the local engine clamps LLM requests to available
  cash, current holdings, and configured unit limits.

Note:
    Logic strictly mirrors the original implementation. Any apparent edge cases
    (e.g., clamping semantics when cash is insufficient) are intentionally
    preserved to avoid functional changes.

The code style follows the Google Python Style Guide.
"""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Deque, Optional

from core.llm.chat_agent import ChatAgent
from core.metrics import RollingWindow
from core.strategy_base import Strategy
from core.types import MarketData, Signal

# Matches JSON-like replies containing "signal" and "units" even if not strict JSON.
_JSON_RE = re.compile(
    r'"?signal"?\s*:\s*"?(\w+)"?\s*,\s*"?(units)"?\s*:\s*"?([0-9]+)"?',
    re.IGNORECASE,
)


class AIStrategy(Strategy):
    """LLM-driven strategy with rich context and multi-unit capability.

    The strategy composes a JSON ``context`` for the LLM and expects back a JSON
    object containing a ``signal`` (``BUY``, ``SELL``, or ``HOLD``) and an
    integer ``units`` field. The class then enforces cash/position constraints
    and updates internal cash and unit balances accordingly.

    Args:
        start_cash: Starting cash balance.
        history_days: If set, number of latest prices to accumulate before
            sending a one-time historical summary (stats) to the LLM.
            If ``None``, history is appended continuously but no summary is sent.
        metrics_window: Rolling window length for SMA/volatility.
        rsi_window: Rolling window length for the RSI-like metric.
        verbose_llm: Whether the chat agent should be verbose.
        max_units: Hard cap on units the LLM can request in a single action.

    Attributes:
        _cash: Current cash balance.
        _units: Current position in units.
        _max_units: Per-action unit cap enforced locally.
        _history_days: Number of days to accumulate before sending stats.
        _hist_prices: Recent price history used for the one-off summary.
        _history_sent: Whether the historical summary has been sent.
        _sma: Rolling window used to compute SMA and volatility.
        _rsi: Rolling window used to compute an RSI-like value.
        _chat: Chat agent instance used to communicate with the LLM.
    """

    def __init__(
        self,
        *,
        start_cash: float = None,
        history_days: Optional[int] = 60,
        metrics_window: int = 20,
        rsi_window: int = 14,
        verbose_llm: bool = True,
        max_units: int = 10,
    ) -> None:
        # ---- Cash & position management -----------------------------------
        self._cash: float = start_cash
        self._units: int = 0
        self._max_units: int = max_units

        # ---- History warm-up ----------------------------------------------
        self._history_days: Optional[int] = history_days
        self._hist_prices: Deque[float] = deque()
        self._history_sent: bool = False

        # ---- Technical metrics --------------------------------------------
        self._sma: RollingWindow = RollingWindow(metrics_window)
        self._rsi: RollingWindow = RollingWindow(rsi_window)

        # ---- Chat agent ----------------------------------------------------
        system_prompt = (
            "You are a professional trading agent.\n"
            "Always output STRICT JSON:\n"
            '{"signal":"BUY|SELL|HOLD","units": <integer>} '
            "When BUY, units must be > 0 and <= max_units.\n"
            "When HOLD, units can be 0 (no action).\n"
            "When SELL, units must be <= current position_unit and > 0.\n"
            "Consider provided indicators and cash/position constraints."
        )
        self._chat: ChatAgent = ChatAgent(
            system_prompt=system_prompt, verbose=verbose_llm
        )

    # --------------------------------------------------------------------- #
    #                                Observe                                #
    # --------------------------------------------------------------------- #
    def observe(self, bar: MarketData) -> None:
        """Ingest one bar and optionally send a one-off history summary.

        Appends the latest price (if available) to the internal history deque.
        Once at least ``history_days`` observations have accumulated (and only
        if ``history_days`` is not ``None``), computes simple summary stats and
        sends them to the chat agent. This is done at most once.

        Args:
            bar: Latest market data with an optional ``price`` field.
        """
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

    # --------------------------------------------------------------------- #
    #                               Utilities                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _safe_json(reply: str) -> tuple[str, int]:
        """Parse ``signal`` and ``units`` from an LLM reply safely.

        The method first attempts to parse strict JSON. If that fails, it
        applies a regex fallback to recover values from JSON-like text.

        Args:
            reply: Raw string returned from the chat agent.

        Returns:
            A tuple of (signal_text, units). On failure, returns (``"HOLD"``, 0).
        """
        try:
            obj = json.loads(reply)
            return obj["signal"].upper(), int(obj["units"])
        except Exception:
            match = _JSON_RE.search(reply)
            if match:
                return match.group(1).upper(), int(match.group(3))
            return "HOLD", 0

    # --------------------------------------------------------------------- #
    #                           Strategy Interface                           #
    # --------------------------------------------------------------------- #
    def generate_signal(self, bar: MarketData) -> Signal:
        """Generate a trading signal given the latest bar.

        This method:
        1) Optionally records history and sends a one-off summary to the LLM.
        2) Updates rolling metrics and requires them to be "full" before acting.
        3) Builds a JSON context and queries the LLM for (signal, units).
        4) Enforces local cash/position/unit constraints.
        5) Updates cash/position state and returns the resulting ``Signal``.

        Args:
            bar: Latest market data tick/candle.

        Returns:
            A ``Signal`` enum value: ``BUY``, ``SELL``, or ``HOLD``.
        """
        if bar.price is None:
            return Signal.HOLD

        # Ensure some history is collected if unlimited, mirroring original logic.
        if not self._history_sent and self._history_days is None:
            self.observe(bar)

        # Update technical indicators (gated on "full" windows).
        self._sma.push(bar.price)
        self._rsi.push(bar.price)
        if not (self._sma.full and self._rsi.full):
            return Signal.HOLD

        # ---------------------- Build LLM context --------------------------
        context = {
            "price": bar.price,
            "sma": self._sma.sma(),
            "volatility": self._sma.std(),
            "rsi": self._rsi.rsi(),
            "cash": self._cash,
            "position_units": self._units,
            "position_value": self._units * bar.price,
            "max_units": self._max_units,
        }
        reply = self._chat.send(json.dumps(context))

        signal_text, requested_units = self._safe_json(reply)
        if signal_text not in ("BUY", "SELL", "HOLD"):
            signal_text, requested_units = "HOLD", 0

        # -------------------- Enforce local constraints --------------------
        if signal_text == "BUY":
            max_affordable = int(self._cash // bar.price)
            # NOTE: The following clamping mirrors original implementation.
            # Any behavioral quirks are preserved intentionally.
            requested_units = max(
                1, min(requested_units, max_affordable, self._max_units)
            )
            if requested_units == 0:
                signal_text = "HOLD"

        elif signal_text == "SELL":
            requested_units = min(requested_units, self._units)
            if requested_units == 0:
                signal_text = "HOLD"

        # ------------------- Update cash/position state --------------------
        if signal_text == "BUY":
            cost = requested_units * bar.price
            self._cash -= cost
            self._units += requested_units
            return Signal.BUY

        if signal_text == "SELL":
            revenue = requested_units * bar.price
            self._cash += revenue
            self._units -= requested_units
            return Signal.SELL

        return Signal.HOLD


# ------------------------------------------------------------------------- #
#                            Factory / Dynamic Load                         #
# ------------------------------------------------------------------------- #
def build(**kwargs) -> Strategy:
    """Factory for dynamic import by the strategy engine.

    Args:
        **kwargs: Keyword arguments forwarded to ``AIStrategy`` constructor.

    Returns:
        A configured ``AIStrategy`` instance.
    """
    return AIStrategy(**kwargs)
