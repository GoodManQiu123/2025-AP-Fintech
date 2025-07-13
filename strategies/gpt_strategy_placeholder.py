"""Skeleton for a GPT-driven strategy (not wired to OpenAI yet)."""
from __future__ import annotations

from core.strategy_base import Strategy
from core.types import MarketData, Signal


class GPTStrategy(Strategy):
    """Calls a GPT endpoint to decide each signal (placeholder)."""

    def __init__(self) -> None:
        self._holding = False

    # TODO: Replace with real OpenAI calls
    def _mock_gpt_decision(self, price: float) -> Signal:
        return Signal.HOLD if price % 2 else Signal.BUY

    def generate_signal(self, tick: MarketData) -> Signal:
        signal = self._mock_gpt_decision(tick.price)
        if signal == Signal.BUY:
            self._holding = True
        elif signal == Signal.SELL:
            self._holding = False
        return signal


def build() -> Strategy:  # factory for dynamic import
    return GPTStrategy()
