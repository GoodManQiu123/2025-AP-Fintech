"""Minimal technical metrics helpers."""
from collections import deque
from typing import Deque, List


def sma(values: List[float]) -> float:
    return sum(values) / len(values)


def stddev(values: List[float]) -> float:
    mu = sma(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


class RollingWindow:
    """Fixed-size rolling window storing recent floats."""

    def __init__(self, size: int) -> None:
        self._dq: Deque[float] = deque(maxlen=size)

    def push(self, value: float) -> None:
        self._dq.append(value)

    @property
    def full(self) -> bool:
        return len(self._dq) == self._dq.maxlen

    def sma(self) -> float:
        return sma(list(self._dq))

    def std(self) -> float:
        return stddev(list(self._dq))

    def rsi(self) -> float:
        gains = 0.0
        losses = 0.0
        vals = list(self._dq)
        for prev, cur in zip(vals, vals[1:]):
            delta = cur - prev
            gains += max(delta, 0)
            losses += max(-delta, 0)
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1 + rs))
