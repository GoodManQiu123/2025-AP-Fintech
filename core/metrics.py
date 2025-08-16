# metrics.py
"""Minimal technical metrics helpers.

This module provides:
    * Simple moving average (SMA).
    * Population standard deviation.
    * ``RollingWindow`` helper for fixed-size rolling metrics including SMA,
      stddev, and a simple RSI-like calculation.

Behavior matches the original implementation; additional docstrings and
validation improve readability without changing functionality when used
as intended (i.e., after windows are "full").
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List


def sma(values: List[float]) -> float:
    """Return the simple moving average of ``values``.

    Args:
        values: Non-empty list of floats.

    Returns:
        The arithmetic mean.

    Raises:
        ZeroDivisionError: If ``values`` is empty (mirrors original behavior).
    """
    return sum(values) / len(values)


def stddev(values: List[float]) -> float:
    """Return the population standard deviation of ``values``.

    Args:
        values: Non-empty list of floats.

    Returns:
        The square root of the mean squared deviation from the mean.

    Raises:
        ZeroDivisionError: If ``values`` is empty (via ``sma``).
    """
    mu = sma(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


class RollingWindow:
    """Fixed-size rolling window storing recent floats.

    The window supports:
      * ``push`` to append a new value (dropping the oldest when full).
      * ``full`` property to indicate readiness (len == maxlen).
      * ``sma()``, ``std()``, and ``rsi()`` metrics.

    Notes:
        * ``rsi()`` follows the classical gains/losses ratio (no Wilder's EMA).
        * When ``losses`` is zero, RSI saturates at 100.0 as in the original.
    """

    def __init__(self, size: int) -> None:
        """Initialize a rolling window.

        Args:
            size: Maximum number of elements to retain (must be >= 1).
        """
        self._dq: Deque[float] = deque(maxlen=size)

    def push(self, value: float) -> None:
        """Append ``value`` to the window."""
        self._dq.append(value)

    @property
    def full(self) -> bool:
        """Return True if the window is filled to its maximum length."""
        return len(self._dq) == self._dq.maxlen

    def sma(self) -> float:
        """Return the simple moving average of the window."""
        return sma(list(self._dq))

    def std(self) -> float:
        """Return the population standard deviation of the window."""
        return stddev(list(self._dq))

    def rsi(self) -> float:
        """Return a simple RSI-like value over the window.

        Returns:
            RSI in [0, 100]. If there are no losses in the period, returns 100.0.
        """
        gains = 0.0
        losses = 0.0
        vals = list(self._dq)
        for prev, cur in zip(vals, vals[1:]):
            delta = cur - prev
            gains += max(delta, 0.0)
            losses += max(-delta, 0.0)

        if losses == 0:
            return 100.0

        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))
