import math

import pytest

from core.metrics import RollingWindow, sma, stddev  # type: ignore  # project import


def test_sma_and_stddev_basic():
    """Basic correctness for SMA and population STDDEV."""
    vals = [1.0, 2.0, 3.0]
    assert sma(vals) == pytest.approx(2.0)
    # population std of [1,2,3] = sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/3) = sqrt(2/3)
    assert stddev(vals) == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-6)


def test_rollingwindow_full_and_rsi_midpoint():
    """RSI should be 50 when gains == losses over the window."""
    rw = RollingWindow(5)
    for v in [1, 2, 3, 2, 1]:
        rw.push(v)
    assert rw.full is True
    assert rw.sma() == pytest.approx(1.8)
    # gains=2, losses=2 -> RS = 1 -> RSI = 50
    assert rw.rsi() == pytest.approx(50.0, abs=1e-6)


def test_rollingwindow_rsi_extremes():
    """No losses -> RSI 100; no gains -> RSI 0 (by formula saturation)."""
    up = RollingWindow(3)
    for v in [1, 2, 3]:
        up.push(v)
    assert up.rsi() == pytest.approx(100.0)

    down = RollingWindow(3)
    for v in [3, 2, 1]:
        down.push(v)
    # When gains=0 and losses>0, RS=0 -> RSI = 0
    assert down.rsi() == pytest.approx(0.0)
