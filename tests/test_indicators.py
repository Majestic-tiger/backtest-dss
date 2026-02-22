"""Tests for indicator functions."""
import numpy as np
import pandas as pd
import pytest

from dongpa_engine import (
    wilder_rsi,
    cross_up,
    cross_down,
    moving_average,
    golden_cross,
    death_cross,
    weekly_roc,
    to_weekly_close,
)


class TestWilderRSI:
    def test_returns_series(self):
        prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103,
                            104, 103, 102, 101, 100, 99, 98, 99, 100, 101])
        rsi = wilder_rsi(prices, period=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)

    def test_fills_nan_with_50(self):
        prices = pd.Series([100.0])
        rsi = wilder_rsi(prices, period=14)
        assert rsi.iloc[0] == 50.0

    def test_mostly_up_high_rsi(self):
        # Mostly up with small dips to avoid 0 denominator
        prices = [100, 102, 101, 103, 104, 103.5, 105, 106, 105.5, 107,
                  108, 107.5, 109, 110, 109.5, 111, 112, 111.5, 113, 114]
        rsi = wilder_rsi(pd.Series(prices), period=5)
        assert rsi.iloc[-1] > 65

    def test_all_down_near_0(self):
        prices = pd.Series(range(120, 100, -1))
        rsi = wilder_rsi(prices, period=5)
        assert rsi.iloc[-1] < 10


class TestCrossUpDown:
    def test_cross_up(self):
        assert cross_up(49.0, 51.0, 50.0) is True
        assert cross_up(50.0, 51.0, 50.0) is False  # prev not < level
        assert cross_up(49.0, 49.5, 50.0) is False  # curr < level

    def test_cross_down(self):
        assert cross_down(51.0, 49.0, 50.0) is True
        assert cross_down(50.0, 49.0, 50.0) is False  # prev not > level
        assert cross_down(51.0, 50.5, 50.0) is False  # curr > level


class TestMovingAverage:
    def test_single_value(self):
        s = pd.Series([10.0])
        result = moving_average(s, 5)
        assert result.iloc[0] == 10.0

    def test_matches_manual(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(s, 3)
        assert result.iloc[-1] == pytest.approx(4.0)  # (3+4+5)/3


class TestGoldenDeathCross:
    def test_golden_cross(self):
        short = pd.Series([10.0, 10.0, 11.0])
        long = pd.Series([11.0, 11.0, 10.0])
        gc = golden_cross(short, long)
        assert gc.iloc[-1] is np.True_

    def test_death_cross(self):
        short = pd.Series([11.0, 11.0, 10.0])
        long = pd.Series([10.0, 10.0, 11.0])
        dc = death_cross(short, long)
        assert dc.iloc[-1] is np.True_


class TestWeeklyROC:
    def test_positive_roc(self):
        s = pd.Series([100, 101, 102, 103, 110])
        roc = weekly_roc(s, period=1)
        assert roc.iloc[-1] > 0

    def test_negative_roc(self):
        s = pd.Series([110, 109, 108, 107, 100])
        roc = weekly_roc(s, period=1)
        assert roc.iloc[-1] < 0
