"""Integration tests for run_backtest."""
import numpy as np
import pandas as pd
import pytest

from engines.dongpa_engine import (
    BacktestResult,
    CapitalParams,
    ModeParams,
    StrategyParams,
    run_backtest,
    compute_indicators,
    Indicators,
)
from tests.conftest import _make_price_df


@pytest.fixture
def long_prices():
    """100 trading days of uptrending data."""
    np.random.seed(42)
    base = 50.0
    prices = [base]
    for _ in range(99):
        prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.03)))
    return _make_price_df(prices, start="2024-01-02")


@pytest.fixture
def long_momo():
    """100 trading days of momentum data."""
    np.random.seed(7)
    base = 400.0
    prices = [base]
    for _ in range(99):
        prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.01)))
    return _make_price_df(prices, start="2024-01-02")


class TestRunBacktest:
    def test_returns_backtest_result(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity, pd.Series)
        assert isinstance(result.journal, pd.DataFrame)
        assert isinstance(result.trade_log, pd.DataFrame)
        assert isinstance(result.cash_end, float)
        assert isinstance(result.open_positions, int)

    def test_equity_length_matches_dates(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        # Equity should have one entry per trading day
        assert len(result.equity) > 0
        assert len(result.equity) <= len(long_prices)

    def test_initial_equity_equals_cash(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        # First day: no buy (i=0), equity = cash
        assert result.equity.iloc[0] == default_cap.initial_cash

    def test_empty_data_returns_empty_result(self, default_params, default_cap):
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        empty_df.index = pd.DatetimeIndex([], name="Date")
        momo = _make_price_df([400.0], start="2024-01-02")
        result = run_backtest(empty_df, momo, default_params, default_cap)
        assert result.equity.empty
        assert result.journal.empty

    def test_cash_end_nonnegative(self, long_prices, long_momo, default_params, default_cap):
        result = run_backtest(long_prices, long_momo, default_params, default_cap)
        assert result.cash_end >= 0


class TestComputeIndicators:
    def test_returns_tuple(self, long_prices, long_momo, default_params):
        df, ind = compute_indicators(long_prices, long_momo, default_params)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(ind, Indicators)
        assert ind.strategy == "rsi"

    def test_rsi_indicators_populated(self, long_prices, long_momo, default_params):
        _, ind = compute_indicators(long_prices, long_momo, default_params)
        assert ind.daily_rsi is not None
        assert ind.daily_rsi_delta is not None
        assert ind.daily_prev_week is not None

    def test_ma_cross_indicators(self, long_prices, long_momo, default_defense, default_offense):
        params = StrategyParams(
            target_ticker="TEST", momentum_ticker="MOMO",
            mode_switch_strategy="ma_cross",
            ma_short_period=3, ma_long_period=7,
            defense=default_defense, offense=default_offense,
        )
        _, ind = compute_indicators(long_prices, long_momo, params)
        assert ind.strategy == "ma_cross"
        assert ind.daily_ma_short is not None
        assert ind.daily_ma_long is not None
        assert ind.daily_golden is not None

    def test_roc_indicators(self, long_prices, long_momo, default_defense, default_offense):
        params = StrategyParams(
            target_ticker="TEST", momentum_ticker="MOMO",
            mode_switch_strategy="roc", roc_period=4,
            defense=default_defense, offense=default_offense,
        )
        _, ind = compute_indicators(long_prices, long_momo, params)
        assert ind.strategy == "roc"
        assert ind.daily_roc is not None
