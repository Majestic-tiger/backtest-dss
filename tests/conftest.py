"""Shared fixtures for dongpa tests."""
import pandas as pd
import numpy as np
import pytest

from dongpa_engine import ModeParams, CapitalParams, StrategyParams


@pytest.fixture
def default_defense():
    return ModeParams(buy_cond_pct=3.0, tp_pct=0.2, max_hold_days=30, slices=7)


@pytest.fixture
def default_offense():
    return ModeParams(buy_cond_pct=5.0, tp_pct=2.5, max_hold_days=7, slices=7)


@pytest.fixture
def default_params(default_defense, default_offense):
    return StrategyParams(
        target_ticker="TEST",
        momentum_ticker="MOMO",
        defense=default_defense,
        offense=default_offense,
    )


@pytest.fixture
def default_cap():
    return CapitalParams(initial_cash=10000.0)


def _make_price_df(prices: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    """Create a simple OHLCV DataFrame from a list of close prices."""
    dates = pd.bdate_range(start=start, periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [1000000] * len(prices),
        },
        index=dates,
    )


@pytest.fixture
def sample_prices():
    """20 trading days of synthetic price data."""
    np.random.seed(42)
    base = 100.0
    returns = np.random.normal(0.001, 0.02, 20)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return _make_price_df(prices[:20])


@pytest.fixture
def sample_momo():
    """20 trading days of momentum ticker data (for RSI)."""
    np.random.seed(99)
    base = 400.0
    returns = np.random.normal(0.0005, 0.01, 20)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return _make_price_df(prices[:20])
