"""Tests for mode decision pure functions."""
import pytest

from engines.dongpa_engine import (
    ModeParams,
    StrategyParams,
    eval_rsi_conditions,
)


@pytest.fixture
def rsi_params():
    d = ModeParams(buy_cond_pct=3.0, tp_pct=0.2, max_hold_days=30, slices=7)
    o = ModeParams(buy_cond_pct=5.0, tp_pct=2.5, max_hold_days=7, slices=7)
    return StrategyParams(
        target_ticker="TEST", momentum_ticker="MOMO",
        defense=d, offense=o,
        rsi_high_threshold=65.0,
        rsi_mid_low=40.0,
        rsi_mid_high=60.0,
        rsi_low_threshold=35.0,
        rsi_neutral=50.0,
    )


class TestEvalRsiConditions:
    def test_high_rsi_declining_triggers_defense(self, rsi_params):
        # RSI=70, delta=-2 → defense (high RSI + declining)
        result = eval_rsi_conditions(70.0, 72.0, -2.0, rsi_params)
        assert result == "defense"

    def test_low_rsi_rising_triggers_offense(self, rsi_params):
        # RSI=30, delta=+3 → offense (below low threshold + rising)
        result = eval_rsi_conditions(30.0, 27.0, 3.0, rsi_params)
        assert result == "offense"

    def test_cross_up_50_triggers_offense(self, rsi_params):
        # prev_w=49, rsi=51, delta=+2 → cross_up(49, 51, 50) = True
        result = eval_rsi_conditions(51.0, 49.0, 2.0, rsi_params)
        assert result == "offense"

    def test_cross_down_50_triggers_defense(self, rsi_params):
        # prev_w=51, rsi=49, delta=-2 → cross_down(51, 49, 50) = True
        result = eval_rsi_conditions(49.0, 51.0, -2.0, rsi_params)
        assert result == "defense"

    def test_mid_range_neutral_returns_none(self, rsi_params):
        # RSI=55, delta=0 → no clear signal
        result = eval_rsi_conditions(55.0, 55.0, 0.0, rsi_params)
        assert result is None

    def test_mid_low_declining_triggers_defense(self, rsi_params):
        # RSI=45 (between mid_low=40 and neutral=50), delta=-1
        result = eval_rsi_conditions(45.0, 46.0, -1.0, rsi_params)
        assert result == "defense"

    def test_mid_high_rising_triggers_offense(self, rsi_params):
        # RSI=55 (between neutral=50 and mid_high=60), delta=+1
        result = eval_rsi_conditions(55.0, 54.0, 1.0, rsi_params)
        assert result == "offense"
