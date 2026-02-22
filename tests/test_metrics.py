"""Tests for metrics functions."""
import pandas as pd
import pytest

from engines.dongpa_engine import (
    CAGR,
    max_drawdown,
    summarize,
    compute_buy_and_hold_return,
    compute_equity_return,
    compute_trade_metrics,
    compute_mode_bands,
)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        s = pd.Series([100, 101, 102, 103], index=pd.date_range("2024-01-01", periods=4))
        assert max_drawdown(s) == pytest.approx(0.0)

    def test_drawdown(self):
        s = pd.Series([100, 90, 80, 100], index=pd.date_range("2024-01-01", periods=4))
        assert max_drawdown(s) == pytest.approx(-0.2)  # 80/100 - 1

    def test_full_recovery(self):
        s = pd.Series([100, 50, 100], index=pd.date_range("2024-01-01", periods=3))
        assert max_drawdown(s) == pytest.approx(-0.5)


class TestCAGR:
    def test_positive_return(self):
        dates = pd.date_range("2020-01-01", "2021-01-01", periods=2)
        s = pd.Series([100, 110], index=dates)
        cagr = CAGR(s)
        assert cagr > 0

    def test_zero_start_returns_zero(self):
        dates = pd.date_range("2020-01-01", periods=2)
        s = pd.Series([0, 100], index=dates)
        assert CAGR(s) == 0.0

    def test_empty_returns_zero(self):
        assert CAGR(pd.Series(dtype=float)) == 0.0


class TestSummarize:
    def test_returns_dict(self):
        s = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3))
        result = summarize(s)
        assert "Final Equity" in result
        assert "CAGR" in result
        assert "Max Drawdown" in result
        assert "Sharpe (rf=0)" in result
        assert "Calmar Ratio" in result


class TestComputeBuyAndHoldReturn:
    def test_positive_return(self):
        df = pd.DataFrame({"Close": [100, 110]}, index=pd.date_range("2024-01-01", periods=2))
        assert compute_buy_and_hold_return(df) == pytest.approx(10.0)

    def test_empty_df(self):
        assert compute_buy_and_hold_return(pd.DataFrame()) is None

    def test_zero_start(self):
        df = pd.DataFrame({"Close": [0, 100]}, index=pd.date_range("2024-01-01", periods=2))
        assert compute_buy_and_hold_return(df) is None


class TestComputeEquityReturn:
    def test_positive(self):
        s = pd.Series([100, 120])
        assert compute_equity_return(s) == pytest.approx(20.0)

    def test_empty(self):
        assert compute_equity_return(pd.Series(dtype=float)) is None


class TestComputeTradeMetrics:
    def test_none_input(self):
        assert compute_trade_metrics(None, 10000) is None

    def test_empty_df(self):
        assert compute_trade_metrics(pd.DataFrame(), 10000) is None

    def test_closed_trades(self):
        df = pd.DataFrame({
            "상태": ["완료", "완료"],
            "실현손익": [100.0, -50.0],
            "보유기간(일)": [5, 3],
            "수익률(%)": [2.0, -1.0],
            "청산사유": ["TP", "MOC"],
        })
        result = compute_trade_metrics(df, 10000)
        assert result["trade_count"] == 2
        assert result["moc_count"] == 1
        assert result["net_profit"] == pytest.approx(50.0)
        assert result["avg_hold_days"] == pytest.approx(4.0)


class TestComputeModeBands:
    def test_empty_journal(self):
        result = compute_mode_bands(pd.DataFrame())
        assert result.empty

    def test_single_mode(self):
        df = pd.DataFrame({
            "거래일자": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "모드": ["안전", "안전", "안전"],
        })
        result = compute_mode_bands(df)
        assert len(result) == 1
        assert result.iloc[0]["mode"] == "안전"

    def test_mode_switch(self):
        df = pd.DataFrame({
            "거래일자": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "모드": ["안전", "안전", "공세", "공세"],
        })
        result = compute_mode_bands(df)
        assert len(result) == 2
        assert result.iloc[0]["mode"] == "안전"
        assert result.iloc[1]["mode"] == "공세"
