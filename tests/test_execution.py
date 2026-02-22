"""Tests for monetary helpers and share calculations."""
from decimal import Decimal

import pytest

from dongpa_engine import money, to_decimal, shares, shares_to_float, money_to_float


class TestToDecimal:
    def test_from_int(self):
        assert to_decimal(42) == Decimal("42")

    def test_from_float(self):
        assert to_decimal(3.14) == Decimal("3.14")

    def test_from_string(self):
        assert to_decimal("99.99") == Decimal("99.99")

    def test_from_none(self):
        assert to_decimal(None) == Decimal("0")

    def test_from_decimal(self):
        d = Decimal("1.23")
        assert to_decimal(d) is d

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            to_decimal(float("nan"))

    def test_inf_raises(self):
        with pytest.raises(ValueError):
            to_decimal(float("inf"))


class TestMoney:
    def test_rounds_to_cents(self):
        assert money(1.234) == Decimal("1.23")
        assert money(1.235) == Decimal("1.24")  # ROUND_HALF_UP
        assert money(1.999) == Decimal("2.00")

    def test_from_string(self):
        assert money("99.999") == Decimal("100.00")


class TestShares:
    def test_integer_mode(self):
        assert shares(3.7, allow_fractional=False) == Decimal("3")
        assert shares(0.9, allow_fractional=False) == Decimal("0")

    def test_fractional_mode(self):
        result = shares(0.123456789, allow_fractional=True)
        assert result == Decimal("0.12345679")  # 8 decimal places


class TestSharesToFloat:
    def test_integer_mode(self):
        assert shares_to_float(Decimal("3"), False) == 3.0

    def test_fractional_mode(self):
        assert shares_to_float(Decimal("0.12345679"), True) == 0.12345679

    def test_none(self):
        assert shares_to_float(None) is None


class TestMoneyToFloat:
    def test_normal(self):
        assert money_to_float(Decimal("99.99")) == 99.99

    def test_none(self):
        assert money_to_float(None) is None
