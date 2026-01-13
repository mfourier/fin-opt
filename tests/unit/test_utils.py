"""
Unit tests for utils.py module.

Tests validation functions, rate conversions, and formatting utilities.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from src.utils import (
    check_non_negative,
    annual_to_monthly,
    monthly_to_annual,
    month_index,
    normalize_start_month,
    format_currency,
    millions_formatter,
    ensure_1d,
    to_series,
    drawdown,
    compute_cagr,
    rescale_returns,
    bootstrap_returns,
)


class TestValidation:
    """Test input validation functions."""

    def test_check_non_negative_valid(self):
        """Valid non-negative values should pass."""
        # Correct signature: check_non_negative(name, value)
        check_non_negative("test", 0)
        check_non_negative("test", 1.5)
        check_non_negative("test", 1000)

    def test_check_non_negative_invalid(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError, match="test must be non-negative"):
            check_non_negative("test", -0.1)

        with pytest.raises(ValueError, match="other must be non-negative"):
            check_non_negative("other", -100)


class TestRateConversion:
    """Test annual<->monthly rate conversion functions."""

    def test_annual_to_monthly_zero(self):
        """Zero annual rate should give zero monthly rate."""
        monthly = annual_to_monthly(0.0)
        assert monthly == pytest.approx(0.0, abs=1e-10)

    def test_annual_to_monthly_positive(self):
        """Positive annual rates should convert correctly."""
        # 4% annual ≈ 0.327% monthly
        monthly = annual_to_monthly(0.04)
        assert monthly == pytest.approx(0.00327, abs=1e-4)

        # 12% annual ≈ 0.949% monthly
        monthly = annual_to_monthly(0.12)
        assert monthly == pytest.approx(0.00949, abs=1e-4)

    def test_monthly_to_annual_zero(self):
        """Zero monthly rate should give zero annual rate."""
        annual = monthly_to_annual(0.0)
        assert annual == pytest.approx(0.0, abs=1e-10)

    def test_monthly_to_annual_positive(self):
        """Positive monthly rates should convert correctly."""
        # 0.5% monthly ≈ 6.17% annual
        annual = monthly_to_annual(0.005)
        assert annual == pytest.approx(0.0617, abs=1e-4)

    def test_roundtrip_conversion(self):
        """Annual -> Monthly -> Annual should be identity."""
        test_rates = [0.0, 0.03, 0.08, 0.15]

        for rate in test_rates:
            monthly = annual_to_monthly(rate)
            recovered = monthly_to_annual(monthly)
            assert recovered == pytest.approx(rate, abs=1e-10)


class TestNormalizeStartMonth:
    """Test normalize_start_month function."""

    def test_normalize_from_date(self):
        """Should extract month offset from date."""
        # January -> 0
        assert normalize_start_month(date(2025, 1, 1)) == 0
        # March -> 2
        assert normalize_start_month(date(2025, 3, 15)) == 2
        # December -> 11
        assert normalize_start_month(date(2025, 12, 31)) == 11

    def test_normalize_from_int(self):
        """Should convert 1-12 to 0-11 offset."""
        assert normalize_start_month(1) == 0
        assert normalize_start_month(6) == 5
        assert normalize_start_month(12) == 11

    def test_normalize_none(self):
        """None should default to 0 (January)."""
        assert normalize_start_month(None) == 0

    def test_normalize_invalid_month(self):
        """Invalid month numbers should raise."""
        with pytest.raises(ValueError):
            normalize_start_month(0)
        with pytest.raises(ValueError):
            normalize_start_month(13)


class TestMonthIndex:
    """Test month_index function."""

    def test_month_index_basic(self):
        """Should generate correct DatetimeIndex."""
        start = date(2025, 1, 1)
        index = month_index(start=start, months=3)

        assert len(index) == 3
        assert isinstance(index, pd.DatetimeIndex)
        assert index[0].month == 1
        assert index[1].month == 2
        assert index[2].month == 3

    def test_month_index_year_boundary(self):
        """Should handle year transitions correctly."""
        start = date(2024, 11, 1)
        index = month_index(start=start, months=4)

        assert len(index) == 4
        assert index[0].month == 11 and index[0].year == 2024
        assert index[1].month == 12 and index[1].year == 2024
        assert index[2].month == 1 and index[2].year == 2025
        assert index[3].month == 2 and index[3].year == 2025

    def test_month_index_empty(self):
        """months=0 should return empty index."""
        index = month_index(start=date(2025, 1, 1), months=0)
        assert len(index) == 0

    def test_month_index_none_start(self):
        """None start should use today's month."""
        index = month_index(start=None, months=3)
        assert len(index) == 3


class TestFormatting:
    """Test currency and number formatting utilities."""

    def test_format_currency_default(self):
        """Should format with default parameters."""
        # Default: 1 decimal, $ symbol, M unit
        result = format_currency(1_500_000)
        assert result == "$1.5M"

    def test_format_currency_zero_decimals(self):
        """Should handle zero decimals."""
        result = format_currency(10_000_000, decimals=0)
        assert result == "$10M"

    def test_format_currency_custom_decimals(self):
        """Should handle custom decimals."""
        result = format_currency(5_500_000, decimals=2)
        assert result == "$5.50M"

    def test_format_currency_zero(self):
        """Should handle zero correctly."""
        result = format_currency(0)
        assert result == "$0.0M"

    def test_millions_formatter(self):
        """Should format values for matplotlib axis."""
        assert millions_formatter(0, 0) == "0"
        assert millions_formatter(25_000_000, 0) == "25M"
        assert millions_formatter(12_500_000, 0) == "12.5M"


class TestArrayHelpers:
    """Test array manipulation functions."""

    def test_ensure_1d_from_list(self):
        """Should convert list to 1D array."""
        arr = ensure_1d([1.0, 2.0, 3.0], name="test")
        assert arr.shape == (3,)
        assert arr.dtype == float

    def test_ensure_1d_from_array(self):
        """Should accept numpy array."""
        arr = ensure_1d(np.array([1, 2, 3]), name="test")
        assert arr.shape == (3,)

    def test_ensure_1d_invalid_shape(self):
        """Should reject 2D arrays."""
        with pytest.raises(ValueError, match="must be 1-D"):
            ensure_1d(np.array([[1, 2], [3, 4]]), name="test")

    def test_ensure_1d_non_finite(self):
        """Should reject non-finite values."""
        with pytest.raises(ValueError, match="finite values"):
            ensure_1d([1.0, np.nan, 3.0], name="test")

    def test_to_series_with_index(self):
        """Should create Series with index."""
        idx = pd.date_range("2025-01-01", periods=3, freq="MS")
        s = to_series([1.0, 2.0, 3.0], index=idx, name="values")

        assert isinstance(s, pd.Series)
        assert len(s) == 3
        assert s.name == "values"


class TestFinanceHelpers:
    """Test financial calculation functions."""

    def test_drawdown_basic(self):
        """Should compute drawdown from peak."""
        prices = pd.Series([100, 110, 90, 95, 80])
        dd = drawdown(prices)

        # Peak at 110, drawdown at 90 is (90-110)/110 ≈ -0.182
        assert dd.iloc[2] == pytest.approx(-0.1818, abs=1e-3)

    def test_drawdown_empty(self):
        """Should handle empty series."""
        dd = drawdown(pd.Series(dtype=float))
        assert len(dd) == 0

    def test_compute_cagr_basic(self):
        """Should compute CAGR correctly."""
        # 100 -> 200 over 12 months = 100% annual
        wealth = pd.Series([100] + [100] * 10 + [200])
        cagr = compute_cagr(wealth, periods_per_year=12)
        assert cagr == pytest.approx(1.0, abs=0.01)

    def test_compute_cagr_zero_start(self):
        """Should handle zero starting value."""
        wealth = pd.Series([0, 50, 100, 150])
        cagr = compute_cagr(wealth)
        # Should use first positive value (50) as base
        assert cagr > 0


class TestScenarioHelpers:
    """Test scenario/randomness helpers."""

    def test_rescale_returns(self):
        """Should rescale to target mean and vol."""
        original = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
        rescaled = rescale_returns(original, target_mean=0.05, target_vol=0.10)

        assert np.mean(rescaled) == pytest.approx(0.05, abs=1e-6)
        assert np.std(rescaled, ddof=1) == pytest.approx(0.10, abs=1e-6)

    def test_bootstrap_returns(self):
        """Should sample from history."""
        history = np.array([0.01, 0.02, 0.03, 0.04])
        sampled = bootstrap_returns(history, months=10, seed=42)

        assert len(sampled) == 10
        # All values should be from original history
        assert all(v in history for v in sampled)

    def test_bootstrap_returns_empty(self):
        """Should handle empty history."""
        result = bootstrap_returns(np.array([]), months=5)
        assert len(result) == 0
