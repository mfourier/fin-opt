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
    align_index_like,
    set_random_seed,
    summary_metrics,
    fixed_rate_path,
    lognormal_iid,
    PortfolioMetrics,
    compute_metrics,
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

    def test_bootstrap_returns_zero_months(self):
        """Should return empty array for zero months."""
        history = np.array([0.01, 0.02, 0.03])
        result = bootstrap_returns(history, months=0)
        assert len(result) == 0


class TestAlignIndexLike:
    """Test align_index_like function."""

    def test_align_from_datetimeindex(self):
        """Should reuse DatetimeIndex if long enough."""
        idx = pd.date_range("2025-01-01", periods=10, freq="MS")
        result = align_index_like(5, like=idx)

        assert len(result) == 5
        assert isinstance(result, pd.DatetimeIndex)
        assert result[0] == idx[0]

    def test_align_from_series(self):
        """Should extract index from Series."""
        idx = pd.date_range("2025-01-01", periods=10, freq="MS")
        s = pd.Series(range(10), index=idx)
        result = align_index_like(5, like=s)

        assert len(result) == 5
        assert result[0] == idx[0]

    def test_align_from_dataframe(self):
        """Should extract index from DataFrame."""
        idx = pd.date_range("2025-01-01", periods=10, freq="MS")
        df = pd.DataFrame({"a": range(10)}, index=idx)
        result = align_index_like(5, like=df)

        assert len(result) == 5
        assert result[0] == idx[0]

    def test_align_fallback_when_none(self):
        """Should create default index when like is None."""
        result = align_index_like(5, like=None)

        assert len(result) == 5
        assert isinstance(result, pd.DatetimeIndex)

    def test_align_fallback_when_too_short(self):
        """Should create default index when like is too short."""
        idx = pd.date_range("2025-01-01", periods=3, freq="MS")
        result = align_index_like(10, like=idx)

        assert len(result) == 10


class TestSetRandomSeed:
    """Test set_random_seed function."""

    def test_set_seed_reproducibility(self):
        """Setting seed should produce reproducible results."""
        set_random_seed(42)
        a1 = np.random.random(5)

        set_random_seed(42)
        a2 = np.random.random(5)

        np.testing.assert_array_equal(a1, a2)

    def test_set_seed_none_does_nothing(self):
        """None seed should not raise."""
        set_random_seed(None)
        # Just verify no exception


class TestSummaryMetrics:
    """Test summary_metrics function."""

    def test_summary_metrics_basic(self):
        """Should build DataFrame from results dict."""
        # Create mock result with metrics
        class MockMetrics:
            final_wealth = 1_000_000
            total_contributions = 500_000
            cagr = 0.08
            vol = 0.12
            max_drawdown = -0.15

        class MockResult:
            metrics = MockMetrics()

        results = {"scenario1": MockResult()}
        df = summary_metrics(results)

        assert isinstance(df, pd.DataFrame)
        assert "scenario1" in df.index
        assert df.loc["scenario1", "final_wealth"] == 1_000_000
        assert df.loc["scenario1", "cagr"] == 0.08

    def test_summary_metrics_empty(self):
        """Should return empty DataFrame for empty input."""
        df = summary_metrics({})

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "final_wealth" in df.columns

    def test_summary_metrics_skips_no_metrics(self):
        """Should skip objects without metrics attribute."""
        class NoMetrics:
            pass

        results = {"bad": NoMetrics()}
        df = summary_metrics(results)

        assert len(df) == 0

    def test_summary_metrics_multiple_scenarios(self):
        """Should handle multiple scenarios."""
        class MockMetrics:
            def __init__(self, fw):
                self.final_wealth = fw
                self.total_contributions = 0
                self.cagr = 0
                self.vol = 0
                self.max_drawdown = 0

        class MockResult:
            def __init__(self, fw):
                self.metrics = MockMetrics(fw)

        results = {
            "a": MockResult(100),
            "b": MockResult(200),
            "c": MockResult(300),
        }
        df = summary_metrics(results)

        assert len(df) == 3
        assert df.loc["a", "final_wealth"] == 100
        assert df.loc["c", "final_wealth"] == 300


class TestFixedRatePath:
    """Test fixed_rate_path function."""

    def test_fixed_rate_path_basic(self):
        """Should return constant rate array."""
        path = fixed_rate_path(12, 0.01)

        assert len(path) == 12
        assert all(r == 0.01 for r in path)

    def test_fixed_rate_path_zero_months(self):
        """Should return empty array for zero months."""
        path = fixed_rate_path(0, 0.01)

        assert len(path) == 0

    def test_fixed_rate_path_negative_months(self):
        """Should return empty array for negative months."""
        path = fixed_rate_path(-5, 0.01)

        assert len(path) == 0

    def test_fixed_rate_path_zero_rate(self):
        """Should handle zero rate."""
        path = fixed_rate_path(6, 0.0)

        assert len(path) == 6
        assert all(r == 0.0 for r in path)


class TestLognormalIID:
    """Test lognormal_iid function."""

    def test_lognormal_iid_basic(self):
        """Should generate returns from lognormal distribution."""
        returns = lognormal_iid(100, mu=0.0, sigma=0.1, seed=42)

        assert len(returns) == 100
        # All returns should be > -1 (lognormal guarantee)
        assert all(r > -1 for r in returns)

    def test_lognormal_iid_reproducible(self):
        """Same seed should produce same results."""
        r1 = lognormal_iid(10, mu=0.0, sigma=0.1, seed=42)
        r2 = lognormal_iid(10, mu=0.0, sigma=0.1, seed=42)

        np.testing.assert_array_equal(r1, r2)

    def test_lognormal_iid_zero_months(self):
        """Should return empty array for zero months."""
        returns = lognormal_iid(0, mu=0.0, sigma=0.1)

        assert len(returns) == 0

    def test_lognormal_iid_negative_months(self):
        """Should return empty array for negative months."""
        returns = lognormal_iid(-5, mu=0.0, sigma=0.1)

        assert len(returns) == 0

    def test_lognormal_iid_different_seeds(self):
        """Different seeds should produce different results."""
        r1 = lognormal_iid(10, mu=0.0, sigma=0.1, seed=42)
        r2 = lognormal_iid(10, mu=0.0, sigma=0.1, seed=123)

        assert not np.array_equal(r1, r2)


class TestPortfolioMetrics:
    """Test PortfolioMetrics dataclass."""

    def test_portfolio_metrics_creation(self):
        """Should create immutable metrics object."""
        metrics = PortfolioMetrics(
            final_wealth=1_000_000,
            total_contributions=500_000,
            cagr=0.08,
            vol=0.12,
            max_drawdown=-0.15,
        )

        assert metrics.final_wealth == 1_000_000
        assert metrics.total_contributions == 500_000
        assert metrics.cagr == 0.08
        assert metrics.vol == 0.12
        assert metrics.max_drawdown == -0.15

    def test_portfolio_metrics_frozen(self):
        """Should be immutable (frozen dataclass)."""
        metrics = PortfolioMetrics(
            final_wealth=100,
            total_contributions=50,
            cagr=0.05,
            vol=0.10,
            max_drawdown=-0.10,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            metrics.final_wealth = 200


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_compute_metrics_basic(self):
        """Should compute all metrics from wealth series."""
        wealth = pd.Series([100, 105, 110, 108, 115, 120])
        contributions = pd.Series([0, 5, 5, 0, 5, 5])

        metrics = compute_metrics(wealth, contributions=contributions)

        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.final_wealth == 120
        assert metrics.total_contributions == 20
        assert metrics.cagr > 0
        assert metrics.vol > 0
        assert metrics.max_drawdown <= 0

    def test_compute_metrics_empty_wealth(self):
        """Should return zeros for empty wealth."""
        metrics = compute_metrics(pd.Series(dtype=float))

        assert metrics.final_wealth == 0.0
        assert metrics.total_contributions == 0.0
        assert metrics.cagr == 0.0
        assert metrics.vol == 0.0
        assert metrics.max_drawdown == 0.0

    def test_compute_metrics_no_contributions(self):
        """Should handle missing contributions."""
        wealth = pd.Series([100, 110, 120])

        metrics = compute_metrics(wealth)

        assert metrics.final_wealth == 120
        assert metrics.total_contributions == 0.0

    def test_compute_metrics_with_drawdown(self):
        """Should compute max drawdown correctly."""
        # Wealth drops from 110 to 90, then recovers
        wealth = pd.Series([100, 110, 90, 95, 100])

        metrics = compute_metrics(wealth)

        # Max drawdown is (90-110)/110 ≈ -0.182
        assert metrics.max_drawdown == pytest.approx(-0.1818, abs=1e-3)

    def test_compute_metrics_non_series(self):
        """Should return zeros for non-Series input."""
        metrics = compute_metrics([100, 110, 120])  # type: ignore

        assert metrics.final_wealth == 0.0

    def test_compute_metrics_zero_start(self):
        """Should handle zero starting wealth."""
        wealth = pd.Series([0, 50, 100, 150])

        metrics = compute_metrics(wealth)

        assert metrics.final_wealth == 150
        # CAGR uses first positive value as base
        assert metrics.cagr > 0
