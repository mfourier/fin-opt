"""
Unit tests for utils.py module.

Tests validation functions, rate conversions, and formatting utilities.
"""

import pytest
import numpy as np
from datetime import date

from src.utils import (
    check_non_negative,
    annual_to_monthly,
    monthly_to_annual,
    month_index,
    normalize_start_month,
    format_currency,
)


class TestValidation:
    """Test input validation functions."""

    def test_check_non_negative_valid(self):
        """Valid non-negative values should pass."""
        # API: check_non_negative(name, value)
        check_non_negative("test", 0)
        check_non_negative("test", 1.5)
        check_non_negative("test", 1000)

    def test_check_non_negative_invalid(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError, match="test must be non-negative"):
            check_non_negative("test", -0.1)

        with pytest.raises(ValueError, match="test must be non-negative"):
            check_non_negative("test", -100)


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
        assert monthly == pytest.approx(0.00327, abs=1e-5)

        # 12% annual ≈ 0.949% monthly
        monthly = annual_to_monthly(0.12)
        assert monthly == pytest.approx(0.00949, abs=1e-5)

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


class TestDateUtilities:
    """Test date handling and month index generation."""

    def test_normalize_start_month(self):
        """Should convert dates to month offset (0-11)."""
        # API: normalize_start_month returns int offset (Jan=0, Dec=11)
        # March -> 2
        d1 = date(2025, 3, 1)
        assert normalize_start_month(d1) == 2

        # Mid-month same result
        d2 = date(2025, 3, 15)
        assert normalize_start_month(d2) == 2

        # January -> 0
        d3 = date(2025, 1, 31)
        assert normalize_start_month(d3) == 0

        # December -> 11
        d4 = date(2025, 12, 1)
        assert normalize_start_month(d4) == 11

    def test_month_index_basic(self):
        """Should generate correct sequence of first-of-month dates."""
        # API: month_index(start, months)
        start = date(2025, 1, 1)
        months = 3

        index = month_index(start, months)

        assert len(index) == 3
        # Returns DatetimeIndex, compare with Timestamp
        import pandas as pd
        assert index[0] == pd.Timestamp("2025-01-01")
        assert index[1] == pd.Timestamp("2025-02-01")
        assert index[2] == pd.Timestamp("2025-03-01")

    def test_month_index_year_boundary(self):
        """Should handle year transitions correctly."""
        start = date(2024, 11, 1)
        months = 4

        index = month_index(start, months)

        import pandas as pd
        assert len(index) == 4
        assert index[0] == pd.Timestamp("2024-11-01")
        assert index[1] == pd.Timestamp("2024-12-01")
        assert index[2] == pd.Timestamp("2025-01-01")
        assert index[3] == pd.Timestamp("2025-02-01")

    def test_month_index_normalizes_start(self):
        """Should normalize start date to first of month."""
        start = date(2025, 3, 15)  # Mid-month
        months = 2

        index = month_index(start, months)

        import pandas as pd
        assert index[0] == pd.Timestamp("2025-03-01")
        assert index[1] == pd.Timestamp("2025-04-01")


class TestFormatting:
    """Test currency and number formatting utilities."""

    def test_format_currency_millions(self):
        """Should format large amounts with M suffix."""
        # API: format_currency(value, decimals=1, symbol='$', unit='M')
        # Default: 1 decimal place, no CLP suffix
        assert format_currency(1_500_000) == "$1.5M"
        assert format_currency(10_000_000) == "$10.0M"
        assert format_currency(123_456_789) == "$123.5M"

    def test_format_currency_thousands(self):
        """Should format smaller amounts without M suffix."""
        assert format_currency(500_000) == "$0.5M"
        assert format_currency(100_000) == "$0.1M"

    def test_format_currency_zero(self):
        """Should handle zero correctly."""
        assert format_currency(0) == "$0.0M"

    def test_format_currency_negative(self):
        """Should handle negative amounts."""
        assert format_currency(-1_000_000) == "$-1.0M"
