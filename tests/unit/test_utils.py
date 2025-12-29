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
        check_non_negative(0, "test")
        check_non_negative(1.5, "test")
        check_non_negative(1000, "test")

    def test_check_non_negative_invalid(self):
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError, match="test must be non-negative"):
            check_non_negative(-0.1, "test")

        with pytest.raises(ValueError, match="test must be non-negative"):
            check_non_negative(-100, "test")


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
        """Should convert dates to first of month."""
        # Already first of month
        d1 = date(2025, 3, 1)
        assert normalize_start_month(d1) == d1

        # Mid-month -> first of month
        d2 = date(2025, 3, 15)
        assert normalize_start_month(d2) == date(2025, 3, 1)

        # End of month -> first of month
        d3 = date(2025, 3, 31)
        assert normalize_start_month(d3) == date(2025, 3, 1)

    def test_month_index_basic(self):
        """Should generate correct sequence of first-of-month dates."""
        start = date(2025, 1, 1)
        months = 3

        index = month_index(months, start)

        assert len(index) == 3
        assert index[0] == date(2025, 1, 1)
        assert index[1] == date(2025, 2, 1)
        assert index[2] == date(2025, 3, 1)

    def test_month_index_year_boundary(self):
        """Should handle year transitions correctly."""
        start = date(2024, 11, 1)
        months = 4

        index = month_index(months, start)

        assert len(index) == 4
        assert index[0] == date(2024, 11, 1)
        assert index[1] == date(2024, 12, 1)
        assert index[2] == date(2025, 1, 1)
        assert index[3] == date(2025, 2, 1)

    def test_month_index_normalizes_start(self):
        """Should normalize start date to first of month."""
        start = date(2025, 3, 15)  # Mid-month
        months = 2

        index = month_index(months, start)

        assert index[0] == date(2025, 3, 1)
        assert index[1] == date(2025, 4, 1)


class TestFormatting:
    """Test currency and number formatting utilities."""

    def test_format_currency_millions(self):
        """Should format large amounts with M suffix."""
        assert format_currency(1_500_000) == "$1.50M CLP"
        assert format_currency(10_000_000) == "$10.00M CLP"
        assert format_currency(123_456_789) == "$123.46M CLP"

    def test_format_currency_thousands(self):
        """Should format smaller amounts without M suffix."""
        assert format_currency(500_000) == "$0.50M CLP"
        assert format_currency(100_000) == "$0.10M CLP"

    def test_format_currency_zero(self):
        """Should handle zero correctly."""
        assert format_currency(0) == "$0.00M CLP"

    def test_format_currency_negative(self):
        """Should handle negative amounts."""
        assert format_currency(-1_000_000) == "-$1.00M CLP"
