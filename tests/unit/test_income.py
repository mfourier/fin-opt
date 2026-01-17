"""
Unit tests for income.py module.

Tests FixedIncome, VariableIncome, and IncomeModel classes.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from src.income import (
    FixedIncome,
    VariableIncome,
    IncomeModel,
    IncomeMetrics,
)


# ============================================================================
# FIXEDINCOME TESTS
# ============================================================================

class TestFixedIncomeInstantiation:
    """Test FixedIncome initialization."""

    def test_basic_instantiation(self):
        """Test basic FixedIncome creation."""
        fi = FixedIncome(base=1_500_000)
        assert fi.base == 1_500_000
        assert fi.annual_growth == 0.0
        assert fi.name == "fixed"

    def test_with_growth(self):
        """Test FixedIncome with annual growth."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.05)
        assert fi.annual_growth == 0.05

    def test_with_salary_raises(self):
        """Test FixedIncome with salary raises."""
        raises = {date(2025, 7, 1): 200_000}
        fi = FixedIncome(base=1_000_000, salary_raises=raises)
        assert fi.salary_raises == raises

    def test_negative_base_raises(self):
        """Test that negative base raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            FixedIncome(base=-100)

    def test_frozen_dataclass(self):
        """Test that FixedIncome is immutable."""
        fi = FixedIncome(base=1_000_000)
        with pytest.raises(Exception):
            fi.base = 2_000_000


class TestFixedIncomeProject:
    """Test FixedIncome.project() method."""

    def test_project_array_output(self):
        """Test projection with array output."""
        fi = FixedIncome(base=1_000_000)
        result = fi.project(months=6, output="array")

        assert isinstance(result, np.ndarray)
        assert result.shape == (6,)
        assert np.allclose(result, 1_000_000)

    def test_project_series_output(self):
        """Test projection with Series output."""
        fi = FixedIncome(base=1_000_000)
        result = fi.project(months=6, start=date(2025, 1, 1), output="series")

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        assert result.name == "fixed"

    def test_project_with_growth(self):
        """Test projection applies growth correctly."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.12)
        result = fi.project(months=12, output="array")

        # Final value should be ~12% higher than initial
        assert result[-1] > result[0]
        assert result[-1] / result[0] == pytest.approx(1.12, rel=0.01)

    def test_project_with_n_sims(self):
        """Test projection with n_sims parameter."""
        fi = FixedIncome(base=1_000_000)
        result = fi.project(months=6, n_sims=10, output="array")

        assert result.shape == (10, 6)
        # All rows should be identical (deterministic)
        np.testing.assert_array_equal(result[0, :], result[1, :])

    def test_project_zero_months(self):
        """Test projection with zero months."""
        fi = FixedIncome(base=1_000_000)
        result = fi.project(months=0, output="array")

        assert result.shape == (0,)

    def test_project_with_salary_raises(self):
        """Test projection with salary raises."""
        raises = {date(2025, 7, 1): 200_000}
        fi = FixedIncome(base=1_000_000, salary_raises=raises)
        result = fi.project(months=12, start=date(2025, 1, 1), output="array")

        # After raise (month 6+), income should increase
        assert result[6] > result[0]

    def test_project_invalid_output_raises(self):
        """Test that invalid output raises ValueError."""
        fi = FixedIncome(base=1_000_000)
        with pytest.raises(ValueError, match="output must be"):
            fi.project(months=6, output="invalid")


# ============================================================================
# VARIABLEINCOME TESTS
# ============================================================================

class TestVariableIncomeInstantiation:
    """Test VariableIncome initialization."""

    def test_basic_instantiation(self):
        """Test basic VariableIncome creation."""
        vi = VariableIncome(base=200_000)
        assert vi.base == 200_000
        assert vi.sigma == 0.0
        assert vi.name == "variable"

    def test_with_sigma(self):
        """Test VariableIncome with volatility."""
        vi = VariableIncome(base=200_000, sigma=0.15)
        assert vi.sigma == 0.15

    def test_with_seasonality(self):
        """Test VariableIncome with seasonality."""
        seasonality = [1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.0, 0.8, 0.9, 1.0, 1.05, 1.15]
        vi = VariableIncome(base=200_000, seasonality=seasonality)
        assert vi.seasonality is not None

    def test_invalid_seasonality_length(self):
        """Test that seasonality must have 12 elements."""
        with pytest.raises(ValueError, match="length 12"):
            VariableIncome(base=200_000, seasonality=[1.0] * 10)

    def test_floor_greater_than_cap_raises(self):
        """Test that floor > cap raises ValueError."""
        with pytest.raises(ValueError, match="floor cannot be greater than cap"):
            VariableIncome(base=200_000, floor=500_000, cap=100_000)

    def test_negative_sigma_raises(self):
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            VariableIncome(base=200_000, sigma=-0.1)


class TestVariableIncomeProject:
    """Test VariableIncome.project() method."""

    def test_project_deterministic(self):
        """Test deterministic projection (sigma=0)."""
        vi = VariableIncome(base=200_000, sigma=0.0)
        result = vi.project(months=6, output="array")

        assert result.shape == (6,)
        np.testing.assert_array_almost_equal(result, 200_000)

    def test_project_stochastic(self):
        """Test stochastic projection (sigma>0)."""
        vi = VariableIncome(base=200_000, sigma=0.15, seed=42)
        result = vi.project(months=6, output="array")

        assert result.shape == (6,)
        # Should have variation
        assert not np.allclose(result, 200_000)

    def test_project_with_floor(self):
        """Test projection respects floor."""
        vi = VariableIncome(base=200_000, sigma=1.0, floor=100_000, seed=42)
        result = vi.project(months=100, output="array")

        assert np.all(result >= 100_000)

    def test_project_with_cap(self):
        """Test projection respects cap."""
        vi = VariableIncome(base=200_000, sigma=1.0, cap=300_000, seed=42)
        result = vi.project(months=100, output="array")

        assert np.all(result <= 300_000)

    def test_project_n_sims(self):
        """Test projection with n_sims parameter."""
        vi = VariableIncome(base=200_000, sigma=0.15, seed=42)
        result = vi.project(months=6, n_sims=10, output="array")

        assert result.shape == (10, 6)
        # Different simulations should be different
        assert not np.allclose(result[0, :], result[1, :])

    def test_project_reproducibility(self):
        """Test that same seed produces same results."""
        vi = VariableIncome(base=200_000, sigma=0.15, seed=42)
        result1 = vi.project(months=6, output="array")
        result2 = vi.project(months=6, output="array")

        np.testing.assert_array_equal(result1, result2)

    def test_project_seasonality_rotation(self):
        """Test seasonality rotation based on start month."""
        seasonality = [1.0, 0.8, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        vi = VariableIncome(base=100_000, sigma=0.0, seasonality=seasonality)

        result = vi.project(months=3, start=date(2025, 1, 1), output="array")
        expected = np.array([1.0, 0.8, 1.2]) * 100_000

        np.testing.assert_array_almost_equal(result, expected)


# ============================================================================
# INCOMEMODEL TESTS
# ============================================================================

class TestIncomeModelInstantiation:
    """Test IncomeModel initialization."""

    def test_basic_instantiation(self):
        """Test basic IncomeModel creation."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000)
        model = IncomeModel(fixed=fi, variable=vi)

        assert model.fixed is fi
        assert model.variable is vi

    def test_with_custom_names(self):
        """Test IncomeModel with custom stream names."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000)
        model = IncomeModel(fixed=fi, variable=vi, name_fixed="salary", name_variable="bonus")

        assert model.name_fixed == "salary"
        assert model.name_variable == "bonus"


class TestIncomeModelProject:
    """Test IncomeModel.project() method."""

    def test_project_series(self):
        """Test project returning Series."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=fi, variable=vi)

        result = model.project(months=6, start=date(2025, 1, 1), output="series")

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        # Total should be 1.2M (fixed + variable)
        np.testing.assert_array_almost_equal(result, 1_200_000)

    def test_project_dataframe(self):
        """Test project returning DataFrame."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=fi, variable=vi)

        result = model.project(months=6, start=date(2025, 1, 1), output="dataframe")

        assert isinstance(result, pd.DataFrame)
        assert "fixed" in result.columns
        assert "variable" in result.columns
        assert "total" in result.columns
        # Verify total = fixed + variable
        np.testing.assert_array_almost_equal(result["total"], result["fixed"] + result["variable"])

    def test_project_array(self):
        """Test project returning array dict."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=fi, variable=vi)

        result = model.project(months=6, start=date(2025, 1, 1), output="array")

        assert isinstance(result, dict)
        assert "fixed" in result
        assert "variable" in result
        assert "total" in result
        assert result["fixed"].shape == (6,)

    def test_project_n_sims(self):
        """Test project with n_sims parameter."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000, sigma=0.1, seed=42)
        model = IncomeModel(fixed=fi, variable=vi)

        result = model.project(months=6, n_sims=10, output="array")

        assert result["fixed"].shape == (10, 6)
        assert result["variable"].shape == (10, 6)
        assert result["total"].shape == (10, 6)


class TestIncomeModelContributions:
    """Test IncomeModel.contributions() method."""

    def test_contributions_default(self):
        """Test contributions with default fractions."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=fi, variable=vi)

        result = model.contributions(months=6, start=date(2025, 1, 1), output="array")

        # Default: 30% of fixed + 100% of variable = 500k
        expected = 1_000_000 * 0.3 + 200_000 * 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_contributions_custom_fractions(self):
        """Test contributions with custom fractions."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=fi, variable=vi)
        model.monthly_contribution = {"fixed": [0.5]*12, "variable": [0.5]*12}

        result = model.contributions(months=6, start=date(2025, 1, 1), output="array")

        # 50% of 1M + 50% of 200k = 600k
        expected = 500_000 + 100_000
        np.testing.assert_array_almost_equal(result, expected)

    def test_contributions_series(self):
        """Test contributions as Series."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000)
        model = IncomeModel(fixed=fi, variable=vi)

        result = model.contributions(months=6, start=date(2025, 1, 1), output="series")

        assert isinstance(result, pd.Series)
        assert result.name == "contribution"

    def test_contributions_non_negative(self):
        """Test that contributions are non-negative."""
        fi = FixedIncome(base=100_000)
        vi = VariableIncome(base=200_000, sigma=1.0, seed=42)
        model = IncomeModel(fixed=fi, variable=vi)
        model.monthly_contribution = {"fixed": [-0.5]*12, "variable": [-0.5]*12}

        result = model.contributions(months=50, start=date(2025, 1, 1), output="array")

        assert np.all(result >= 0)


class TestIncomeModelMetrics:
    """Test IncomeModel metrics methods."""

    def test_income_metrics(self):
        """Test income_metrics returns IncomeMetrics."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.12)
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=fi, variable=vi)

        metrics = model.income_metrics(months=12, start=date(2025, 1, 1))

        assert isinstance(metrics, IncomeMetrics)
        assert metrics.months == 12
        assert metrics.total_income > 0
        assert metrics.total_fixed > 0
        assert metrics.total_variable > 0

    def test_summary(self):
        """Test summary returns Series."""
        fi = FixedIncome(base=1_000_000)
        vi = VariableIncome(base=200_000, sigma=0.1, seed=42)
        model = IncomeModel(fixed=fi, variable=vi)

        summary = model.summary(months=12, start=date(2025, 1, 1))

        assert isinstance(summary, pd.Series)
        assert "total_income" in summary.index
        assert "fixed_share" in summary.index


class TestIncomeModelSerialization:
    """Test IncomeModel serialization."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.05)
        vi = VariableIncome(base=200_000, sigma=0.1, seed=42)
        model = IncomeModel(fixed=fi, variable=vi)

        result = model.to_dict()

        assert "fixed" in result
        assert "variable" in result
        assert result["fixed"]["base"] == 1_000_000
        assert result["variable"]["sigma"] == 0.1

    def test_from_dict(self):
        """Test from_dict deserialization."""
        payload = {
            "fixed": {"base": 1_000_000, "annual_growth": 0.05},
            "variable": {"base": 200_000, "sigma": 0.1},
        }

        model = IncomeModel.from_dict(payload)

        assert model.fixed.base == 1_000_000
        assert model.variable.sigma == 0.1


class TestIncomeModelIntegration:
    """Integration tests for IncomeModel."""

    def test_full_workflow(self):
        """Test complete workflow."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.04)
        vi = VariableIncome(
            base=200_000,
            sigma=0.1,
            seasonality=[0.8]*3 + [1.0]*3 + [1.1]*3 + [1.2]*3,
            seed=42
        )
        model = IncomeModel(fixed=fi, variable=vi)

        # Project
        income_df = model.project(months=24, start=date(2025, 1, 1), output="dataframe")
        assert len(income_df) == 24

        # Contributions
        contrib = model.contributions(months=24, start=date(2025, 1, 1), output="array")
        assert contrib.shape == (24,)

        # Metrics
        metrics = model.income_metrics(months=24, start=date(2025, 1, 1))
        assert metrics.months == 24

    def test_multi_simulation_workflow(self):
        """Test workflow with multiple simulations."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.04)
        vi = VariableIncome(base=200_000, sigma=0.1, seed=42)
        model = IncomeModel(fixed=fi, variable=vi)

        income = model.project(months=12, start=date(2025, 1, 1), output="array", n_sims=100)
        assert income["variable"].shape == (100, 12)

        contrib = model.contributions(months=12, start=date(2025, 1, 1), output="array", n_sims=100)
        assert contrib.shape == (100, 12)


# ============================================================================
# OPTIONAL COMPONENT TESTS (fixed=None or variable=None)
# ============================================================================

class TestIncomeModelOptionalComponents:
    """Test IncomeModel with optional fixed or variable components."""

    def test_both_none_raises(self):
        """Test that both None raises ValueError."""
        with pytest.raises(ValueError, match="At least one income stream"):
            IncomeModel(fixed=None, variable=None)

    def test_fixed_only_instantiation(self):
        """Test IncomeModel with only fixed income."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.05)
        model = IncomeModel(fixed=fi, variable=None)

        assert model.fixed is fi
        assert model.variable is None

    def test_variable_only_instantiation(self):
        """Test IncomeModel with only variable income."""
        vi = VariableIncome(base=200_000, sigma=0.1, seed=42)
        model = IncomeModel(fixed=None, variable=vi)

        assert model.fixed is None
        assert model.variable is vi

    def test_fixed_only_project_series(self):
        """Test project() with fixed-only model returns correct values."""
        fi = FixedIncome(base=1_000_000)
        model = IncomeModel(fixed=fi, variable=None)

        result = model.project(months=6, start=date(2025, 1, 1), output="series")

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        # Total should equal fixed (no variable)
        np.testing.assert_array_almost_equal(result, 1_000_000)

    def test_variable_only_project_series(self):
        """Test project() with variable-only model returns correct values."""
        vi = VariableIncome(base=200_000, sigma=0.0)  # deterministic
        model = IncomeModel(fixed=None, variable=vi)

        result = model.project(months=6, start=date(2025, 1, 1), output="series")

        assert isinstance(result, pd.Series)
        assert len(result) == 6
        # Total should equal variable (no fixed)
        np.testing.assert_array_almost_equal(result, 200_000)

    def test_fixed_only_project_dataframe(self):
        """Test project() DataFrame with fixed-only model."""
        fi = FixedIncome(base=1_000_000)
        model = IncomeModel(fixed=fi, variable=None)

        result = model.project(months=6, start=date(2025, 1, 1), output="dataframe")

        assert isinstance(result, pd.DataFrame)
        assert "fixed" in result.columns
        assert "variable" in result.columns
        assert "total" in result.columns
        # Fixed should be 1M, variable should be 0
        np.testing.assert_array_almost_equal(result["fixed"], 1_000_000)
        np.testing.assert_array_almost_equal(result["variable"], 0)
        np.testing.assert_array_almost_equal(result["total"], 1_000_000)

    def test_variable_only_project_dataframe(self):
        """Test project() DataFrame with variable-only model."""
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=None, variable=vi)

        result = model.project(months=6, start=date(2025, 1, 1), output="dataframe")

        # Fixed should be 0, variable should be 200k
        np.testing.assert_array_almost_equal(result["fixed"], 0)
        np.testing.assert_array_almost_equal(result["variable"], 200_000)
        np.testing.assert_array_almost_equal(result["total"], 200_000)

    def test_fixed_only_project_n_sims(self):
        """Test project() with n_sims on fixed-only model."""
        fi = FixedIncome(base=1_000_000)
        model = IncomeModel(fixed=fi, variable=None)

        result = model.project(months=6, n_sims=10, output="array")

        assert result["fixed"].shape == (10, 6)
        assert result["variable"].shape == (10, 6)
        assert result["total"].shape == (10, 6)
        # All variable values should be zero
        np.testing.assert_array_equal(result["variable"], 0)

    def test_variable_only_project_n_sims(self):
        """Test project() with n_sims on variable-only model."""
        vi = VariableIncome(base=200_000, sigma=0.1, seed=42)
        model = IncomeModel(fixed=None, variable=vi)

        result = model.project(months=6, n_sims=10, output="array")

        assert result["fixed"].shape == (10, 6)
        # All fixed values should be zero
        np.testing.assert_array_equal(result["fixed"], 0)
        # Variable should have variation
        assert not np.allclose(result["variable"][0, :], result["variable"][1, :])

    def test_fixed_only_contributions(self):
        """Test contributions() with fixed-only model."""
        fi = FixedIncome(base=1_000_000)
        model = IncomeModel(fixed=fi, variable=None)

        result = model.contributions(months=6, start=date(2025, 1, 1), output="array")

        # Default: 30% of fixed, 100% of variable (which is 0)
        expected = 1_000_000 * 0.3
        np.testing.assert_array_almost_equal(result, expected)

    def test_variable_only_contributions(self):
        """Test contributions() with variable-only model."""
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=None, variable=vi)

        result = model.contributions(months=6, start=date(2025, 1, 1), output="array")

        # Default: 30% of fixed (0), 100% of variable
        expected = 200_000 * 1.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_fixed_only_income_metrics(self):
        """Test income_metrics() with fixed-only model."""
        fi = FixedIncome(base=1_000_000)
        model = IncomeModel(fixed=fi, variable=None)

        metrics = model.income_metrics(months=12, start=date(2025, 1, 1))

        assert metrics.months == 12
        assert metrics.total_fixed == 12_000_000
        assert metrics.total_variable == 0
        assert metrics.total_income == 12_000_000
        assert metrics.fixed_share == 1.0
        assert metrics.variable_share == 0.0

    def test_variable_only_income_metrics(self):
        """Test income_metrics() with variable-only model."""
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=None, variable=vi)

        metrics = model.income_metrics(months=12, start=date(2025, 1, 1))

        assert metrics.months == 12
        assert metrics.total_fixed == 0
        assert metrics.total_variable == 2_400_000
        assert metrics.total_income == 2_400_000
        assert metrics.fixed_share == 0.0
        assert metrics.variable_share == 1.0

    def test_fixed_only_to_dict(self):
        """Test to_dict() with fixed-only model."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.05)
        model = IncomeModel(fixed=fi, variable=None)

        result = model.to_dict()

        assert result["fixed"] is not None
        assert result["fixed"]["base"] == 1_000_000
        assert result["variable"] is None

    def test_variable_only_to_dict(self):
        """Test to_dict() with variable-only model."""
        vi = VariableIncome(base=200_000, sigma=0.1)
        model = IncomeModel(fixed=None, variable=vi)

        result = model.to_dict()

        assert result["fixed"] is None
        assert result["variable"] is not None
        assert result["variable"]["base"] == 200_000

    def test_fixed_only_from_dict(self):
        """Test from_dict() with fixed-only payload."""
        payload = {
            "fixed": {"base": 1_000_000, "annual_growth": 0.05},
            "variable": None,
        }

        model = IncomeModel.from_dict(payload)

        assert model.fixed is not None
        assert model.fixed.base == 1_000_000
        assert model.variable is None

    def test_variable_only_from_dict(self):
        """Test from_dict() with variable-only payload."""
        payload = {
            "fixed": None,
            "variable": {"base": 200_000, "sigma": 0.1},
        }

        model = IncomeModel.from_dict(payload)

        assert model.fixed is None
        assert model.variable is not None
        assert model.variable.base == 200_000

    def test_fixed_only_repr(self):
        """Test __repr__() with fixed-only model."""
        fi = FixedIncome(base=1_000_000)
        model = IncomeModel(fixed=fi, variable=None)

        repr_str = repr(model)

        assert "IncomeModel" in repr_str
        assert "fixed" in repr_str
        assert "variable" not in repr_str or "variable(base" not in repr_str

    def test_variable_only_repr(self):
        """Test __repr__() with variable-only model."""
        vi = VariableIncome(base=200_000, sigma=0.0)
        model = IncomeModel(fixed=None, variable=vi)

        repr_str = repr(model)

        assert "IncomeModel" in repr_str
        assert "variable" in repr_str

    def test_roundtrip_serialization_fixed_only(self):
        """Test serialization roundtrip for fixed-only model."""
        fi = FixedIncome(base=1_000_000, annual_growth=0.05)
        original = IncomeModel(fixed=fi, variable=None)

        payload = original.to_dict()
        restored = IncomeModel.from_dict(payload)

        assert restored.fixed.base == original.fixed.base
        assert restored.fixed.annual_growth == original.fixed.annual_growth
        assert restored.variable is None

    def test_roundtrip_serialization_variable_only(self):
        """Test serialization roundtrip for variable-only model."""
        vi = VariableIncome(base=200_000, sigma=0.15, seed=42)
        original = IncomeModel(fixed=None, variable=vi)

        payload = original.to_dict()
        restored = IncomeModel.from_dict(payload)

        assert restored.fixed is None
        assert restored.variable.base == original.variable.base
        assert restored.variable.sigma == original.variable.sigma


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
