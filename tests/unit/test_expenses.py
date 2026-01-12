"""
Unit tests for expenses.py module.

Tests validation, projections, and statistical properties of expense classes.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date
from dataclasses import FrozenInstanceError

from src.expenses import (
    FixedExpense,
    VariableExpense,
    MicroExpense,
    ExpenseModel,
    ExpenseMetrics,
)


class TestFixedExpense:
    """Tests for FixedExpense class."""

    def test_basic_projection(self):
        """Test basic deterministic projection."""
        fe = FixedExpense(base=500_000)
        arr = fe.project(12)

        assert arr.shape == (12,)
        assert np.allclose(arr, 500_000)  # No inflation

    def test_deterministic_across_sims(self):
        """Fixed expense is identical across simulations."""
        fe = FixedExpense(base=500_000, annual_inflation=0.04)
        arr = fe.project(12, n_sims=100)

        assert arr.shape == (100, 12)
        # All simulations should be identical
        assert np.allclose(arr[0], arr[50])
        assert np.allclose(arr[0], arr[99])

    def test_inflation_compounding(self):
        """Monthly compounding of inflation."""
        fe = FixedExpense(base=100_000, annual_inflation=0.12)
        arr = fe.project(12)

        # After 12 months, should be ~1.12x base
        assert arr[-1] / arr[0] == pytest.approx(1.12, rel=0.01)

    def test_zero_inflation(self):
        """Zero inflation produces constant projection."""
        fe = FixedExpense(base=500_000, annual_inflation=0.0)
        arr = fe.project(24)

        assert np.allclose(arr, 500_000)

    def test_step_changes(self):
        """Step changes applied at correct months."""
        fe = FixedExpense(
            base=500_000,
            step_changes={date(2025, 7, 1): 50_000}
        )
        arr = fe.project(12, start=date(2025, 1, 1))

        # Before step change (months 0-5)
        assert arr[5] == pytest.approx(500_000)
        # After step change (months 6-11)
        assert arr[6] == pytest.approx(550_000)
        assert arr[11] == pytest.approx(550_000)

    def test_multiple_step_changes(self):
        """Multiple step changes accumulate correctly."""
        fe = FixedExpense(
            base=500_000,
            step_changes={
                date(2025, 4, 1): 50_000,
                date(2025, 7, 1): 30_000,
            }
        )
        arr = fe.project(12, start=date(2025, 1, 1))

        assert arr[2] == pytest.approx(500_000)  # March
        assert arr[3] == pytest.approx(550_000)  # April
        assert arr[6] == pytest.approx(580_000)  # July

    def test_series_output(self):
        """Series output with correct index."""
        fe = FixedExpense(base=500_000)
        series = fe.project(12, start=date(2025, 1, 1), output="series")

        assert isinstance(series, pd.Series)
        assert len(series) == 12
        assert series.name == "fixed_expense"
        assert series.index[0] == pd.Timestamp("2025-01-01")
        assert series.index[-1] == pd.Timestamp("2025-12-01")

    def test_empty_projection(self):
        """Zero months returns empty array."""
        fe = FixedExpense(base=500_000)
        arr = fe.project(0)

        assert arr.shape == (0,)

    def test_negative_base_validation(self):
        """Negative base raises error."""
        with pytest.raises(ValueError):
            FixedExpense(base=-100_000)

    def test_frozen_dataclass(self):
        """FixedExpense is immutable."""
        fe = FixedExpense(base=500_000)
        with pytest.raises(FrozenInstanceError):
            fe.base = 600_000


class TestVariableExpense:
    """Tests for VariableExpense class."""

    def test_basic_projection(self):
        """Basic stochastic projection."""
        ve = VariableExpense(base=200_000, sigma=0.1, seed=42)
        arr = ve.project(12, n_sims=100)

        assert arr.shape == (100, 12)

    def test_stochastic_noise(self):
        """Non-zero sigma produces variation across simulations."""
        ve = VariableExpense(base=200_000, sigma=0.1, seed=42)
        arr = ve.project(12, n_sims=100)

        # Should have variance across simulations
        assert arr.std(axis=0).mean() > 0

    def test_zero_sigma_deterministic(self):
        """Zero sigma produces deterministic output."""
        ve = VariableExpense(base=200_000, sigma=0.0)
        arr = ve.project(12, n_sims=100)

        # All simulations identical
        assert np.allclose(arr[0], arr[50])

    def test_seasonality_applied(self):
        """Seasonality factors applied correctly."""
        seasonality = [1.0] * 11 + [1.5]  # December 50% higher
        ve = VariableExpense(base=100_000, seasonality=seasonality, sigma=0.0)
        arr = ve.project(12, start=date(2025, 1, 1))

        # December should be 50% higher
        assert arr[11] / arr[0] == pytest.approx(1.5, rel=0.01)

    def test_seasonality_rotation(self):
        """Seasonality rotates correctly with start month."""
        seasonality = [1.0] * 11 + [2.0]  # December = 2x
        ve = VariableExpense(base=100_000, seasonality=seasonality, sigma=0.0)
        
        # Start in July
        arr = ve.project(12, start=date(2025, 7, 1))
        
        # December is month index 5 (July=0, Aug=1, ..., Dec=5)
        assert arr[5] / arr[0] == pytest.approx(2.0, rel=0.01)

    def test_floor_enforced(self):
        """Floor constraint enforced."""
        ve = VariableExpense(base=100_000, sigma=0.5, floor=50_000, seed=42)
        arr = ve.project(120, n_sims=1000)

        assert arr.min() >= 50_000

    def test_cap_enforced(self):
        """Cap constraint enforced."""
        ve = VariableExpense(base=100_000, sigma=0.5, cap=200_000, seed=42)
        arr = ve.project(120, n_sims=1000)

        assert arr.max() <= 200_000

    def test_floor_cap_combined(self):
        """Both floor and cap enforced."""
        ve = VariableExpense(
            base=100_000, sigma=0.5, floor=50_000, cap=200_000, seed=42
        )
        arr = ve.project(120, n_sims=1000)

        assert arr.min() >= 50_000
        assert arr.max() <= 200_000

    def test_inflation_applied(self):
        """Annual inflation applied correctly."""
        ve = VariableExpense(base=100_000, annual_inflation=0.12, sigma=0.0)
        arr = ve.project(12)

        # After 12 months, should be ~1.12x base
        assert arr[-1] / arr[0] == pytest.approx(1.12, rel=0.01)

    def test_series_output(self):
        """Series output returns mean across simulations."""
        ve = VariableExpense(base=200_000, sigma=0.1, seed=42)
        series = ve.project(12, start=date(2025, 1, 1), output="series", n_sims=100)

        assert isinstance(series, pd.Series)
        assert len(series) == 12
        assert series.name == "variable_expense"

    def test_reproducibility_with_seed(self):
        """Same seed produces same results."""
        ve = VariableExpense(base=200_000, sigma=0.1, seed=42)
        arr1 = ve.project(12, n_sims=100)
        arr2 = ve.project(12, n_sims=100)

        assert np.allclose(arr1, arr2)

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        ve1 = VariableExpense(base=200_000, sigma=0.1, seed=42)
        ve2 = VariableExpense(base=200_000, sigma=0.1, seed=123)
        
        arr1 = ve1.project(12, n_sims=100)
        arr2 = ve2.project(12, n_sims=100)

        assert not np.allclose(arr1, arr2)

    def test_invalid_seasonality_length(self):
        """Seasonality with wrong length raises error."""
        with pytest.raises(ValueError):
            VariableExpense(base=100_000, seasonality=[1.0] * 6)

    def test_cap_less_than_floor_validation(self):
        """Cap < floor raises error."""
        with pytest.raises(ValueError):
            VariableExpense(base=100_000, floor=100_000, cap=50_000)

    def test_frozen_dataclass(self):
        """VariableExpense is immutable."""
        ve = VariableExpense(base=200_000)
        with pytest.raises(FrozenInstanceError):
            ve.base = 300_000


class TestMicroExpense:
    """Tests for MicroExpense class."""

    def test_basic_projection(self):
        """Basic compound Poisson projection."""
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
            seed=42
        )
        arr = me.project(12, n_sims=100)

        assert arr.shape == (100, 12)
        assert arr.min() >= 0  # Expenses non-negative

    def test_expected_monthly(self):
        """Expected monthly matches lambda * E[S]."""
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
        )
        expected = me.expected_monthly()

        assert expected == 30 * 2_000  # 60,000

    def test_compound_poisson_mean_converges(self):
        """Empirical mean converges to theoretical E[C] = lambda * E[S]."""
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
            seed=42
        )
        arr = me.project(120, n_sims=5000)
        empirical_mean = arr.mean()
        theoretical_mean = 30 * 2_000  # 60,000

        # Should be within 5% of theoretical
        assert empirical_mean == pytest.approx(theoretical_mean, rel=0.05)

    def test_lognormal_distribution(self):
        """Lognormal severity distribution works."""
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
            severity_distribution="lognormal",
            seed=42
        )
        arr = me.project(12, n_sims=100)

        assert arr.shape == (100, 12)
        assert arr.min() >= 0

    def test_gamma_distribution(self):
        """Gamma severity distribution works."""
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
            severity_distribution="gamma",
            seed=42
        )
        arr = me.project(12, n_sims=100)

        assert arr.shape == (100, 12)
        assert arr.min() >= 0

    def test_lambda_seasonality(self):
        """Lambda seasonality applied correctly."""
        seasonality = [1.0] * 11 + [2.0]  # December double events
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
            lambda_seasonality=seasonality,
            seed=42
        )
        arr = me.project(12, start=date(2025, 1, 1), n_sims=5000)

        # December should have ~2x expense on average
        december_mean = arr[:, 11].mean()
        january_mean = arr[:, 0].mean()
        
        # Should be roughly 2x (with some variance)
        assert december_mean / january_mean == pytest.approx(2.0, rel=0.15)

    def test_zero_lambda(self):
        """Zero lambda produces zero expenses."""
        me = MicroExpense(
            lambda_base=0,
            severity_mean=2_000,
            severity_std=500,
            seed=42
        )
        arr = me.project(12, n_sims=100)

        assert np.allclose(arr, 0)

    def test_series_output(self):
        """Series output works correctly."""
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
            seed=42
        )
        series = me.project(12, start=date(2025, 1, 1), output="series", n_sims=100)

        assert isinstance(series, pd.Series)
        assert len(series) == 12
        assert series.name == "micro_expense"

    def test_reproducibility(self):
        """Same seed produces same results."""
        me = MicroExpense(
            lambda_base=30,
            severity_mean=2_000,
            severity_std=500,
            seed=42
        )
        arr1 = me.project(12, n_sims=100)
        arr2 = me.project(12, n_sims=100)

        assert np.allclose(arr1, arr2)

    def test_frozen_dataclass(self):
        """MicroExpense is immutable."""
        me = MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500)
        with pytest.raises(FrozenInstanceError):
            me.lambda_base = 50


class TestExpenseModel:
    """Tests for ExpenseModel facade."""

    def test_combined_projection(self):
        """All expense types combined correctly."""
        em = ExpenseModel(
            fixed=FixedExpense(base=500_000),
            variable=VariableExpense(base=200_000, sigma=0.1, seed=42),
            micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500, seed=43)
        )
        result = em.project(12, n_sims=100, output="array")

        assert "fixed" in result
        assert "variable" in result
        assert "micro" in result
        assert "total" in result
        assert result["total"].shape == (100, 12)

    def test_total_equals_sum(self):
        """Total equals sum of components."""
        em = ExpenseModel(
            fixed=FixedExpense(base=500_000),
            variable=VariableExpense(base=200_000, sigma=0.0),
            micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500, seed=42)
        )
        result = em.project(12, n_sims=100, output="array")

        total_computed = result["fixed"] + result["variable"] + result["micro"]
        assert np.allclose(result["total"], total_computed)

    def test_fixed_only(self):
        """Model with only fixed expense."""
        em = ExpenseModel(fixed=FixedExpense(base=500_000))
        result = em.project(12, output="array")

        assert np.allclose(result["total"], 500_000)

    def test_variable_only(self):
        """Model with only variable expense."""
        em = ExpenseModel(variable=VariableExpense(base=200_000, sigma=0.0))
        result = em.project(12, output="array")

        assert np.allclose(result["total"], 200_000)

    def test_micro_only(self):
        """Model with only micro expense."""
        em = ExpenseModel(
            micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500, seed=42)
        )
        result = em.project(12, n_sims=100, output="array")

        assert result["total"].shape == (100, 12)
        assert result["fixed"].shape == (100, 12)
        assert np.allclose(result["fixed"], 0)

    def test_dataframe_output(self):
        """DataFrame output format."""
        em = ExpenseModel(
            fixed=FixedExpense(base=500_000),
            variable=VariableExpense(base=200_000, sigma=0.1, seed=42)
        )
        df = em.project(12, start=date(2025, 1, 1), output="dataframe", n_sims=100)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["fixed", "variable", "micro", "total"]
        assert len(df) == 12

    def test_series_output(self):
        """Series output format."""
        em = ExpenseModel(
            fixed=FixedExpense(base=500_000),
            variable=VariableExpense(base=200_000, sigma=0.1, seed=42)
        )
        series = em.project(12, start=date(2025, 1, 1), output="series", n_sims=100)

        assert isinstance(series, pd.Series)
        assert series.name == "total_expenses"
        assert len(series) == 12

    def test_expected_monthly(self):
        """Expected monthly sums all components."""
        em = ExpenseModel(
            fixed=FixedExpense(base=500_000),
            variable=VariableExpense(base=200_000),
            micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500)
        )
        expected = em.expected_monthly()

        # 500k + 200k + 30*2k = 760k
        assert expected == 500_000 + 200_000 + 60_000

    def test_summary_metrics(self):
        """Summary returns ExpenseMetrics."""
        em = ExpenseModel(
            fixed=FixedExpense(base=500_000),
            variable=VariableExpense(base=200_000, sigma=0.1, seed=42),
            micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500, seed=43)
        )
        metrics = em.summary(12, n_sims=500, seed=42)

        assert isinstance(metrics, ExpenseMetrics)
        assert metrics.months == 12
        assert metrics.mean_fixed == pytest.approx(500_000, rel=0.01)
        assert metrics.mean_variable == pytest.approx(200_000, rel=0.1)
        assert metrics.total_expenses > 0

    def test_empty_model_validation(self):
        """Model with no expenses raises error."""
        with pytest.raises(ValueError):
            ExpenseModel()

    def test_reproducibility_with_seed(self):
        """Same seed produces reproducible results."""
        em = ExpenseModel(
            fixed=FixedExpense(base=500_000),
            variable=VariableExpense(base=200_000, sigma=0.1),
            micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500)
        )
        result1 = em.project(12, n_sims=100, seed=42, output="array")
        result2 = em.project(12, n_sims=100, seed=42, output="array")

        assert np.allclose(result1["total"], result2["total"])


class TestExpenseMetrics:
    """Tests for ExpenseMetrics dataclass."""

    def test_frozen_dataclass(self):
        """ExpenseMetrics is immutable."""
        metrics = ExpenseMetrics(
            months=12,
            total_fixed=6_000_000,
            total_variable=2_400_000,
            total_micro=720_000,
            total_expenses=9_120_000,
            mean_fixed=500_000,
            mean_variable=200_000,
            mean_micro=60_000,
            mean_total=760_000,
            std_variable=20_000,
            std_micro=15_000,
            std_total=25_000,
        )
        with pytest.raises(FrozenInstanceError):
            metrics.months = 24
