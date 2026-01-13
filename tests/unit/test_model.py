"""
Unit tests for model.py module.

Tests FinancialModel and SimulationResult classes.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from src.income import IncomeModel, FixedIncome, VariableIncome
from src.portfolio import Account
from src.model import FinancialModel, SimulationResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def income():
    """Create test income model."""
    return IncomeModel(
        fixed=FixedIncome(base=1_500_000, annual_growth=0.03),
        variable=VariableIncome(base=200_000, sigma=0.1, seed=42)
    )


@pytest.fixture
def accounts():
    """Create test accounts."""
    return [
        Account.from_annual("Conservative", 0.04, 0.05),
        Account.from_annual("Aggressive", 0.14, 0.15),
    ]


@pytest.fixture
def model(income, accounts):
    """Create test FinancialModel."""
    return FinancialModel(income=income, accounts=accounts)


# ============================================================================
# FINANCIALMODEL INSTANTIATION TESTS
# ============================================================================

class TestFinancialModelInstantiation:
    """Test FinancialModel initialization."""

    def test_basic_instantiation(self, income, accounts):
        """Test basic FinancialModel creation."""
        model = FinancialModel(income=income, accounts=accounts)

        assert model.income is income
        assert model.accounts == accounts
        assert model.M == 2

    def test_with_correlation(self, income, accounts):
        """Test FinancialModel with correlation matrix."""
        corr = np.array([
            [1.0, 0.5],
            [0.5, 1.0],
        ])
        model = FinancialModel(income=income, accounts=accounts, correlation=corr)

        np.testing.assert_array_almost_equal(model.returns.default_correlation, corr)

    def test_single_account(self, income):
        """Test FinancialModel with single account."""
        accounts = [Account.from_annual("Single", 0.08, 0.10)]
        model = FinancialModel(income=income, accounts=accounts)

        assert model.M == 1

    def test_account_names_property(self, model, accounts):
        """Test account_names property."""
        names = model.account_names

        assert len(names) == len(accounts)
        assert names[0] == "Conservative"
        assert names[1] == "Aggressive"


# ============================================================================
# FINANCIALMODEL SIMULATE TESTS
# ============================================================================

class TestFinancialModelSimulate:
    """Test FinancialModel.simulate() method."""

    def test_simulate_returns_result(self, model):
        """Test simulate returns SimulationResult."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert isinstance(result, SimulationResult)

    def test_simulate_wealth_shape(self, model):
        """Test simulate returns correct wealth shape."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert result.wealth.shape == (n_sims, T + 1, 2)

    def test_simulate_contributions_shape(self, model):
        """Test simulate returns correct contributions shape."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert result.contributions.shape == (n_sims, T)

    def test_simulate_returns_shape(self, model):
        """Test simulate returns correct returns shape."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert result.returns.shape == (n_sims, T, 2)

    def test_simulate_reproducibility(self, model):
        """Test simulate reproducibility with seed."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)
        result2 = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        np.testing.assert_array_equal(result1.wealth, result2.wealth)

    def test_simulate_different_seeds(self, model):
        """Test simulate produces different results with different seeds."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)
        result2 = model.simulate(T=T, n_sims=n_sims, X=X, seed=43)

        assert not np.allclose(result1.wealth, result2.wealth)

    def test_simulate_start_date(self, model):
        """Test simulate with start_date parameter."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42, start=date(2025, 1, 1))

        assert isinstance(result, SimulationResult)


# ============================================================================
# SIMULATIONRESULT TESTS
# ============================================================================

class TestSimulationResult:
    """Test SimulationResult dataclass."""

    @pytest.fixture
    def sample_result(self, model):
        """Create sample simulation result."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))
        return model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

    def test_n_sims_property(self, sample_result):
        """Test n_sims property."""
        assert sample_result.n_sims == 50

    def test_T_property(self, sample_result):
        """Test T property."""
        assert sample_result.T == 12

    def test_allocation_policy(self, sample_result):
        """Test allocation_policy stored correctly."""
        assert sample_result.allocation_policy.shape == (12, 2)

    def test_total_wealth_property(self, sample_result):
        """Test total_wealth computed correctly."""
        total = sample_result.wealth.sum(axis=2)
        np.testing.assert_array_almost_equal(total, sample_result.wealth[:, :, 0] + sample_result.wealth[:, :, 1])


# ============================================================================
# FINANCIALMODEL CACHING TESTS
# ============================================================================

class TestFinancialModelCaching:
    """Test FinancialModel caching behavior."""

    def test_cache_enabled_by_default(self, model):
        """Test that caching is enabled by default."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        # First simulation
        result1 = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        # Check params_hash is generated
        assert result1.params_hash is not None


# ============================================================================
# FINANCIALMODEL INTEGRATION TESTS
# ============================================================================

class TestFinancialModelIntegration:
    """Integration tests for FinancialModel."""

    def test_full_workflow(self, income, accounts):
        """Test complete workflow."""
        model = FinancialModel(income=income, accounts=accounts)

        T = 24
        n_sims = 100
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42, start=date(2025, 1, 1))

        # Verify result structure
        assert result.wealth.shape == (n_sims, T + 1, 2)
        assert result.contributions.shape == (n_sims, T)
        assert result.returns.shape == (n_sims, T, 2)

        # Verify wealth is non-negative
        assert np.all(result.wealth >= -1e-6)

    def test_different_allocation_policies(self, model):
        """Test different allocation policies."""
        T = 12
        n_sims = 50

        # Aggressive: 20-80
        X_aggressive = np.tile([0.2, 0.8], (T, 1))
        result_aggressive = model.simulate(T=T, n_sims=n_sims, X=X_aggressive, seed=42)

        # Conservative: 80-20
        X_conservative = np.tile([0.8, 0.2], (T, 1))
        result_conservative = model.simulate(T=T, n_sims=n_sims, X=X_conservative, seed=42)

        # Results should differ
        assert not np.allclose(result_aggressive.wealth, result_conservative.wealth)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
