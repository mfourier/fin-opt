"""
Unit tests for portfolio.py module.

Tests Account and Portfolio classes.
"""

import pytest
import numpy as np
from datetime import date

from src.portfolio import Account, Portfolio


# ============================================================================
# ACCOUNT TESTS
# ============================================================================

class TestAccountInstantiation:
    """Test Account initialization."""

    def test_from_annual_basic(self):
        """Test Account.from_annual() basic creation."""
        acc = Account.from_annual("Conservative", annual_return=0.04, annual_volatility=0.05)

        assert acc.name == "Conservative"
        assert acc.initial_wealth == 0.0
        assert "mu" in acc.return_strategy
        assert "sigma" in acc.return_strategy

    def test_from_annual_with_initial_wealth(self):
        """Test Account.from_annual() with initial wealth."""
        acc = Account.from_annual("Emergency", annual_return=0.04,
                                   annual_volatility=0.05, initial_wealth=1_000_000)

        assert acc.initial_wealth == 1_000_000

    def test_annual_params_property(self):
        """Test annual_params property returns original values."""
        acc = Account.from_annual("Test", annual_return=0.08, annual_volatility=0.12)

        params = acc.annual_params
        assert params["return"] == pytest.approx(0.08, rel=0.01)
        assert params["volatility"] == pytest.approx(0.12, rel=0.01)

    def test_monthly_params_property(self):
        """Test monthly_params property returns converted values."""
        acc = Account.from_annual("Test", annual_return=0.12, annual_volatility=0.10)

        params = acc.monthly_params
        assert "mu" in params
        assert "sigma" in params
        assert params["mu"] > 0
        assert params["sigma"] > 0

    def test_negative_initial_wealth_raises(self):
        """Test that negative initial wealth raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Account.from_annual("Test", annual_return=0.04,
                               annual_volatility=0.05, initial_wealth=-100)

    def test_negative_sigma_raises(self):
        """Test that negative volatility raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Account.from_annual("Test", annual_return=0.04, annual_volatility=-0.05)

    def test_frozen_dataclass(self):
        """Test that Account is immutable."""
        acc = Account.from_annual("Test", annual_return=0.04, annual_volatility=0.05)
        with pytest.raises(Exception):
            acc.name = "Changed"


class TestAccountFromMonthly:
    """Test Account.from_monthly() factory method."""

    def test_from_monthly_basic(self):
        """Test Account.from_monthly() creation."""
        acc = Account.from_monthly("Custom", monthly_mu=0.005, monthly_sigma=0.03)

        assert acc.name == "Custom"
        assert acc.return_strategy["mu"] == 0.005
        assert acc.return_strategy["sigma"] == 0.03

    def test_monthly_annual_roundtrip(self):
        """Test that monthly -> annual conversion is consistent."""
        acc = Account.from_monthly("Test", monthly_mu=0.005, monthly_sigma=0.03)

        annual = acc.annual_params
        assert annual["return"] > 0
        assert annual["volatility"] > 0


# ============================================================================
# PORTFOLIO TESTS
# ============================================================================

class TestPortfolioInstantiation:
    """Test Portfolio initialization."""

    def test_basic_instantiation(self):
        """Test basic Portfolio creation."""
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        portfolio = Portfolio(accounts)

        assert portfolio.M == 2
        assert portfolio.accounts == accounts

    def test_single_account(self):
        """Test Portfolio with single account."""
        accounts = [Account.from_annual("Single", 0.08, 0.10)]
        portfolio = Portfolio(accounts)

        assert portfolio.M == 1

    def test_initial_wealth_vector(self):
        """Test initial_wealth_vector property."""
        accounts = [
            Account.from_annual("A", 0.04, 0.05, initial_wealth=1_000_000),
            Account.from_annual("B", 0.08, 0.10, initial_wealth=500_000),
        ]
        portfolio = Portfolio(accounts)

        W0 = portfolio.initial_wealth_vector
        assert W0.shape == (2,)
        np.testing.assert_array_equal(W0, [1_000_000, 500_000])


class TestPortfolioSimulate:
    """Test Portfolio.simulate() method."""

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        return Portfolio(accounts)

    @pytest.fixture
    def simulation_inputs(self):
        """Create mock simulation inputs."""
        T = 12
        n_sims = 50
        M = 2

        A = np.full(T, 100_000.0)  # 1D contributions
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        X = np.tile([0.6, 0.4], (T, 1))  # 60-40 allocation

        return {"A": A, "R": R, "X": X, "T": T, "n_sims": n_sims, "M": M}

    def test_simulate_basic(self, portfolio, simulation_inputs):
        """Test basic simulate call."""
        A = simulation_inputs["A"]
        R = simulation_inputs["R"]
        X = simulation_inputs["X"]

        result = portfolio.simulate(A=A, R=R, X=X)

        assert "wealth" in result
        assert "total_wealth" in result

    def test_simulate_wealth_shape(self, portfolio, simulation_inputs):
        """Test wealth output shape."""
        A = simulation_inputs["A"]
        R = simulation_inputs["R"]
        X = simulation_inputs["X"]
        T = simulation_inputs["T"]
        n_sims = simulation_inputs["n_sims"]
        M = simulation_inputs["M"]

        result = portfolio.simulate(A=A, R=R, X=X)

        # Shape: (n_sims, T+1, M) - includes initial wealth at t=0
        assert result["wealth"].shape == (n_sims, T + 1, M)
        assert result["total_wealth"].shape == (n_sims, T + 1)

    def test_simulate_2d_contributions(self, portfolio):
        """Test simulate with 2D contributions."""
        T = 12
        n_sims = 50
        M = 2

        A = np.full((n_sims, T), 100_000.0)  # 2D contributions
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        X = np.tile([0.6, 0.4], (T, 1))

        result = portfolio.simulate(A=A, R=R, X=X)

        assert result["wealth"].shape == (n_sims, T + 1, M)

    def test_simulate_recursive_method(self, portfolio, simulation_inputs):
        """Test recursive computation method."""
        A = simulation_inputs["A"]
        R = simulation_inputs["R"]
        X = simulation_inputs["X"]

        result = portfolio.simulate(A=A, R=R, X=X, method="recursive")

        assert result["wealth"].shape[1] == simulation_inputs["T"] + 1

    def test_simulate_affine_method(self, portfolio, simulation_inputs):
        """Test affine computation method."""
        A = simulation_inputs["A"]
        R = simulation_inputs["R"]
        X = simulation_inputs["X"]

        result = portfolio.simulate(A=A, R=R, X=X, method="affine")

        assert result["wealth"].shape[1] == simulation_inputs["T"] + 1

    def test_simulate_methods_agree(self, portfolio, simulation_inputs):
        """Test that recursive and affine methods produce same results."""
        A = simulation_inputs["A"]
        R = simulation_inputs["R"]
        X = simulation_inputs["X"]

        result_rec = portfolio.simulate(A=A, R=R, X=X, method="recursive")
        result_aff = portfolio.simulate(A=A, R=R, X=X, method="affine")

        np.testing.assert_array_almost_equal(
            result_rec["wealth"],
            result_aff["wealth"],
            decimal=6
        )

    def test_simulate_w0_override(self, portfolio, simulation_inputs):
        """Test simulate with W0_override parameter."""
        A = simulation_inputs["A"]
        R = simulation_inputs["R"]
        X = simulation_inputs["X"]

        W0_override = np.array([5_000_000, 2_000_000])
        result = portfolio.simulate(A=A, R=R, X=X, W0_override=W0_override)

        # Initial wealth should match override
        np.testing.assert_array_almost_equal(
            result["wealth"][:, 0, :].mean(axis=0),
            W0_override
        )

    def test_simulate_wealth_non_negative(self, portfolio, simulation_inputs):
        """Test that wealth remains non-negative."""
        A = simulation_inputs["A"]
        R = simulation_inputs["R"]
        X = simulation_inputs["X"]

        result = portfolio.simulate(A=A, R=R, X=X)

        # With positive contributions and R > -1, wealth should be non-negative
        assert np.all(result["wealth"] >= -1e-6)


class TestPortfolioAllocationValidation:
    """Test allocation policy validation."""

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        accounts = [
            Account.from_annual("A", 0.04, 0.05),
            Account.from_annual("B", 0.08, 0.10),
        ]
        return Portfolio(accounts)

    def test_valid_allocation(self, portfolio):
        """Test valid allocation passes."""
        T = 12
        n_sims = 50
        M = 2

        A = np.full(T, 100_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        X = np.tile([0.6, 0.4], (T, 1))  # Sums to 1.0

        result = portfolio.simulate(A=A, R=R, X=X)
        assert result["wealth"].shape == (n_sims, T + 1, M)

    def test_allocation_all_to_one_account(self, portfolio):
        """Test allocation 100% to one account."""
        T = 12
        n_sims = 50
        M = 2

        A = np.full(T, 100_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        X = np.zeros((T, M))
        X[:, 0] = 1.0  # All to account 0

        result = portfolio.simulate(A=A, R=R, X=X)
        assert result["wealth"].shape == (n_sims, T + 1, M)


class TestPortfolioAccumulationFactors:
    """Test accumulation factors computation."""

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio."""
        accounts = [
            Account.from_annual("A", 0.04, 0.05),
            Account.from_annual("B", 0.08, 0.10),
        ]
        return Portfolio(accounts)

    def test_accumulation_factors_shape(self, portfolio):
        """Test accumulation factors shape."""
        T = 12
        n_sims = 50
        M = 2

        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005

        F = portfolio.compute_accumulation_factors(R)

        # Shape: (n_sims, T+1, T+1, M)
        assert F.shape == (n_sims, T + 1, T + 1, M)

    def test_accumulation_factors_diagonal(self, portfolio):
        """Test that F[s, s, :] = 1 (diagonal)."""
        T = 12
        n_sims = 50
        M = 2

        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005

        F = portfolio.compute_accumulation_factors(R)

        # Diagonal should be 1.0
        for s in range(T + 1):
            assert np.allclose(F[:, s, s, :], 1.0)

    def test_accumulation_factors_zero_returns(self, portfolio):
        """Test accumulation factors with zero returns."""
        T = 12
        n_sims = 50
        M = 2

        R = np.zeros((n_sims, T, M))

        F = portfolio.compute_accumulation_factors(R)

        # All factors should be 1.0
        assert np.allclose(F, 1.0)


class TestPortfolioWealthDynamics:
    """Test wealth dynamics behavior."""

    @pytest.fixture
    def portfolio(self):
        """Create test portfolio with initial wealth."""
        accounts = [
            Account.from_annual("A", 0.04, 0.05, initial_wealth=1_000_000),
            Account.from_annual("B", 0.08, 0.10, initial_wealth=500_000),
        ]
        return Portfolio(accounts)

    def test_zero_contributions_zero_returns(self, portfolio):
        """Test wealth preserved with zero contributions and returns."""
        T = 10
        n_sims = 50
        M = 2

        A = np.zeros(T)
        R = np.zeros((n_sims, T, M))
        X = np.tile([0.5, 0.5], (T, 1))

        result = portfolio.simulate(A=A, R=R, X=X)

        # Wealth should be constant
        W0 = portfolio.initial_wealth_vector
        for t in range(T + 1):
            for i in range(n_sims):
                np.testing.assert_array_almost_equal(result["wealth"][i, t, :], W0)

    def test_positive_returns_increase_wealth(self, portfolio):
        """Test wealth increases with positive returns."""
        T = 12
        n_sims = 50
        M = 2

        A = np.full(T, 100_000.0)
        R = np.full((n_sims, T, M), 0.02)  # 2% per month
        X = np.tile([0.5, 0.5], (T, 1))

        result = portfolio.simulate(A=A, R=R, X=X)

        # Final wealth should exceed initial
        final_total = result["total_wealth"][:, -1]
        initial_total = result["total_wealth"][:, 0]
        assert np.all(final_total > initial_total)


class TestPortfolioEdgeCases:
    """Test edge cases."""

    def test_single_period(self):
        """Test simulation with T=1."""
        accounts = [Account.from_annual("A", 0.04, 0.05)]
        portfolio = Portfolio(accounts)

        A = np.array([100_000.0])
        R = np.random.randn(10, 1, 1) * 0.02
        X = np.array([[1.0]])

        result = portfolio.simulate(A=A, R=R, X=X)
        assert result["wealth"].shape == (10, 2, 1)

    def test_many_accounts(self):
        """Test portfolio with many accounts."""
        accounts = [
            Account.from_annual(f"Acc{i}", 0.04 + 0.02*i, 0.05 + 0.02*i)
            for i in range(5)
        ]
        portfolio = Portfolio(accounts)

        T = 12
        n_sims = 50
        M = 5

        A = np.full(T, 100_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02
        X = np.full((T, M), 1/M)  # Equal allocation

        result = portfolio.simulate(A=A, R=R, X=X)
        assert result["wealth"].shape == (n_sims, T + 1, M)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
