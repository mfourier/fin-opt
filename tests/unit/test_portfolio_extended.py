"""
Tests for Portfolio withdrawal (Y) extension.

Tests the extended simulate() method that supports planned withdrawals
for reward/goal funding via the Y parameter.
"""

import numpy as np
import pytest
from datetime import date

from src.portfolio import Account, Portfolio


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_portfolio():
    """Portfolio with 2 accounts, zero initial wealth."""
    return Portfolio(
        accounts=[
            Account.from_annual("taxable", annual_return=0.08, annual_volatility=0.15),
            Account.from_annual("tax_advantaged", annual_return=0.07, annual_volatility=0.12),
        ]
    )


@pytest.fixture
def portfolio_with_wealth():
    """Portfolio with initial wealth."""
    return Portfolio(
        accounts=[
            Account.from_annual("taxable", annual_return=0.08, annual_volatility=0.15, initial_wealth=100_000),
            Account.from_annual("tax_advantaged", annual_return=0.07, annual_volatility=0.12, initial_wealth=50_000),
        ]
    )


@pytest.fixture
def deterministic_inputs():
    """Deterministic inputs for reproducible testing."""
    T, M, n_sims = 12, 2, 100
    
    # Fixed returns (5% per period for both accounts)
    R = np.full((n_sims, T, M), 0.05)
    
    # Fixed contributions ($1000/month)
    A = np.full(T, 1000.0)
    
    # 60/40 allocation
    X = np.tile([0.6, 0.4], (T, 1))
    
    return {"A": A, "R": R, "X": X, "T": T, "M": M, "n_sims": n_sims}


# =============================================================================
# Basic Y Parameter Tests
# =============================================================================

class TestYParameterBasic:
    """Basic tests for Y parameter handling."""
    
    def test_simulate_without_y_defaults_to_no_withdrawals(self, simple_portfolio, deterministic_inputs):
        """Y=None should behave as Y=0."""
        result_no_y = simple_portfolio.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=None
        )
        
        # Explicit zero Y
        Y_zeros = np.zeros((deterministic_inputs["n_sims"], deterministic_inputs["T"], deterministic_inputs["M"]))
        result_with_zero_y = simple_portfolio.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=Y_zeros
        )
        
        np.testing.assert_allclose(result_no_y["wealth"], result_with_zero_y["wealth"])
    
    def test_simulate_y_2d_broadcast(self, simple_portfolio, deterministic_inputs):
        """Y shape (T, M) should broadcast across simulations."""
        T, M, n_sims = deterministic_inputs["T"], deterministic_inputs["M"], deterministic_inputs["n_sims"]
        
        # Deterministic withdrawal of $100 from each account each period
        Y_2d = np.full((T, M), 100.0)
        
        result = simple_portfolio.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=Y_2d
        )
        
        assert result["wealth"].shape == (n_sims, T + 1, M)
        # All simulations should be identical since all inputs are deterministic
        assert np.allclose(result["wealth"][0], result["wealth"][50])
    
    def test_simulate_y_3d_stochastic(self, simple_portfolio, deterministic_inputs):
        """Y shape (n_sims, T, M) for stochastic withdrawals."""
        T, M, n_sims = deterministic_inputs["T"], deterministic_inputs["M"], deterministic_inputs["n_sims"]
        
        # Different withdrawals per simulation
        rng = np.random.default_rng(42)
        Y_3d = rng.uniform(0, 50, size=(n_sims, T, M))
        
        result = simple_portfolio.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=Y_3d
        )
        
        assert result["wealth"].shape == (n_sims, T + 1, M)
        # Simulations should differ
        assert not np.allclose(result["wealth"][0], result["wealth"][1])


# =============================================================================
# Y Validation Tests
# =============================================================================

class TestYValidation:
    """Tests for Y parameter validation."""
    
    def test_y_invalid_shape_2d(self, simple_portfolio, deterministic_inputs):
        """Y with wrong 2D shape should raise."""
        T, M = deterministic_inputs["T"], deterministic_inputs["M"]
        
        # Wrong T dimension
        Y_bad = np.zeros((T + 5, M))
        with pytest.raises(ValueError, match="Y shape"):
            simple_portfolio.simulate(
                A=deterministic_inputs["A"],
                R=deterministic_inputs["R"],
                X=deterministic_inputs["X"],
                Y=Y_bad
            )
    
    def test_y_invalid_shape_3d(self, simple_portfolio, deterministic_inputs):
        """Y with wrong 3D shape should raise."""
        T, M, n_sims = deterministic_inputs["T"], deterministic_inputs["M"], deterministic_inputs["n_sims"]
        
        # Wrong n_sims dimension
        Y_bad = np.zeros((n_sims + 10, T, M))
        with pytest.raises(ValueError, match="Y shape"):
            simple_portfolio.simulate(
                A=deterministic_inputs["A"],
                R=deterministic_inputs["R"],
                X=deterministic_inputs["X"],
                Y=Y_bad
            )
    
    def test_y_invalid_1d(self, simple_portfolio, deterministic_inputs):
        """1D Y should raise."""
        Y_1d = np.zeros(deterministic_inputs["T"])
        with pytest.raises(ValueError, match="Y must be 2D or 3D"):
            simple_portfolio.simulate(
                A=deterministic_inputs["A"],
                R=deterministic_inputs["R"],
                X=deterministic_inputs["X"],
                Y=Y_1d
            )
    
    def test_y_negative_values_raise(self, simple_portfolio, deterministic_inputs):
        """Negative Y values should raise."""
        T, M = deterministic_inputs["T"], deterministic_inputs["M"]
        
        Y_negative = np.full((T, M), -100.0)
        with pytest.raises(ValueError, match="non-negative"):
            simple_portfolio.simulate(
                A=deterministic_inputs["A"],
                R=deterministic_inputs["R"],
                X=deterministic_inputs["X"],
                Y=Y_negative
            )


# =============================================================================
# Withdrawal Effect Tests
# =============================================================================

class TestWithdrawalEffects:
    """Tests verifying withdrawal mechanics."""
    
    def test_withdrawal_reduces_wealth(self, portfolio_with_wealth, deterministic_inputs):
        """Withdrawals should reduce ending wealth."""
        T, M = deterministic_inputs["T"], deterministic_inputs["M"]
        
        # No withdrawals
        result_no_y = portfolio_with_wealth.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=None
        )
        
        # $500 withdrawal per period from each account
        Y = np.full((T, M), 500.0)
        result_with_y = portfolio_with_wealth.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=Y
        )
        
        # Final wealth should be lower with withdrawals
        assert np.all(result_with_y["wealth"][:, -1, :] < result_no_y["wealth"][:, -1, :])
    
    def test_single_period_withdrawal_math(self, simple_portfolio):
        """Verify exact math for single period withdrawal."""
        # W_1 = (W_0 + A*x - Y) * (1 + R)
        # With W_0 = 0, A = 1000, x = [0.6, 0.4], Y = [100, 50], R = 0.05
        # W_1^0 = (0 + 600 - 100) * 1.05 = 500 * 1.05 = 525
        # W_1^1 = (0 + 400 - 50) * 1.05 = 350 * 1.05 = 367.5
        
        T, M, n_sims = 1, 2, 1
        A = np.array([1000.0])
        R = np.full((n_sims, T, M), 0.05)
        X = np.array([[0.6, 0.4]])
        Y = np.array([[100.0, 50.0]])
        
        result = simple_portfolio.simulate(A, R, X, method="recursive", Y=Y)
        
        expected = np.array([[[0.0, 0.0], [525.0, 367.5]]])
        np.testing.assert_allclose(result["wealth"], expected)
    
    def test_multi_period_withdrawal_accumulation(self, portfolio_with_wealth):
        """Test withdrawal effects compound over time."""
        T, M, n_sims = 3, 2, 1
        R = np.full((n_sims, T, M), 0.10)  # 10% return
        A = np.full(T, 0.0)  # No contributions
        X = np.tile([0.5, 0.5], (T, 1))  # Doesn't matter with A=0
        
        # Withdraw $10,000 from taxable each period
        Y = np.zeros((T, M))
        Y[:, 0] = 10_000  # From taxable
        
        # W_0 = [100_000, 50_000]
        # W_1 = [(100k - 10k)*1.1, 50k*1.1] = [99k, 55k]
        # W_2 = [(99k - 10k)*1.1, 55k*1.1] = [97.9k, 60.5k]
        # W_3 = [(97.9k - 10k)*1.1, 60.5k*1.1] = [96.69k, 66.55k]
        
        result = portfolio_with_wealth.simulate(A, R, X, method="recursive", Y=Y)
        
        expected_taxable = [100_000, 99_000, 97_900, 96_690]
        expected_tax_adv = [50_000, 55_000, 60_500, 66_550]
        
        np.testing.assert_allclose(result["wealth"][0, :, 0], expected_taxable, rtol=1e-10)
        np.testing.assert_allclose(result["wealth"][0, :, 1], expected_tax_adv, rtol=1e-10)


# =============================================================================
# Method Consistency Tests
# =============================================================================

class TestMethodConsistency:
    """Tests that recursive and affine methods agree with withdrawals."""
    
    def test_recursive_affine_match_with_y(self, portfolio_with_wealth, deterministic_inputs):
        """Both methods should produce identical results."""
        T, M = deterministic_inputs["T"], deterministic_inputs["M"]
        
        Y = np.zeros((T, M))
        Y[3, 0] = 5000  # Single withdrawal at t=3 from account 0
        Y[6, 1] = 3000  # Single withdrawal at t=6 from account 1
        
        result_recursive = portfolio_with_wealth.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=Y
        )
        
        result_affine = portfolio_with_wealth.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="affine",
            Y=Y
        )
        
        np.testing.assert_allclose(
            result_recursive["wealth"],
            result_affine["wealth"],
            rtol=1e-10
        )
    
    def test_recursive_affine_match_stochastic_y(self, portfolio_with_wealth, deterministic_inputs):
        """Both methods match with stochastic Y."""
        T, M, n_sims = deterministic_inputs["T"], deterministic_inputs["M"], deterministic_inputs["n_sims"]
        
        rng = np.random.default_rng(123)
        Y = rng.uniform(0, 200, size=(n_sims, T, M))
        
        result_recursive = portfolio_with_wealth.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="recursive",
            Y=Y
        )
        
        result_affine = portfolio_with_wealth.simulate(
            A=deterministic_inputs["A"],
            R=deterministic_inputs["R"],
            X=deterministic_inputs["X"],
            method="affine",
            Y=Y
        )
        
        np.testing.assert_allclose(
            result_recursive["wealth"],
            result_affine["wealth"],
            rtol=1e-10
        )
    
    def test_recursive_affine_match_with_stochastic_returns(self, portfolio_with_wealth):
        """Both methods match with stochastic returns and fixed Y."""
        T, M, n_sims = 12, 2, 50
        
        rng = np.random.default_rng(456)
        R = rng.normal(0.05, 0.15, size=(n_sims, T, M))
        A = np.full(T, 2000.0)
        X = np.tile([0.7, 0.3], (T, 1))
        
        Y = np.zeros((T, M))
        Y[6, :] = 1000  # Withdraw at t=6
        
        result_recursive = portfolio_with_wealth.simulate(A, R, X, method="recursive", Y=Y)
        result_affine = portfolio_with_wealth.simulate(A, R, X, method="affine", Y=Y)
        
        np.testing.assert_allclose(
            result_recursive["wealth"],
            result_affine["wealth"],
            rtol=1e-10
        )


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for withdrawal functionality."""
    
    def test_withdrawal_equal_to_wealth(self, simple_portfolio):
        """Withdrawal exactly equal to available funds."""
        T, M, n_sims = 2, 2, 1
        A = np.array([1000.0, 0.0])  # Only contribute in first period
        R = np.zeros((n_sims, T, M))  # Zero returns
        X = np.array([[1.0, 0.0], [0.5, 0.5]])  # All to first account initially
        
        # After period 0: W = [1000, 0]
        # Withdraw exactly 1000 from account 0 in period 1
        Y = np.array([[0.0, 0.0], [1000.0, 0.0]])
        
        result = simple_portfolio.simulate(A, R, X, method="recursive", Y=Y)
        
        # Final wealth should be zero
        np.testing.assert_allclose(result["wealth"][0, -1, :], [0.0, 0.0])
    
    def test_withdrawal_exceeds_available(self, simple_portfolio):
        """Withdrawal exceeding available funds (negative wealth)."""
        # Note: Currently no constraint preventing negative wealth
        T, M, n_sims = 1, 2, 1
        A = np.array([100.0])
        R = np.zeros((n_sims, T, M))
        X = np.array([[0.5, 0.5]])
        Y = np.array([[200.0, 200.0]])  # Withdraw more than contributed
        
        result = simple_portfolio.simulate(A, R, X, method="recursive", Y=Y)
        
        # Wealth becomes negative (no constraint in simulation)
        assert np.all(result["wealth"][0, -1, :] < 0)
    
    def test_zero_contributions_with_withdrawals(self, portfolio_with_wealth):
        """Pure withdrawal scenario (living off wealth)."""
        T, M, n_sims = 6, 2, 1
        A = np.zeros(T)  # No contributions
        R = np.full((n_sims, T, M), 0.05)
        X = np.tile([0.5, 0.5], (T, 1))
        
        # Withdraw $5000/month from taxable
        Y = np.zeros((T, M))
        Y[:, 0] = 5000
        
        result = portfolio_with_wealth.simulate(A, R, X, method="recursive", Y=Y)
        
        # Verify wealth decreases in taxable but grows in tax_advantaged
        # Taxable: 100k -> (100k-5k)*1.05 = 99.75k -> (99.75k-5k)*1.05 = 99.4875k ...
        assert result["wealth"][0, -1, 0] < 100_000  # Taxable decreased
        assert result["wealth"][0, -1, 1] > 50_000   # Tax-adv grew
    
    def test_single_simulation(self, simple_portfolio):
        """Test with n_sims = 1."""
        T, M = 6, 2
        R = np.full((1, T, M), 0.03)
        A = np.full(T, 500.0)
        X = np.tile([0.6, 0.4], (T, 1))
        Y = np.full((T, M), 50.0)
        
        result = simple_portfolio.simulate(A, R, X, method="recursive", Y=Y)
        
        assert result["wealth"].shape == (1, T + 1, M)
        assert result["total_wealth"].shape == (1, T + 1)
    
    def test_large_simulation(self, simple_portfolio):
        """Performance test with large simulation."""
        T, M, n_sims = 240, 2, 1000  # 20 years, 1000 sims
        
        rng = np.random.default_rng(789)
        R = rng.normal(0.006, 0.04, size=(n_sims, T, M))
        A = np.full(T, 3000.0)
        X = np.tile([0.6, 0.4], (T, 1))
        Y = np.zeros((T, M))
        Y[120, 0] = 50_000  # Single withdrawal at year 10
        
        # Should complete without memory issues
        result = simple_portfolio.simulate(A, R, X, method="recursive", Y=Y)
        
        assert result["wealth"].shape == (n_sims, T + 1, M)


# =============================================================================
# Integration with RewardSchedule
# =============================================================================

class TestIntegrationWithRewards:
    """Tests integrating Portfolio.simulate with RewardSchedule.fixed_withdrawals."""
    
    def test_reward_withdrawals_in_simulation(self, portfolio_with_wealth):
        """Use RewardSchedule.get_fixed_withdrawals as Y input."""
        from src.rewards import Reward, RewardSchedule
        
        T, M = 12, 2
        
        # Create rewards (month is 1-indexed offset from start)
        vacation = Reward(
            name="vacation",
            amount=5000,
            month=6  # June (month 6 from start)
        )
        gadget = Reward(
            name="gadget", 
            amount=2000,
            month=9  # September
        )
        
        schedule = RewardSchedule(rewards=[vacation, gadget])
        
        # Get withdrawal matrix using proportional policy (default)
        Y = schedule.get_fixed_withdrawals(
            T=T,
            M=M,
            accounts=portfolio_with_wealth.accounts
        )
        
        # Run simulation
        rng = np.random.default_rng(42)
        R = rng.normal(0.005, 0.03, size=(100, T, M))
        A = np.full(T, 2000.0)
        X = np.tile([0.6, 0.4], (T, 1))
        
        result = portfolio_with_wealth.simulate(A, R, X, method="recursive", Y=Y)
        
        assert result["wealth"].shape == (100, T + 1, M)
        
        # Verify withdrawals happened at expected times
        # Month 6 = index 5 (0-indexed), Month 9 = index 8
        # Before and after should show impact
        assert result["total_wealth"][:, 6].mean() < result["total_wealth"][:, 5].mean() + A[5]
    
    def test_reward_specific_account_withdrawal(self, portfolio_with_wealth):
        """Test reward targeted at specific account."""
        from src.rewards import Reward, RewardSchedule
        
        T, M = 6, 2
        
        # Reward from taxable only
        reward = Reward(
            name="car",
            amount=10000,
            month=3,  # Month 3 from start
            account="taxable"
        )
        
        # Use single_account policy to respect specific account
        schedule = RewardSchedule(
            rewards=[reward],
            withdrawal_policy="single_account",
            default_account="taxable"  # Required for single_account policy
        )
        
        Y = schedule.get_fixed_withdrawals(
            T=T,
            M=M,
            accounts=portfolio_with_wealth.accounts
        )
        
        # Verify withdrawal is only from taxable (account 0)
        assert Y[2, 0] == 10000  # Month 3 (index 2), taxable
        assert Y[2, 1] == 0      # No withdrawal from tax_advantaged
        assert Y.sum() == 10000  # Total withdrawals
