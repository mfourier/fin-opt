"""
Integration tests for FinOpt workflow.

Tests end-to-end scenarios combining income, portfolio, returns,
and optimization modules.
"""

import pytest
import numpy as np
from datetime import date

from src.income import IncomeModel, FixedIncome, VariableIncome
from src.portfolio import Account, Portfolio
from src.model import FinancialModel
from src.goals import IntermediateGoal, TerminalGoal, GoalSet


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete workflows."""

    def test_simple_two_account_optimization(self):
        """Test complete workflow with two accounts."""
        # Setup income
        fixed = FixedIncome(base=1_500_000, annual_growth=0.03)
        variable = VariableIncome(base=200_000, sigma=0.10, seed=42)
        income = IncomeModel(fixed=fixed, variable=variable)

        # Setup accounts
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

        # Create model
        model = FinancialModel(income=income, accounts=accounts)

        # Simulate
        T = 24
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        # Verify results
        assert result.wealth.shape == (n_sims, T + 1, 2)
        assert result.contributions.shape == (n_sims, T)
        assert result.returns.shape == (n_sims, T, 2)

        # Verify wealth is non-negative
        assert np.all(result.wealth >= -1e-6)

        # Verify final wealth is positive (contributions accumulated)
        assert np.all(result.wealth[:, -1, :].sum(axis=1) > 0)

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        fixed = FixedIncome(base=1_500_000)
        variable = VariableIncome(base=200_000, sigma=0.15)
        income = IncomeModel(fixed=fixed, variable=variable)

        accounts = [Account.from_annual("Test", 0.08, 0.12)]
        model = FinancialModel(income=income, accounts=accounts)

        X = np.ones((12, 1))

        # Run twice with same seed
        result1 = model.simulate(T=12, n_sims=50, X=X, seed=42)
        result2 = model.simulate(T=12, n_sims=50, X=X, seed=42)

        # Results should be identical
        np.testing.assert_array_equal(result1.wealth, result2.wealth)
        np.testing.assert_array_equal(result1.contributions, result2.contributions)
        np.testing.assert_array_equal(result1.returns, result2.returns)

    def test_goal_set_creation(self):
        """Test creating GoalSet with multiple goals."""
        accounts = [
            Account.from_annual("Emergency", 0.04, 0.05),
            Account.from_annual("Growth", 0.12, 0.14),
        ]

        goals = [
            IntermediateGoal(
                month=6,
                account="Emergency",
                threshold=3_000_000,
                confidence=0.80,
            ),
            TerminalGoal(
                account="Growth",
                threshold=20_000_000,
                confidence=0.70,
            ),
        ]

        goal_set = GoalSet(
            goals=goals,
            accounts=accounts,
            start_date=date(2025, 1, 1)
        )

        assert len(goal_set) == 2
        assert goal_set.T_min == 6


@pytest.mark.integration
class TestOptimizationIntegration:
    """Integration tests for optimization."""

    def test_cvar_optimizer_solve(self):
        """Test CVaROptimizer solve with simple scenario."""
        try:
            from src.optimization import CVaROptimizer

            accounts = [
                Account.from_annual("Conservative", 0.04, 0.05),
                Account.from_annual("Aggressive", 0.14, 0.15),
            ]

            T = 12
            n_sims = 100
            M = 2
            start_date = date(2025, 1, 1)

            np.random.seed(42)
            A = np.full((n_sims, T), 500_000.0)
            R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
            W0 = np.array([0, 0])

            goals = [TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.60)]
            goal_set = GoalSet(goals, accounts, start_date)

            optimizer = CVaROptimizer(n_accounts=M, objective="balanced")

            result = optimizer.solve(
                T=T,
                A=A,
                R=R,
                W0=W0,
                goal_set=goal_set,
            )

            # Verify result
            assert result.X.shape == (T, M)
            assert np.all(result.X >= -1e-6)
            assert np.allclose(result.X.sum(axis=1), 1.0, atol=1e-5)

        except ImportError:
            pytest.skip("cvxpy not installed - skipping optimization test")


@pytest.mark.integration
class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_zero_initial_wealth(self):
        """Test model with zero initial wealth (contribution-driven)."""
        fixed = FixedIncome(base=1_000_000)
        variable = VariableIncome(base=100_000, sigma=0.05)
        income = IncomeModel(fixed=fixed, variable=variable)

        accounts = [Account.from_annual("Test", 0.08, 0.10, initial_wealth=0)]

        model = FinancialModel(income=income, accounts=accounts)
        X = np.ones((12, 1))

        result = model.simulate(T=12, n_sims=10, X=X, seed=42)

        # Initial wealth should be zero
        assert np.all(result.wealth[:, 0, :] == 0)

        # Final wealth should be positive (from contributions)
        assert np.all(result.wealth[:, -1, :] > 0)

    def test_single_account_portfolio(self):
        """Test workflow with single account."""
        fixed = FixedIncome(base=1_500_000)
        variable = VariableIncome(base=200_000, sigma=0.1)
        income = IncomeModel(fixed=fixed, variable=variable)

        accounts = [Account.from_annual("Single", 0.08, 0.12)]
        model = FinancialModel(income=income, accounts=accounts)

        X = np.ones((24, 1))
        result = model.simulate(T=24, n_sims=50, X=X, seed=42)

        assert result.wealth.shape == (50, 25, 1)
        assert np.all(np.isfinite(result.wealth))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
