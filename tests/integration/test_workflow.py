"""
Integration test for full FinOpt workflow.

Tests the complete pipeline from income modeling through optimization
to verify all components work together correctly.
"""

from datetime import date
import numpy as np
import pytest

from src.income import FixedIncome, VariableIncome, IncomeModel
from src.portfolio import Account, Portfolio
from src.returns import ReturnModel
from src.goals import TerminalGoal, IntermediateGoal
from src.model import FinancialModel


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete optimization workflow."""

    def test_simple_two_account_optimization(self):
        """
        Test complete workflow with 2 accounts and 1 terminal goal.

        This is a smoke test to ensure all components integrate properly.
        """
        # 1. Setup income
        fixed = FixedIncome(base=1_500_000, annual_growth=0.03)
        variable = VariableIncome(base=200_000, sigma=0.10, seed=42)
        income = IncomeModel(fixed=fixed, variable=variable)

        # 2. Setup accounts
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05, initial_wealth=0),
            Account.from_annual("Aggressive", 0.14, 0.15, initial_wealth=0),
        ]

        # 3. Create financial model
        model = FinancialModel(income=income, accounts=accounts)

        # Verify model was created
        assert model.income == income
        assert len(model.accounts) == 2
        assert model.returns is not None

        # 4. Run a simple simulation (no optimization)
        T = 24
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))  # 60-40 split

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        # Verify simulation result structure
        assert result.wealth.shape == (n_sims, T + 1, 2)
        assert result.contributions.shape == (n_sims, T)
        assert result.returns.shape == (n_sims, T, 2)

        # Wealth should be non-negative
        assert np.all(result.wealth >= 0)

        # Total wealth should generally increase (with high probability)
        mean_final_wealth = result.wealth[:, -1, :].sum(axis=1).mean()
        mean_initial_wealth = result.wealth[:, 0, :].sum(axis=1).mean()
        assert mean_final_wealth > mean_initial_wealth

    def test_optimization_with_terminal_goal(self):
        """
        Test optimization workflow with terminal goal.

        Note: This test uses small n_sims for speed. Real optimization
        would use 500-1000 simulations.
        """
        # Setup
        fixed = FixedIncome(base=1_500_000, annual_growth=0.03)
        variable = VariableIncome(base=200_000, sigma=0.0, seed=42)  # Deterministic
        income = IncomeModel(fixed=fixed, variable=variable)

        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

        model = FinancialModel(income=income, accounts=accounts)

        # Define goal - modest to ensure feasibility
        goals = [
            TerminalGoal(
                account="Aggressive",
                threshold=5_000_000,  # 5M CLP
                confidence=0.70,
            )
        ]

        # Note: Actual optimization requires cvxpy, which may not be installed
        # This test verifies the setup is correct
        try:
            from src.optimization import CVaROptimizer

            optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

            # Run optimization with small parameters for speed
            result = model.optimize(
                goals=goals,
                optimizer=optimizer,
                T_max=60,
                n_sims=50,  # Small for CI
                seed=42
            )

            # Verify optimization result
            assert result is not None
            assert hasattr(result, 'wealth')
            assert hasattr(result, 'allocation')

        except ImportError:
            pytest.skip("cvxpy not installed - skipping optimization test")

    def test_model_caching(self):
        """Test that simulation results are cached correctly."""
        fixed = FixedIncome(base=1_500_000)
        income = IncomeModel(fixed=fixed)
        accounts = [Account.from_annual("Test", 0.08, 0.10)]

        model = FinancialModel(income=income, accounts=accounts)

        # First simulation
        X1 = np.ones((12, 1))
        result1 = model.simulate(T=12, n_sims=10, X=X1, seed=42)

        # Same simulation should use cache
        result2 = model.simulate(T=12, n_sims=10, X=X1, seed=42)

        # Should return same object (cached)
        assert result1 is result2

        # Different parameters should not use cache
        result3 = model.simulate(T=12, n_sims=10, X=X1, seed=99)
        assert result3 is not result1

    def test_metrics_computation(self):
        """Test that metrics are computed correctly on simulation results."""
        fixed = FixedIncome(base=1_500_000, annual_growth=0.0)
        income = IncomeModel(fixed=fixed)
        accounts = [Account.from_annual("Test", 0.08, 0.10)]

        model = FinancialModel(income=income, accounts=accounts)

        X = np.ones((12, 1))
        result = model.simulate(T=12, n_sims=100, X=X, seed=42)

        # Compute metrics
        metrics = result.metrics()

        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "mean_final_wealth" in metrics
        assert "median_final_wealth" in metrics
        assert "total_contributions" in metrics

        # Values should be reasonable
        assert metrics["mean_final_wealth"] > 0
        assert metrics["total_contributions"] > 0

    def test_multiple_goals(self):
        """Test model with both intermediate and terminal goals."""
        fixed = FixedIncome(base=2_000_000, annual_growth=0.02)
        income = IncomeModel(fixed=fixed)

        accounts = [
            Account.from_annual("Emergency", 0.04, 0.05),
            Account.from_annual("Growth", 0.12, 0.14),
        ]

        model = FinancialModel(income=income, accounts=accounts)

        # Define multiple goals
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

        # Verify goals are valid
        from src.goals import GoalSet, check_goals

        check_goals(goals)
        goal_set = GoalSet(
            goals=goals,
            accounts=accounts,
            start_date=date(2025, 1, 1)
        )

        assert len(goal_set.goals) == 2
        assert goal_set.T_min == 6  # Minimum from intermediate goal

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

    def test_large_scale_simulation(self):
        """Test model handles larger simulations efficiently."""
        fixed = FixedIncome(base=1_500_000)
        income = IncomeModel(fixed=fixed)
        accounts = [Account.from_annual("Test", 0.08, 0.10)]

        model = FinancialModel(income=income, accounts=accounts)

        # Large simulation
        T = 120  # 10 years
        n_sims = 500
        X = np.ones((T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        # Verify shape
        assert result.wealth.shape == (n_sims, T + 1, 1)

        # Verify no NaN or inf values
        assert np.all(np.isfinite(result.wealth))
        assert np.all(np.isfinite(result.contributions))
        assert np.all(np.isfinite(result.returns))


@pytest.mark.integration
class TestEdgeCases:
    """Integration tests for edge cases and error handling."""

    def test_zero_initial_wealth(self):
        """Test model with zero initial wealth (contribution-driven)."""
        income = IncomeModel(fixed=FixedIncome(base=1_000_000))
        accounts = [Account.from_annual("Test", 0.08, 0.10, initial_wealth=0)]

        model = FinancialModel(income=income, accounts=accounts)
        X = np.ones((12, 1))

        result = model.simulate(T=12, n_sims=10, X=X, seed=42)

        # Initial wealth should be zero
        assert np.all(result.wealth[:, 0, :] == 0)

        # Final wealth should be positive (from contributions)
        assert np.all(result.wealth[:, -1, :] > 0)

    def test_invalid_allocation_policy(self):
        """Test that invalid allocation policy raises error."""
        income = IncomeModel(fixed=FixedIncome(base=1_000_000))
        accounts = [Account.from_annual("A", 0.08, 0.10)]

        model = FinancialModel(income=income, accounts=accounts)

        # Allocation that doesn't sum to 1
        X_invalid = np.ones((12, 1)) * 0.5

        with pytest.raises(ValueError, match="sum to 1|simplex"):
            model.simulate(T=12, n_sims=10, X=X_invalid, seed=42)

    def test_mismatched_dimensions(self):
        """Test error handling for dimension mismatches."""
        income = IncomeModel(fixed=FixedIncome(base=1_000_000))
        accounts = [
            Account.from_annual("A", 0.08, 0.10),
            Account.from_annual("B", 0.12, 0.15),
        ]

        model = FinancialModel(income=income, accounts=accounts)

        # Allocation for wrong number of accounts
        X_wrong = np.ones((12, 1))  # Should be (12, 2)

        with pytest.raises(ValueError, match="shape|accounts"):
            model.simulate(T=12, n_sims=10, X=X_wrong, seed=42)
