"""
Unit tests for optimization.py module.

Tests OptimizationResult, CVaROptimizer, and GoalSeeker classes.
"""

import pytest
import numpy as np
from datetime import date

from src.portfolio import Account, Portfolio
from src.goals import IntermediateGoal, TerminalGoal, GoalSet
from src.optimization import OptimizationResult, CVaROptimizer, GoalSeeker


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def accounts():
    """Create test accounts."""
    return [
        Account.from_annual("Conservative", 0.04, 0.05),
        Account.from_annual("Aggressive", 0.14, 0.15),
    ]


@pytest.fixture
def start_date():
    """Test start date."""
    return date(2025, 1, 1)


@pytest.fixture
def terminal_goals():
    """Create terminal goals."""
    return [
        TerminalGoal(account="Aggressive", threshold=5_000_000, confidence=0.70)
    ]


@pytest.fixture
def sample_goal_set(terminal_goals, accounts, start_date):
    """Create sample GoalSet."""
    return GoalSet(terminal_goals, accounts, start_date)


# ============================================================================
# OPTIMIZATIONRESULT TESTS
# ============================================================================

class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_basic_instantiation(self, terminal_goals, sample_goal_set):
        """Test basic OptimizationResult creation."""
        X = np.tile([0.5, 0.5], (24, 1))

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.5,
            n_iterations=10,
        )

        assert result.T == 24
        assert result.feasible is True
        assert result.X.shape == (24, 2)

    def test_frozen_dataclass(self, terminal_goals, sample_goal_set):
        """Test that OptimizationResult is immutable."""
        X = np.tile([0.5, 0.5], (24, 1))

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.5,
        )

        with pytest.raises(Exception):
            result.T = 36

    def test_diagnostics(self, terminal_goals, sample_goal_set):
        """Test diagnostics field."""
        X = np.tile([0.5, 0.5], (24, 1))

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.5,
            diagnostics={"solver": "ECOS", "status": "optimal"},
        )

        assert result.diagnostics["solver"] == "ECOS"


# ============================================================================
# CVAROPTIMIZER TESTS
# ============================================================================

class TestCVaROptimizerInstantiation:
    """Test CVaROptimizer initialization."""

    def test_basic_instantiation(self):
        """Test basic CVaROptimizer creation."""
        optimizer = CVaROptimizer(n_accounts=2)

        assert optimizer.M == 2

    def test_with_objective(self):
        """Test CVaROptimizer with objective."""
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        assert optimizer.objective == "balanced"

    def test_invalid_objective_raises(self):
        """Test that invalid objective raises ValueError."""
        with pytest.raises(ValueError):
            CVaROptimizer(n_accounts=2, objective="invalid")


class TestCVaROptimizerSolve:
    """Test CVaROptimizer.solve() method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        return CVaROptimizer(n_accounts=2, objective="balanced")

    @pytest.fixture
    def simulation_data(self, accounts):
        """Create simulation data."""
        T = 12
        n_sims = 100
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0, 0])

        return {"A": A, "R": R, "initial_wealth": initial_wealth, "T": T, "n_sims": n_sims, "M": M}

    def test_solve_returns_result(self, optimizer, simulation_data, terminal_goals, accounts, start_date):
        """Test solve returns OptimizationResult."""
        goal_set = GoalSet(terminal_goals, accounts, start_date)

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)
        assert result.X.shape == (simulation_data["T"], 2)

    def test_solve_allocation_simplex(self, optimizer, simulation_data, terminal_goals, accounts, start_date):
        """Test that allocation satisfies simplex constraints."""
        goal_set = GoalSet(terminal_goals, accounts, start_date)

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        # Non-negative
        assert np.all(result.X >= -1e-6)

        # Sums to 1
        row_sums = result.X.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_solve_risky_objective(self, simulation_data, terminal_goals, accounts, start_date):
        """Test solve with risky objective."""
        optimizer = CVaROptimizer(n_accounts=2, objective="risky")
        goal_set = GoalSet(terminal_goals, accounts, start_date)

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)

    def test_solve_conservative_objective(self, simulation_data, terminal_goals, accounts, start_date):
        """Test solve with conservative objective."""
        optimizer = CVaROptimizer(n_accounts=2, objective="conservative")
        goal_set = GoalSet(terminal_goals, accounts, start_date)

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)


# ============================================================================
# GOALSEEKER TESTS
# ============================================================================

class TestGoalSeekerInstantiation:
    """Test GoalSeeker initialization."""

    def test_basic_instantiation(self):
        """Test basic GoalSeeker creation."""
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer)

        assert seeker.optimizer is optimizer


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestOptimizationIntegration:
    """Integration tests for optimization pipeline."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def portfolio(self, accounts):
        """Create portfolio."""
        return Portfolio(accounts)

    def test_optimize_and_simulate(self, accounts, portfolio):
        """Test optimization followed by simulation."""
        T = 12
        n_sims = 100
        M = 2
        start_date = date(2025, 1, 1)

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0, 0])

        goals = [TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date)

        optimizer = CVaROptimizer(n_accounts=M, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        # Simulate with optimal allocation
        sim_result = portfolio.simulate(A=A, R=R, X=result.X)

        # Verify shapes
        assert sim_result["wealth"].shape == (n_sims, T + 1, M)

    def test_multiple_goals(self, accounts):
        """Test optimization with multiple goals."""
        T = 12
        n_sims = 100
        M = 2
        start_date = date(2025, 1, 1)

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0, 0])

        goals = [
            IntermediateGoal(month=6, account="Conservative", threshold=1_000_000, confidence=0.60),
            TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.60),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        optimizer = CVaROptimizer(n_accounts=M, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.goals) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
