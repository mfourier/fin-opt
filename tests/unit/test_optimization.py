"""
Unit tests for optimization.py module.

Tests OptimizationResult, CVaROptimizer, GoalSeeker, and AllocationOptimizer classes.
"""

import pytest
import numpy as np
from datetime import date
from unittest.mock import Mock, patch

from src.portfolio import Account, Portfolio
from src.goals import IntermediateGoal, TerminalGoal, GoalSet
from src.optimization import OptimizationResult, CVaROptimizer, GoalSeeker, AllocationOptimizer
from src.exceptions import InfeasibleError
from src.income import FixedIncome, IncomeModel
from src.model import FinancialModel


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
def three_accounts():
    """Create three test accounts."""
    return [
        Account.from_annual("Conservative", 0.04, 0.05),
        Account.from_annual("Moderate", 0.08, 0.10),
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
def intermediate_goals(start_date):
    """Create intermediate goals."""
    return [
        IntermediateGoal(
            account="Conservative",
            threshold=1_000_000,
            confidence=0.80,
            date=date(2025, 6, 1)
        )
    ]


@pytest.fixture
def mixed_goals(start_date):
    """Create mixed intermediate and terminal goals."""
    return [
        IntermediateGoal(
            account="Conservative",
            threshold=1_000_000,
            confidence=0.80,
            date=date(2025, 6, 1)
        ),
        TerminalGoal(account="Aggressive", threshold=5_000_000, confidence=0.70)
    ]


@pytest.fixture
def sample_goal_set(terminal_goals, accounts, start_date):
    """Create sample GoalSet."""
    return GoalSet(terminal_goals, accounts, start_date)


@pytest.fixture
def simulation_data_small():
    """Create small simulation data for fast tests."""
    T = 6
    n_sims = 50
    M = 2

    np.random.seed(42)
    A = np.full((n_sims, T), 500_000.0)
    R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
    initial_wealth = np.array([0.0, 0.0])

    return {"A": A, "R": R, "initial_wealth": initial_wealth, "T": T, "n_sims": n_sims, "M": M}


@pytest.fixture
def simulation_data(accounts):
    """Create simulation data for tests."""
    T = 12
    n_sims = 100
    M = 2

    np.random.seed(42)
    A = np.full((n_sims, T), 500_000.0)
    R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
    initial_wealth = np.array([0.0, 0.0])

    return {"A": A, "R": R, "initial_wealth": initial_wealth, "T": T, "n_sims": n_sims, "M": M}


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


class TestOptimizationResultValidation:
    """Test OptimizationResult __post_init__ validation."""

    def test_X_must_be_2d(self, terminal_goals, sample_goal_set):
        """Test X must be 2D array."""
        X_1d = np.array([0.5, 0.5])

        with pytest.raises(ValueError, match="X must be 2D"):
            OptimizationResult(
                X=X_1d,
                T=24,
                objective_value=10_000_000.0,
                feasible=True,
                goals=terminal_goals,
                goal_set=sample_goal_set,
                solve_time=0.5,
            )

    def test_X_shape_must_match_T(self, terminal_goals, sample_goal_set):
        """Test X.shape[0] must equal T."""
        X = np.tile([0.5, 0.5], (12, 1))  # 12 != 24

        with pytest.raises(ValueError, match="X.shape"):
            OptimizationResult(
                X=X,
                T=24,
                objective_value=10_000_000.0,
                feasible=True,
                goals=terminal_goals,
                goal_set=sample_goal_set,
                solve_time=0.5,
            )

    def test_T_must_be_positive(self, terminal_goals, sample_goal_set):
        """Test T must be positive integer."""
        X = np.tile([0.5, 0.5], (0, 1))  # Empty

        with pytest.raises(ValueError, match="T must be positive"):
            OptimizationResult(
                X=X,
                T=0,
                objective_value=10_000_000.0,
                feasible=True,
                goals=terminal_goals,
                goal_set=sample_goal_set,
                solve_time=0.5,
            )

    def test_feasible_must_be_bool(self, terminal_goals, sample_goal_set):
        """Test feasible must be boolean."""
        X = np.tile([0.5, 0.5], (24, 1))

        with pytest.raises(TypeError, match="feasible must be bool"):
            OptimizationResult(
                X=X,
                T=24,
                objective_value=10_000_000.0,
                feasible=1,  # int instead of bool
                goals=terminal_goals,
                goal_set=sample_goal_set,
                solve_time=0.5,
            )

    def test_solve_time_must_be_non_negative(self, terminal_goals, sample_goal_set):
        """Test solve_time must be non-negative."""
        X = np.tile([0.5, 0.5], (24, 1))

        with pytest.raises(ValueError, match="solve_time must be non-negative"):
            OptimizationResult(
                X=X,
                T=24,
                objective_value=10_000_000.0,
                feasible=True,
                goals=terminal_goals,
                goal_set=sample_goal_set,
                solve_time=-0.5,
            )

    def test_goal_set_M_must_match_X(self, terminal_goals, accounts, start_date):
        """Test goal_set.M must match X.shape[1]."""
        # 3 accounts in X but 2 in goal_set
        X = np.tile([0.33, 0.33, 0.34], (24, 1))
        goal_set = GoalSet(terminal_goals, accounts, start_date)  # M=2

        with pytest.raises(ValueError, match="goal_set.M"):
            OptimizationResult(
                X=X,
                T=24,
                objective_value=10_000_000.0,
                feasible=True,
                goals=terminal_goals,
                goal_set=goal_set,
                solve_time=0.5,
            )


class TestOptimizationResultMethods:
    """Test OptimizationResult methods."""

    def test_summary_output(self, terminal_goals, sample_goal_set):
        """Test summary() returns formatted string."""
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

        summary = result.summary()

        assert "OptimizationResult" in summary
        assert "Feasible" in summary
        assert "T=24" in summary
        assert "10000000" in summary
        assert "0.500s" in summary

    def test_summary_infeasible(self, terminal_goals, sample_goal_set):
        """Test summary shows infeasible status."""
        X = np.tile([0.5, 0.5], (24, 1))

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=0.0,
            feasible=False,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.1,
        )

        summary = result.summary()
        assert "Infeasible" in summary

    def test_summary_with_diagnostics(self, terminal_goals, sample_goal_set):
        """Test summary includes diagnostics."""
        X = np.tile([0.5, 0.5], (24, 1))

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.5,
            diagnostics={"duality_gap": 1e-8, "convergence_status": "optimal"},
        )

        summary = result.summary()
        assert "Duality gap" in summary
        assert "Convergence" in summary

    def test_is_valid_allocation_valid(self, terminal_goals, sample_goal_set):
        """Test is_valid_allocation with valid allocation."""
        X = np.tile([0.6, 0.4], (24, 1))

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.5,
        )

        assert result.is_valid_allocation() is True

    def test_is_valid_allocation_negative_weights(self, terminal_goals, sample_goal_set):
        """Test is_valid_allocation with negative weights."""
        X = np.tile([0.6, 0.4], (24, 1))
        X[0, 0] = -0.1  # Invalid

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.5,
        )

        assert result.is_valid_allocation() is False

    def test_is_valid_allocation_wrong_sum(self, terminal_goals, sample_goal_set):
        """Test is_valid_allocation with wrong row sums."""
        X = np.tile([0.3, 0.3], (24, 1))  # Sums to 0.6, not 1.0

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=terminal_goals,
            goal_set=sample_goal_set,
            solve_time=0.5,
        )

        assert result.is_valid_allocation() is False

    def test_M_property(self, terminal_goals, sample_goal_set):
        """Test M property returns number of accounts."""
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

        assert result.M == 2

    def test_M_property_three_accounts(self, three_accounts, start_date):
        """Test M property with three accounts."""
        goals = [TerminalGoal(account="Aggressive", threshold=5_000_000, confidence=0.70)]
        goal_set = GoalSet(goals, three_accounts, start_date)
        X = np.tile([0.33, 0.33, 0.34], (24, 1))

        result = OptimizationResult(
            X=X,
            T=24,
            objective_value=10_000_000.0,
            feasible=True,
            goals=goals,
            goal_set=goal_set,
            solve_time=0.5,
        )

        assert result.M == 3


# ============================================================================
# ALLOCATIONOPTIMIZER BASE CLASS TESTS
# ============================================================================

class TestAllocationOptimizerInit:
    """Test AllocationOptimizer initialization via CVaROptimizer."""

    def test_n_accounts_must_be_positive(self):
        """Test n_accounts must be >= 1."""
        with pytest.raises(ValueError, match="n_accounts must be"):
            CVaROptimizer(n_accounts=0)

    def test_n_accounts_negative_raises(self):
        """Test negative n_accounts raises."""
        with pytest.raises(ValueError, match="n_accounts must be"):
            CVaROptimizer(n_accounts=-1)

    def test_account_names_length_mismatch(self):
        """Test account_names length must match n_accounts."""
        with pytest.raises(ValueError, match="account_names length"):
            CVaROptimizer(n_accounts=2, account_names=["A", "B", "C"])

    def test_default_account_names(self):
        """Test default account names generated."""
        optimizer = CVaROptimizer(n_accounts=3)

        assert len(optimizer.account_names) == 3
        assert optimizer.account_names[0] == "Account_0"

    def test_custom_account_names(self):
        """Test custom account names accepted."""
        optimizer = CVaROptimizer(n_accounts=2, account_names=["Savings", "Investment"])

        assert optimizer.account_names == ["Savings", "Investment"]

    def test_default_objective(self):
        """Test default objective is balanced."""
        optimizer = CVaROptimizer(n_accounts=2)

        assert optimizer.objective == "balanced"

    def test_objective_params_default(self):
        """Test objective_params defaults to empty dict."""
        optimizer = CVaROptimizer(n_accounts=2)

        assert optimizer.objective_params == {}

    def test_objective_params_custom(self):
        """Test custom objective params."""
        optimizer = CVaROptimizer(
            n_accounts=2,
            objective="conservative",
            objective_params={"lambda": 0.3}
        )

        assert optimizer.objective_params["lambda"] == 0.3


class TestAllocationOptimizerObjectives:
    """Test AllocationOptimizer objective functions."""

    @pytest.fixture
    def sample_wealth(self):
        """Create sample wealth array (n_sims=50, T+1=13, M=2)."""
        np.random.seed(42)
        W = np.random.uniform(1e6, 5e6, size=(50, 13, 2))
        return W

    @pytest.fixture
    def sample_allocation(self):
        """Create sample allocation (T=12, M=2)."""
        return np.tile([0.6, 0.4], (12, 1))

    def test_objective_risky(self, sample_wealth, sample_allocation):
        """Test risky objective computes expected terminal wealth."""
        optimizer = CVaROptimizer(n_accounts=2, objective="risky")

        result = optimizer._objective_risky(
            sample_wealth, sample_allocation, T=12, M=2
        )

        # Should be mean of total terminal wealth
        expected = sample_wealth[:, 12, :].sum(axis=1).mean()
        assert result == pytest.approx(expected)

    def test_objective_balanced(self, sample_wealth, sample_allocation):
        """Test balanced objective computes turnover penalty."""
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        result = optimizer._objective_balanced(
            sample_wealth, sample_allocation, T=12, M=2
        )

        # With constant allocation, turnover should be 0
        assert result == pytest.approx(0.0)

    def test_objective_balanced_with_varying_allocation(self, sample_wealth):
        """Test balanced objective with varying allocation."""
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        # Varying allocation
        X = np.zeros((12, 2))
        X[:6, :] = [0.3, 0.7]
        X[6:, :] = [0.7, 0.3]

        result = optimizer._objective_balanced(sample_wealth, X, T=12, M=2)

        # Should be negative (penalizes turnover)
        assert result < 0

    def test_objective_conservative(self, sample_wealth, sample_allocation):
        """Test conservative objective (mean-variance)."""
        optimizer = CVaROptimizer(
            n_accounts=2,
            objective="conservative",
            objective_params={"lambda": 0.5}
        )

        result = optimizer._objective_conservative(
            sample_wealth, sample_allocation, T=12, M=2
        )

        W_T_total = sample_wealth[:, 12, :].sum(axis=1)
        expected = W_T_total.mean() - 0.5 * W_T_total.std()
        assert result == pytest.approx(expected)

    def test_objective_risky_turnover(self, sample_wealth, sample_allocation):
        """Test risky_turnover objective."""
        optimizer = CVaROptimizer(
            n_accounts=2,
            objective="risky_turnover",
            objective_params={"lambda": 0.5}
        )

        result = optimizer._objective_risky_turnover(
            sample_wealth, sample_allocation, T=12, M=2
        )

        # With constant allocation, equals mean wealth
        W_T_total = sample_wealth[:, 12, :].sum(axis=1)
        expected = W_T_total.mean()
        assert result == pytest.approx(expected)

    def test_compute_objective_dispatches_risky(self, sample_wealth, sample_allocation):
        """Test _compute_objective dispatches to risky."""
        optimizer = CVaROptimizer(n_accounts=2, objective="risky")

        result = optimizer._compute_objective(
            sample_wealth, sample_allocation, T=12, M=2
        )

        expected = optimizer._objective_risky(
            sample_wealth, sample_allocation, T=12, M=2
        )
        assert result == expected

    def test_compute_objective_dispatches_balanced(self, sample_wealth, sample_allocation):
        """Test _compute_objective dispatches to balanced."""
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        result = optimizer._compute_objective(
            sample_wealth, sample_allocation, T=12, M=2
        )

        expected = optimizer._objective_balanced(
            sample_wealth, sample_allocation, T=12, M=2
        )
        assert result == expected

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

    def test_all_valid_objectives(self):
        """Test all valid objective types."""
        for objective in ["risky", "balanced", "conservative", "risky_turnover"]:
            optimizer = CVaROptimizer(n_accounts=2, objective=objective)
            assert optimizer.objective == objective


class TestCVaROptimizerSolve:
    """Test CVaROptimizer.solve() method."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer."""
        return CVaROptimizer(n_accounts=2, objective="balanced")

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

    def test_solve_with_warm_start(self, optimizer, simulation_data, terminal_goals, accounts, start_date):
        """Test solve with warm start allocation."""
        goal_set = GoalSet(terminal_goals, accounts, start_date)
        X_init = np.tile([0.5, 0.5], (simulation_data["T"], 1))

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
            X_init=X_init,
        )

        assert isinstance(result, OptimizationResult)

    def test_solve_with_initial_wealth(self, optimizer, terminal_goals, accounts, start_date):
        """Test solve with non-zero initial wealth."""
        T = 12
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 300_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([1_000_000.0, 500_000.0])

        goal_set = GoalSet(terminal_goals, accounts, start_date)

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)

    def test_solve_result_has_goal_set(self, optimizer, simulation_data, terminal_goals, accounts, start_date):
        """Test result contains goal_set reference."""
        goal_set = GoalSet(terminal_goals, accounts, start_date)

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert result.goal_set is goal_set

    def test_solve_result_has_goals(self, optimizer, simulation_data, terminal_goals, accounts, start_date):
        """Test result contains goals list."""
        goal_set = GoalSet(terminal_goals, accounts, start_date)

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert len(result.goals) == len(terminal_goals)

    def test_solve_short_horizon(self, optimizer, terminal_goals, accounts, start_date):
        """Test solve with short horizon T=3."""
        T = 3
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0.0, 0.0])

        # Reduce threshold for short horizon
        goals = [TerminalGoal(account="Aggressive", threshold=500_000, confidence=0.70)]
        goal_set = GoalSet(goals, accounts, start_date)

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        assert result.T == 3
        assert result.X.shape == (3, 2)


class TestCVaROptimizerWithIntermediateGoals:
    """Test CVaROptimizer with intermediate goals."""

    def test_solve_with_intermediate_goal(self, accounts, start_date):
        """Test solve with intermediate goal."""
        T = 12
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0.0, 0.0])

        goals = [
            IntermediateGoal(
                account="Conservative",
                threshold=1_000_000,
                confidence=0.70,
                date=date(2025, 7, 1)  # Month 6
            )
        ]
        goal_set = GoalSet(goals, accounts, start_date)
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)

    def test_solve_with_mixed_goals(self, accounts, start_date):
        """Test solve with both intermediate and terminal goals."""
        T = 12
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0.0, 0.0])

        goals = [
            IntermediateGoal(
                account="Conservative",
                threshold=1_000_000,
                confidence=0.70,
                date=date(2025, 7, 1)
            ),
            TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.70)
        ]
        goal_set = GoalSet(goals, accounts, start_date)
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        assert len(result.goals) == 2


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

    def test_T_max_default(self):
        """Test T_max defaults to 240."""
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer)

        assert seeker.T_max == 240

    def test_T_max_custom(self):
        """Test custom T_max."""
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer, T_max=120)

        assert seeker.T_max == 120

    def test_verbose_default(self):
        """Test verbose defaults to True."""
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer)

        assert seeker.verbose is True

    def test_verbose_false(self):
        """Test verbose can be set to False."""
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer, verbose=False)

        assert seeker.verbose is False


class TestGoalSeekerSeek:
    """Test GoalSeeker.seek() method."""

    @pytest.fixture
    def seeker(self):
        """Create GoalSeeker with short T_max for fast tests."""
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
        return GoalSeeker(optimizer, T_max=36, verbose=False)

    @pytest.fixture
    def generators(self, accounts):
        """Create A and R generators."""
        def A_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.full((n_sims, T), 500_000.0)

        def R_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.randn(n_sims, T, 2) * 0.02 + 0.005

        return A_generator, R_generator

    def test_seek_returns_result(self, seeker, generators, accounts, start_date):
        """Test seek returns OptimizationResult."""
        A_gen, R_gen = generators
        goals = [TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.60)]
        initial_wealth = np.array([0.0, 0.0])

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=initial_wealth,
            accounts=accounts,
            start_date=start_date,
            n_sims=50,
            seed=42,
        )

        assert isinstance(result, OptimizationResult)

    def test_seek_linear_search(self, seeker, generators, accounts, start_date):
        """Test seek with linear search method."""
        A_gen, R_gen = generators
        goals = [TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.60)]
        initial_wealth = np.array([0.0, 0.0])

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=initial_wealth,
            accounts=accounts,
            start_date=start_date,
            n_sims=50,
            seed=42,
            search_method="linear",
        )

        assert isinstance(result, OptimizationResult)

    def test_seek_binary_search(self, seeker, generators, accounts, start_date):
        """Test seek with binary search method."""
        A_gen, R_gen = generators
        goals = [TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.60)]
        initial_wealth = np.array([0.0, 0.0])

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=initial_wealth,
            accounts=accounts,
            start_date=start_date,
            n_sims=50,
            seed=42,
            search_method="binary",
        )

        assert isinstance(result, OptimizationResult)

    def test_seek_finds_minimum_horizon(self, generators, accounts, start_date):
        """Test seek finds minimum feasible horizon."""
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
        seeker = GoalSeeker(optimizer, T_max=24, verbose=False)

        A_gen, R_gen = generators
        # Low threshold should be achievable quickly
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        initial_wealth = np.array([0.0, 0.0])

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=initial_wealth,
            accounts=accounts,
            start_date=start_date,
            n_sims=50,
            seed=42,
        )

        # Should find a horizon less than T_max
        assert result.T <= 24
        assert result.feasible is True

    def test_seek_with_initial_wealth(self, seeker, generators, accounts, start_date):
        """Test seek with non-zero initial wealth."""
        A_gen, R_gen = generators
        goals = [TerminalGoal(account="Aggressive", threshold=2_000_000, confidence=0.60)]
        initial_wealth = np.array([500_000.0, 500_000.0])

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=initial_wealth,
            accounts=accounts,
            start_date=start_date,
            n_sims=50,
            seed=42,
        )

        assert isinstance(result, OptimizationResult)


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
        initial_wealth = np.array([0.0, 0.0])

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
        initial_wealth = np.array([0.0, 0.0])

        goals = [
            IntermediateGoal(date=date(2025, 7, 1), account="Conservative", threshold=1_000_000, confidence=0.60),
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

    def test_is_valid_allocation_after_solve(self, accounts):
        """Test result.is_valid_allocation() after solve."""
        T = 12
        n_sims = 50
        M = 2
        start_date = date(2025, 1, 1)

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0.0, 0.0])

        goals = [TerminalGoal(account="Aggressive", threshold=2_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date)

        optimizer = CVaROptimizer(n_accounts=M, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        assert result.is_valid_allocation() is True


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestOptimizationEdgeCases:
    """Test edge cases in optimization."""

    def test_single_account(self, start_date):
        """Test optimization with single account."""
        accounts = [Account.from_annual("Only", 0.08, 0.10)]
        T = 6
        n_sims = 50
        M = 1

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.008
        initial_wealth = np.array([0.0])

        goals = [TerminalGoal(account="Only", threshold=1_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date)

        optimizer = CVaROptimizer(n_accounts=1, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        # Single account should have all allocation = 1.0
        assert np.allclose(result.X, 1.0, atol=1e-5)

    def test_T_equals_1(self, accounts, start_date):
        """Test optimization with T=1."""
        T = 1
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([1_000_000.0, 1_000_000.0])

        # Very low threshold for T=1
        goals = [TerminalGoal(account="Aggressive", threshold=500_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date)

        optimizer = CVaROptimizer(n_accounts=M, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        assert result.T == 1
        assert result.X.shape == (1, 2)

    def test_uniform_allocation(self, accounts, start_date):
        """Test optimization produces valid allocation."""
        T = 6
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0.0, 0.0])

        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date)

        optimizer = CVaROptimizer(n_accounts=M, objective="balanced")

        result = optimizer.solve(
            T=T,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            goal_set=goal_set,
        )

        # All rows should sum to 1
        row_sums = result.X.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_stochastic_contributions(self, accounts, start_date):
        """Test optimization with stochastic contributions."""
        T = 6
        n_sims = 50
        M = 2

        np.random.seed(42)
        # Stochastic contributions
        A = np.random.uniform(400_000, 600_000, size=(n_sims, T))
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0.0, 0.0])

        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
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


class TestCheckFeasibility:
    """Test AllocationOptimizer._check_feasibility() method."""

    @pytest.fixture
    def portfolio(self, accounts):
        """Create portfolio."""
        return Portfolio(accounts)

    def test_feasible_allocation(self, accounts, start_date, portfolio):
        """Test feasibility check with valid allocation."""
        T = 6
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.01  # Higher returns
        initial_wealth = np.array([1_000_000.0, 1_000_000.0])

        # Easy goal (low threshold)
        goals = [TerminalGoal(account="Aggressive", threshold=500_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date)

        X = np.tile([0.3, 0.7], (T, 1))  # Favor aggressive account
        optimizer = CVaROptimizer(n_accounts=M)

        is_feasible = optimizer._check_feasibility(
            X=X,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            portfolio=portfolio,
            goal_set=goal_set,
        )

        assert is_feasible is True

    def test_infeasible_allocation_high_threshold(self, accounts, start_date, portfolio):
        """Test feasibility check with impossible threshold."""
        T = 6
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 100_000.0)  # Low contributions
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([0.0, 0.0])

        # Impossible goal
        goals = [TerminalGoal(account="Aggressive", threshold=100_000_000, confidence=0.95)]
        goal_set = GoalSet(goals, accounts, start_date)

        X = np.tile([0.5, 0.5], (T, 1))
        optimizer = CVaROptimizer(n_accounts=M)

        is_feasible = optimizer._check_feasibility(
            X=X,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            portfolio=portfolio,
            goal_set=goal_set,
        )

        assert is_feasible is False


# ============================================================================
# OPTIMIZATIONRESULT VALIDATE_GOALS TESTS
# ============================================================================

class TestOptimizationResultValidateGoals:
    """Test OptimizationResult.validate_goals() method."""

    @pytest.fixture
    def model(self):
        """Create a FinancialModel for simulation."""
        income = IncomeModel(
            fixed=FixedIncome(base=500_000, annual_growth=0.0),
            variable=None
        )
        accounts = [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]
        return FinancialModel(income, accounts)

    def test_validate_goals_with_matching_horizon(self, model):
        """Test validate_goals with matching simulation horizon."""
        T = 12
        start = date(2025, 1, 1)

        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, model.accounts, start)

        X = np.tile([0.3, 0.7], (T, 1))
        result_sim = model.simulate(T=T, n_sims=100, X=X, seed=42, start=start)

        opt_result = OptimizationResult(
            X=X,
            T=T,
            objective_value=10_000_000.0,
            feasible=True,
            goals=goals,
            goal_set=goal_set,
            solve_time=0.5,
        )

        metrics = opt_result.validate_goals(result_sim)

        assert isinstance(metrics, dict)
        assert len(metrics) == 1

    def test_validate_goals_horizon_mismatch_raises(self, model):
        """Test validate_goals raises when result.T != self.T."""
        start = date(2025, 1, 1)

        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, model.accounts, start)

        X = np.tile([0.3, 0.7], (24, 1))  # T=24
        result_sim = model.simulate(T=12, n_sims=100, X=np.tile([0.3, 0.7], (12, 1)), seed=42, start=start)  # T=12

        opt_result = OptimizationResult(
            X=X,
            T=24,  # Mismatch with result_sim.T=12
            objective_value=10_000_000.0,
            feasible=True,
            goals=goals,
            goal_set=goal_set,
            solve_time=0.5,
        )

        with pytest.raises(ValueError, match="horizon mismatch"):
            opt_result.validate_goals(result_sim)

    def test_validate_goals_with_intermediate_goals(self, model):
        """Test validate_goals with intermediate goals."""
        T = 12
        start = date(2025, 1, 1)

        goals = [
            IntermediateGoal(
                account="Conservative",
                threshold=500_000,
                confidence=0.60,
                date=date(2025, 6, 1)
            )
        ]
        goal_set = GoalSet(goals, model.accounts, start)

        X = np.tile([0.5, 0.5], (T, 1))
        result_sim = model.simulate(T=T, n_sims=100, X=X, seed=42, start=start)

        opt_result = OptimizationResult(
            X=X,
            T=T,
            objective_value=5_000_000.0,
            feasible=True,
            goals=goals,
            goal_set=goal_set,
            solve_time=0.5,
        )

        metrics = opt_result.validate_goals(result_sim)

        assert len(metrics) == 1


# ============================================================================
# CHECK FEASIBILITY WITH WITHDRAWALS TESTS
# ============================================================================

class TestCheckFeasibilityWithdrawals:
    """Test AllocationOptimizer._check_feasibility() with withdrawals."""

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

    def test_feasible_with_small_withdrawals(self, accounts, portfolio):
        """Test feasibility with small withdrawals."""
        T = 6
        n_sims = 50
        M = 2
        start_date_val = date(2025, 1, 1)

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.01
        initial_wealth = np.array([1_000_000.0, 1_000_000.0])

        # Small deterministic withdrawals (well within wealth bounds)
        D = np.zeros((T, M))
        D[3, 0] = 50_000.0  # Small withdrawal from conservative at t=3

        goals = [TerminalGoal(account="Aggressive", threshold=500_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date_val)

        X = np.tile([0.5, 0.5], (T, 1))
        optimizer = CVaROptimizer(n_accounts=M)

        is_feasible = optimizer._check_feasibility(
            X=X,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            portfolio=portfolio,
            goal_set=goal_set,
            D=D,
            withdrawal_epsilon=0.10,
        )

        assert is_feasible is True

    def test_infeasible_with_large_withdrawals(self, accounts, portfolio):
        """Test infeasibility with large withdrawals exceeding wealth."""
        T = 6
        n_sims = 50
        M = 2
        start_date_val = date(2025, 1, 1)

        np.random.seed(42)
        A = np.full((n_sims, T), 100_000.0)  # Small contributions
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.005
        initial_wealth = np.array([100_000.0, 100_000.0])

        # Large withdrawal that exceeds available wealth
        D = np.zeros((T, M))
        D[1, 0] = 500_000.0  # Huge withdrawal at t=1

        goals = [TerminalGoal(account="Aggressive", threshold=100_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date_val)

        X = np.tile([0.5, 0.5], (T, 1))
        optimizer = CVaROptimizer(n_accounts=M)

        is_feasible = optimizer._check_feasibility(
            X=X,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            portfolio=portfolio,
            goal_set=goal_set,
            D=D,
            withdrawal_epsilon=0.05,
        )

        assert is_feasible is False

    def test_feasibility_with_3d_withdrawals(self, accounts, portfolio):
        """Test feasibility with stochastic (3D) withdrawals."""
        T = 6
        n_sims = 50
        M = 2
        start_date_val = date(2025, 1, 1)

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.01
        initial_wealth = np.array([2_000_000.0, 2_000_000.0])

        # 3D stochastic withdrawals
        D = np.zeros((n_sims, T, M))
        D[:, 3, 0] = 100_000.0  # Same withdrawal across all sims

        goals = [TerminalGoal(account="Aggressive", threshold=500_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date_val)

        X = np.tile([0.5, 0.5], (T, 1))
        optimizer = CVaROptimizer(n_accounts=M)

        is_feasible = optimizer._check_feasibility(
            X=X,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            portfolio=portfolio,
            goal_set=goal_set,
            D=D,
            withdrawal_epsilon=0.10,
        )

        assert is_feasible is True

    def test_feasibility_max_row_deviation(self, accounts, portfolio):
        """Test feasibility check when allocation row sums deviate significantly."""
        T = 6
        n_sims = 50
        M = 2
        start_date_val = date(2025, 1, 1)

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.01
        initial_wealth = np.array([1_000_000.0, 1_000_000.0])

        goals = [TerminalGoal(account="Aggressive", threshold=500_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, start_date_val)

        # Allocation with significant deviation from simplex
        X = np.tile([0.3, 0.3], (T, 1))  # Sum is 0.6, deviation is 0.4 > 0.01
        optimizer = CVaROptimizer(n_accounts=M)

        is_feasible = optimizer._check_feasibility(
            X=X,
            A=A,
            R=R,
            initial_wealth=initial_wealth,
            portfolio=portfolio,
            goal_set=goal_set,
        )

        # Should return False because max_deviation > 0.01
        assert is_feasible is False


# ============================================================================
# GOALSEEKER ERROR HANDLING TESTS
# ============================================================================

class TestGoalSeekerErrors:
    """Test GoalSeeker error handling."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def generators(self):
        """Create A and R generators."""
        def A_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.full((n_sims, T), 500_000.0)

        def R_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.randn(n_sims, T, 2) * 0.02 + 0.005

        return A_generator, R_generator

    def test_seek_empty_goals_raises(self, generators, accounts):
        """Test seek raises ValueError for empty goals."""
        A_gen, R_gen = generators
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer, T_max=24, verbose=False)

        with pytest.raises(ValueError, match="goals list cannot be empty"):
            seeker.seek(
                goals=[],  # Empty goals
                A_generator=A_gen,
                R_generator=R_gen,
                initial_wealth=np.array([0.0, 0.0]),
                accounts=accounts,
                start_date=date(2025, 1, 1),
                n_sims=50,
            )

    def test_seek_empty_accounts_raises(self, generators):
        """Test seek raises ValueError for empty accounts."""
        A_gen, R_gen = generators
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer, T_max=24, verbose=False)

        goals = [TerminalGoal(account=0, threshold=1_000_000, confidence=0.60)]

        with pytest.raises(ValueError, match="accounts list cannot be empty"):
            seeker.seek(
                goals=goals,
                A_generator=A_gen,
                R_generator=R_gen,
                initial_wealth=np.array([0.0, 0.0]),
                accounts=[],  # Empty accounts
                start_date=date(2025, 1, 1),
                n_sims=50,
            )

    def test_seek_invalid_search_method_raises(self, generators, accounts):
        """Test seek raises ValueError for invalid search method."""
        A_gen, R_gen = generators
        optimizer = CVaROptimizer(n_accounts=2)
        seeker = GoalSeeker(optimizer, T_max=24, verbose=False)

        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]

        with pytest.raises(ValueError, match="Unknown search_method"):
            seeker.seek(
                goals=goals,
                A_generator=A_gen,
                R_generator=R_gen,
                initial_wealth=np.array([0.0, 0.0]),
                accounts=accounts,
                start_date=date(2025, 1, 1),
                n_sims=50,
                search_method="invalid",
            )


# ============================================================================
# GOALSEEKER BINARY SEARCH TESTS
# ============================================================================

class TestGoalSeekerBinarySearch:
    """Test GoalSeeker binary search specifics."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def generators(self):
        """Create A and R generators."""
        def A_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.full((n_sims, T), 500_000.0)

        def R_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.randn(n_sims, T, 2) * 0.02 + 0.008

        return A_generator, R_generator

    def test_binary_search_finds_minimum(self, generators, accounts):
        """Test binary search finds the minimum feasible horizon."""
        A_gen, R_gen = generators
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
        seeker = GoalSeeker(optimizer, T_max=24, verbose=False)

        # Low threshold - should be found quickly
        goals = [TerminalGoal(account="Aggressive", threshold=1_500_000, confidence=0.60)]

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=np.array([0.0, 0.0]),
            accounts=accounts,
            start_date=date(2025, 1, 1),
            n_sims=50,
            seed=42,
            search_method="binary",
        )

        assert result.feasible is True
        assert result.T <= 24

    def test_binary_search_with_warm_start(self, generators, accounts):
        """Test binary search uses warm start from previous solutions."""
        A_gen, R_gen = generators
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
        seeker = GoalSeeker(optimizer, T_max=24, verbose=False)

        goals = [TerminalGoal(account="Aggressive", threshold=2_000_000, confidence=0.60)]

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=np.array([0.0, 0.0]),
            accounts=accounts,
            start_date=date(2025, 1, 1),
            n_sims=50,
            seed=42,
            search_method="binary",
        )

        assert isinstance(result, OptimizationResult)
        assert result.feasible is True


# ============================================================================
# INFEASIBLE ERROR TESTS
# ============================================================================

class TestInfeasibleError:
    """Test InfeasibleError raising."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def generators(self):
        """Create A and R generators with small contributions."""
        def A_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.full((n_sims, T), 10_000.0)  # Very small contributions

        def R_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.randn(n_sims, T, 2) * 0.02 + 0.005

        return A_generator, R_generator

    def test_linear_search_raises_infeasible(self, generators, accounts):
        """Test linear search raises InfeasibleError when no solution exists."""
        A_gen, R_gen = generators
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
        seeker = GoalSeeker(optimizer, T_max=6, verbose=False)  # Very short T_max

        # Impossible goal with small contributions and short horizon
        goals = [TerminalGoal(account="Aggressive", threshold=100_000_000, confidence=0.95)]

        with pytest.raises(InfeasibleError, match="No feasible solution found"):
            seeker.seek(
                goals=goals,
                A_generator=A_gen,
                R_generator=R_gen,
                initial_wealth=np.array([0.0, 0.0]),
                accounts=accounts,
                start_date=date(2025, 1, 1),
                n_sims=50,
                seed=42,
                search_method="linear",
            )

    def test_binary_search_raises_infeasible(self, generators, accounts):
        """Test binary search raises InfeasibleError when no solution exists."""
        A_gen, R_gen = generators
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
        seeker = GoalSeeker(optimizer, T_max=6, verbose=False)  # Very short T_max

        # Impossible goal
        goals = [TerminalGoal(account="Aggressive", threshold=100_000_000, confidence=0.95)]

        with pytest.raises(InfeasibleError):
            seeker.seek(
                goals=goals,
                A_generator=A_gen,
                R_generator=R_gen,
                initial_wealth=np.array([0.0, 0.0]),
                accounts=accounts,
                start_date=date(2025, 1, 1),
                n_sims=50,
                seed=42,
                search_method="binary",
            )


# ============================================================================
# GOALSEEKER WITH WITHDRAWALS TESTS
# ============================================================================

class TestGoalSeekerWithWithdrawals:
    """Test GoalSeeker with withdrawal generators."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def generators(self):
        """Create A, R, and D generators."""
        def A_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.full((n_sims, T), 500_000.0)

        def R_generator(T, n_sims, seed=None):
            if seed is not None:
                np.random.seed(seed)
            return np.random.randn(n_sims, T, 2) * 0.02 + 0.008

        def D_generator(T, n_sims, seed=None):
            # Small deterministic withdrawals
            D = np.zeros((T, 2))
            if T > 5:
                D[5, 0] = 50_000.0  # Withdrawal at t=5 from first account
            return D

        return A_generator, R_generator, D_generator

    def test_seek_with_d_generator(self, generators, accounts):
        """Test seek with withdrawal generator."""
        A_gen, R_gen, D_gen = generators
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
        seeker = GoalSeeker(optimizer, T_max=24, verbose=False)

        goals = [TerminalGoal(account="Aggressive", threshold=2_000_000, confidence=0.60)]

        result = seeker.seek(
            goals=goals,
            A_generator=A_gen,
            R_generator=R_gen,
            initial_wealth=np.array([500_000.0, 500_000.0]),
            accounts=accounts,
            start_date=date(2025, 1, 1),
            n_sims=50,
            seed=42,
            D_generator=D_gen,
            withdrawal_epsilon=0.10,
        )

        assert isinstance(result, OptimizationResult)
        assert result.feasible is True


# ============================================================================
# CVAROPTIMIZER SOLVER SELECTION TESTS
# ============================================================================

class TestCVaROptimizerSolvers:
    """Test CVaROptimizer with different solvers."""

    @pytest.fixture
    def simulation_data(self):
        """Create simulation data."""
        T = 6
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.008
        initial_wealth = np.array([0.0, 0.0])

        return {"A": A, "R": R, "initial_wealth": initial_wealth, "T": T}

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    def test_solve_with_ecos_solver(self, simulation_data, accounts):
        """Test solve with ECOS solver (default)."""
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, date(2025, 1, 1))

        optimizer = CVaROptimizer(n_accounts=2, objective="balanced", solver="ECOS")

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)

    def test_solve_with_scs_solver(self, simulation_data, accounts):
        """Test solve with SCS solver."""
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, date(2025, 1, 1))

        optimizer = CVaROptimizer(n_accounts=2, objective="balanced", solver="SCS")

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)


# ============================================================================
# CVAROPTIMIZER RISKY TURNOVER OBJECTIVE TESTS
# ============================================================================

class TestCVaROptimizerRiskyTurnover:
    """Test CVaROptimizer with risky_turnover objective."""

    @pytest.fixture
    def simulation_data(self):
        """Create simulation data."""
        T = 12
        n_sims = 50
        M = 2

        np.random.seed(42)
        A = np.full((n_sims, T), 500_000.0)
        R = np.random.randn(n_sims, T, M) * 0.02 + 0.008
        initial_wealth = np.array([0.0, 0.0])

        return {"A": A, "R": R, "initial_wealth": initial_wealth, "T": T}

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    def test_solve_with_risky_turnover(self, simulation_data, accounts):
        """Test solve with risky_turnover objective."""
        goals = [TerminalGoal(account="Aggressive", threshold=2_000_000, confidence=0.60)]
        goal_set = GoalSet(goals, accounts, date(2025, 1, 1))

        optimizer = CVaROptimizer(
            n_accounts=2,
            objective="risky_turnover",
            objective_params={"lambda": 0.3}
        )

        result = optimizer.solve(
            T=simulation_data["T"],
            A=simulation_data["A"],
            R=simulation_data["R"],
            initial_wealth=simulation_data["initial_wealth"],
            goal_set=goal_set,
        )

        assert isinstance(result, OptimizationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
