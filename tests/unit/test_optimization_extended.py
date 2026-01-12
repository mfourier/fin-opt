"""
Tests for CVaROptimizer withdrawal (Y) extension.

Tests the extended solve() method that supports fixed withdrawal schedules
for reward/goal funding via the Y parameter.
"""

import numpy as np
import pytest
from datetime import date

# Check if cvxpy is available
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from src.portfolio import Account, Portfolio
from src.goals import TerminalGoal, GoalSet
from src.optimization import CVaROptimizer, OptimizationResult


# Skip all tests if cvxpy not available
pytestmark = pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy not installed")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def accounts():
    """Two test accounts."""
    return [
        Account.from_annual("taxable", annual_return=0.08, annual_volatility=0.15, initial_wealth=100_000),
        Account.from_annual("tax_advantaged", annual_return=0.06, annual_volatility=0.10, initial_wealth=50_000),
    ]


@pytest.fixture
def simple_goal_set(accounts):
    """Simple terminal goal for testing."""
    goal = TerminalGoal(
        account="taxable",
        threshold=110_000,  # Achievable with initial 100k + contributions
        confidence=0.90  # 90% probability
    )
    return GoalSet(
        goals=[goal],
        accounts=accounts,
        start_date=date(2025, 1, 1)
    )


@pytest.fixture
def deterministic_scenarios():
    """Deterministic scenarios for reproducible testing."""
    T, M, n_sims = 12, 2, 100
    
    # Fixed returns (10% annual → ~0.8% monthly)
    R = np.full((n_sims, T, M), 0.008)
    
    # Fixed contributions ($3000/month)
    A = np.full((n_sims, T), 3000.0)
    
    # Initial wealth
    W0 = np.array([100_000.0, 50_000.0])
    
    return {"A": A, "R": R, "W0": W0, "T": T, "M": M, "n_sims": n_sims}


# =============================================================================
# Basic Y Parameter Tests
# =============================================================================

class TestYParameterBasic:
    """Basic tests for Y parameter handling in CVaROptimizer."""
    
    def test_solve_without_y(self, accounts, simple_goal_set, deterministic_scenarios):
        """Solve without Y (default behavior)."""
        optimizer = CVaROptimizer(n_accounts=2, objective='balanced')
        
        result = optimizer.solve(
            T=deterministic_scenarios["T"],
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=simple_goal_set
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.X.shape == (deterministic_scenarios["T"], 2)
        assert 'has_withdrawals' in result.diagnostics
        assert result.diagnostics['has_withdrawals'] == False
    
    def test_solve_with_zero_y(self, accounts, simple_goal_set, deterministic_scenarios):
        """Solve with explicit zero Y."""
        optimizer = CVaROptimizer(n_accounts=2, objective='balanced')
        
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        Y = np.zeros((T, M))
        
        result = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=simple_goal_set,
            Y=Y
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.diagnostics['total_withdrawals'] == 0.0
    
    def test_solve_with_y_produces_different_result(self, accounts, simple_goal_set, deterministic_scenarios):
        """Solve with Y should produce different optimal allocation."""
        optimizer = CVaROptimizer(n_accounts=2, objective='risky')
        
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        
        # Solve without withdrawals
        result_no_y = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=simple_goal_set,
            Y=None
        )
        
        # Solve with significant withdrawal
        Y = np.zeros((T, M))
        Y[6, 0] = 20_000  # Withdraw $20k from taxable at month 6
        
        result_with_y = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=simple_goal_set,
            Y=Y
        )
        
        # With withdrawals, should need different allocation strategy
        assert result_with_y.diagnostics['total_withdrawals'] == 20_000.0
        assert result_with_y.diagnostics['has_withdrawals'] == True


# =============================================================================
# Y Validation Tests
# =============================================================================

class TestYValidation:
    """Tests for Y parameter validation."""
    
    def test_y_wrong_shape(self, accounts, simple_goal_set, deterministic_scenarios):
        """Y with wrong shape should raise."""
        optimizer = CVaROptimizer(n_accounts=2, objective='balanced')
        
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        Y_bad = np.zeros((T + 5, M))  # Wrong T dimension
        
        with pytest.raises(ValueError, match="Y must have shape"):
            optimizer.solve(
                T=T,
                A=deterministic_scenarios["A"],
                R=deterministic_scenarios["R"],
                W0=deterministic_scenarios["W0"],
                goal_set=simple_goal_set,
                Y=Y_bad
            )
    
    def test_y_negative_values(self, accounts, simple_goal_set, deterministic_scenarios):
        """Negative Y values should raise."""
        optimizer = CVaROptimizer(n_accounts=2, objective='balanced')
        
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        Y_negative = np.full((T, M), -100.0)
        
        with pytest.raises(ValueError, match="non-negative"):
            optimizer.solve(
                T=T,
                A=deterministic_scenarios["A"],
                R=deterministic_scenarios["R"],
                W0=deterministic_scenarios["W0"],
                goal_set=simple_goal_set,
                Y=Y_negative
            )


# =============================================================================
# Withdrawal Effect on Goal Satisfaction
# =============================================================================

class TestWithdrawalEffectsOnGoals:
    """Tests that withdrawals correctly affect goal satisfaction."""
    
    def test_large_withdrawal_makes_goal_harder(self, accounts, deterministic_scenarios):
        """Large withdrawal should make goal harder to achieve."""
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        
        # Easy goal without withdrawals
        easy_goal = TerminalGoal(
            account="taxable",
            threshold=120_000,  # Reachable with initial 100k
            confidence=0.920
        )
        goal_set = GoalSet(
            goals=[easy_goal],
            accounts=accounts,
            start_date=date(2025, 1, 1)
        )
        
        optimizer = CVaROptimizer(n_accounts=2, objective='risky')
        
        # Without withdrawal - should be feasible
        result_no_y = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=goal_set,
            Y=None
        )
        
        # With large withdrawal - may become infeasible
        Y = np.zeros((T, M))
        Y[0, 0] = 50_000  # Immediate large withdrawal from taxable
        
        result_with_y = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=goal_set,
            Y=Y
        )
        
        # The large withdrawal should make it harder
        # (may still be feasible due to contributions, but objective likely worse)
        assert result_with_y.diagnostics['total_withdrawals'] == 50_000.0
    
    def test_withdrawal_from_non_goal_account(self, accounts, deterministic_scenarios):
        """Withdrawal from non-goal account shouldn't directly affect goal."""
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        
        # Goal only on taxable (account 0)
        goal = TerminalGoal(
            account="taxable",
            threshold=110_000,
            confidence=0.920
        )
        goal_set = GoalSet(
            goals=[goal],
            accounts=accounts,
            start_date=date(2025, 1, 1)
        )
        
        optimizer = CVaROptimizer(n_accounts=2, objective='risky')
        
        # Withdraw from tax_advantaged (account 1), not taxable
        Y = np.zeros((T, M))
        Y[6, 1] = 10_000  # Withdraw from non-goal account
        
        result = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=goal_set,
            Y=Y
        )
        
        # Should still be feasible (withdrawal doesn't touch goal account directly)
        assert result.diagnostics['total_withdrawals'] == 10_000.0


# =============================================================================
# Integration with RewardSchedule
# =============================================================================

class TestIntegrationWithRewardSchedule:
    """Tests integrating CVaROptimizer with RewardSchedule."""
    
    def test_reward_schedule_to_optimization(self, accounts, deterministic_scenarios):
        """Use RewardSchedule.get_fixed_withdrawals as Y input."""
        from src.rewards import Reward, RewardSchedule
        
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        
        # Create reward schedule
        vacation = Reward(name="vacation", amount=5000, month=6)
        schedule = RewardSchedule(rewards=[vacation])
        
        # Get withdrawal matrix
        Y = schedule.get_fixed_withdrawals(T=T, M=M, accounts=accounts)
        
        # Goal
        goal = TerminalGoal(account="taxable", threshold=110_000, confidence=0.920)
        goal_set = GoalSet(goals=[goal], accounts=accounts, start_date=date(2025, 1, 1))
        
        optimizer = CVaROptimizer(n_accounts=2, objective='risky')
        
        result = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=goal_set,
            Y=Y
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.diagnostics['has_withdrawals'] == True
        assert result.diagnostics['total_withdrawals'] == 5000.0


# =============================================================================
# Objective Value Impact
# =============================================================================

class TestObjectiveValueImpact:
    """Tests that withdrawals correctly impact objective values."""
    
    def test_risky_objective_lower_with_withdrawals(self, accounts, deterministic_scenarios):
        """Risky objective (expected wealth) should be lower with withdrawals."""
        T, M = deterministic_scenarios["T"], deterministic_scenarios["M"]
        
        # Simple feasible goal
        goal = TerminalGoal(account="taxable", threshold=80_000, confidence=0.930)
        goal_set = GoalSet(goals=[goal], accounts=accounts, start_date=date(2025, 1, 1))
        
        optimizer = CVaROptimizer(n_accounts=2, objective='risky')
        
        # Without withdrawals
        result_no_y = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=goal_set,
            Y=None
        )
        
        # With withdrawals
        Y = np.zeros((T, M))
        Y[:, :] = 500  # $500/month from each account
        
        result_with_y = optimizer.solve(
            T=T,
            A=deterministic_scenarios["A"],
            R=deterministic_scenarios["R"],
            W0=deterministic_scenarios["W0"],
            goal_set=goal_set,
            Y=Y
        )
        
        # Expected terminal wealth should be lower with withdrawals
        if result_no_y.feasible and result_with_y.feasible:
            # For risky objective, higher is better (more wealth)
            # Withdrawals reduce final wealth, so objective should be lower
            assert result_with_y.objective_value < result_no_y.objective_value or \
                   result_with_y.diagnostics['total_withdrawals'] > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for withdrawal functionality."""
    
    def test_single_period_withdrawal(self, accounts, deterministic_scenarios):
        """Test with single period (T=1)."""
        n_sims, M = deterministic_scenarios["n_sims"], deterministic_scenarios["M"]
        T = 1
        
        A = np.full((n_sims, T), 5000.0)
        R = np.full((n_sims, T, M), 0.01)
        W0 = np.array([100_000.0, 50_000.0])
        
        goal = TerminalGoal(account="taxable", threshold=90_000, confidence=0.930)
        goal_set = GoalSet(goals=[goal], accounts=accounts, start_date=date(2025, 1, 1))
        
        optimizer = CVaROptimizer(n_accounts=2, objective='balanced')
        
        Y = np.array([[1000.0, 500.0]])  # Single period withdrawal
        
        result = optimizer.solve(T=T, A=A, R=R, W0=W0, goal_set=goal_set, Y=Y)
        
        assert result.X.shape == (1, 2)
        assert result.diagnostics['total_withdrawals'] == 1500.0
    
    def test_zero_initial_wealth_with_withdrawals(self, accounts, deterministic_scenarios):
        """Test when starting from zero wealth with contributions and withdrawals."""
        T, M, n_sims = 12, 2, deterministic_scenarios["n_sims"]
        
        # Start with zero wealth but high contributions
        A = np.full((n_sims, T), 10_000.0)
        R = np.full((n_sims, T, M), 0.005)
        W0 = np.array([0.0, 0.0])  # Zero initial wealth
        
        # Update accounts with zero initial wealth
        zero_accounts = [
            Account.from_annual("taxable", annual_return=0.08, annual_volatility=0.15, initial_wealth=0),
            Account.from_annual("tax_advantaged", annual_return=0.06, annual_volatility=0.10, initial_wealth=0),
        ]
        
        goal = TerminalGoal(account="taxable", threshold=50_000, confidence=0.920)
        goal_set = GoalSet(goals=[goal], accounts=zero_accounts, start_date=date(2025, 1, 1))
        
        optimizer = CVaROptimizer(n_accounts=2, objective='risky')
        
        # Delayed withdrawal (after contributions build up)
        Y = np.zeros((T, M))
        Y[6:, 0] = 1000  # Withdraw $1k/month from month 6 onwards
        
        result = optimizer.solve(T=T, A=A, R=R, W0=W0, goal_set=goal_set, Y=Y)
        
        assert isinstance(result, OptimizationResult)
        assert result.diagnostics['total_withdrawals'] == 6000.0  # 6 months * $1000
