"""
Unit tests for goals.py module.

Tests IntermediateGoal, TerminalGoal, and GoalSet classes.
"""

import pytest
import numpy as np
from datetime import date

from finopt.portfolio import Account
from finopt.goals import IntermediateGoal, TerminalGoal, GoalSet


# ============================================================================
# INTERMEDIATEGOAL TESTS
# ============================================================================

class TestIntermediateGoalInstantiation:
    """Test IntermediateGoal initialization."""

    def test_basic_instantiation(self):
        """Test basic IntermediateGoal with date."""
        goal = IntermediateGoal(
            date=date(2025, 7, 1),
            account="Conservative",
            threshold=5_000_000,
            confidence=0.80
        )

        assert goal.date == date(2025, 7, 1)
        assert goal.account == "Conservative"
        assert goal.threshold == 5_000_000
        assert goal.confidence == 0.80

    def test_account_by_index(self):
        """Test IntermediateGoal with account index."""
        goal = IntermediateGoal(
            date=date(2026, 1, 1),
            account=0,
            threshold=10_000_000,
            confidence=0.85
        )

        assert goal.account == 0

    def test_frozen_dataclass(self):
        """Test that IntermediateGoal is immutable."""
        goal = IntermediateGoal(
            date=date(2025, 7, 1),
            account="Test",
            threshold=1_000_000,
            confidence=0.80
        )
        with pytest.raises(Exception):
            goal.threshold = 2_000_000


class TestIntermediateGoalValidation:
    """Test IntermediateGoal validation."""

    def test_confidence_below_zero_raises(self):
        """Test that confidence < 0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            IntermediateGoal(
                date=date(2025, 7, 1),
                account="Test",
                threshold=1_000_000,
                confidence=-0.1
            )

    def test_confidence_above_one_raises(self):
        """Test that confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            IntermediateGoal(
                date=date(2025, 7, 1),
                account="Test",
                threshold=1_000_000,
                confidence=1.5
            )

    def test_negative_threshold_raises(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            IntermediateGoal(
                date=date(2025, 7, 1),
                account="Test",
                threshold=-1_000_000,
                confidence=0.80
            )


class TestIntermediateGoalResolveMonth:
    """Test IntermediateGoal.resolve_month() method."""

    def test_resolve_month_same_year(self):
        """Test resolve_month for date in same year."""
        goal = IntermediateGoal(
            date=date(2025, 7, 1),
            account="Test",
            threshold=1_000_000,
            confidence=0.80
        )

        resolved = goal.resolve_month(start_date=date(2025, 1, 1))
        assert resolved == 6  # July is 6 months from January

    def test_resolve_month_cross_year(self):
        """Test resolve_month for date crossing year boundary."""
        goal = IntermediateGoal(
            date=date(2026, 3, 1),
            account="Test",
            threshold=1_000_000,
            confidence=0.80
        )

        resolved = goal.resolve_month(start_date=date(2025, 1, 1))
        assert resolved == 14  # 12 months + 2 more

    def test_resolve_month_minimum_one(self):
        """Test resolve_month returns minimum 1."""
        goal = IntermediateGoal(
            date=date(2025, 1, 1),
            account="Test",
            threshold=1_000_000,
            confidence=0.80
        )

        resolved = goal.resolve_month(start_date=date(2025, 1, 1))
        assert resolved == 1  # Minimum is 1


# ============================================================================
# TERMINALGOAL TESTS
# ============================================================================

class TestTerminalGoalInstantiation:
    """Test TerminalGoal initialization."""

    def test_basic_instantiation(self):
        """Test basic TerminalGoal creation."""
        goal = TerminalGoal(
            account="Retirement",
            threshold=50_000_000,
            confidence=0.85
        )

        assert goal.account == "Retirement"
        assert goal.threshold == 50_000_000
        assert goal.confidence == 0.85

    def test_account_by_index(self):
        """Test TerminalGoal with account index."""
        goal = TerminalGoal(
            account=1,
            threshold=30_000_000,
            confidence=0.90
        )

        assert goal.account == 1

    def test_frozen_dataclass(self):
        """Test that TerminalGoal is immutable."""
        goal = TerminalGoal(
            account="Test",
            threshold=10_000_000,
            confidence=0.80
        )
        with pytest.raises(Exception):
            goal.threshold = 20_000_000


class TestTerminalGoalValidation:
    """Test TerminalGoal validation."""

    def test_confidence_below_zero_raises(self):
        """Test that confidence < 0 raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            TerminalGoal(
                account="Test",
                threshold=10_000_000,
                confidence=-0.1
            )

    def test_confidence_above_one_raises(self):
        """Test that confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            TerminalGoal(
                account="Test",
                threshold=10_000_000,
                confidence=1.5
            )

    def test_negative_threshold_raises(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            TerminalGoal(
                account="Test",
                threshold=-10_000_000,
                confidence=0.80
            )


# ============================================================================
# GOALSET TESTS
# ============================================================================

class TestGoalSetInstantiation:
    """Test GoalSet initialization."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def start_date(self):
        """Create test start date."""
        return date(2025, 1, 1)

    def test_basic_instantiation(self, accounts, start_date):
        """Test basic GoalSet creation."""
        goals = [
            TerminalGoal(account="Conservative", threshold=10_000_000, confidence=0.80)
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert len(goal_set) == 1
        assert goal_set.M == 2

    def test_with_intermediate_and_terminal(self, accounts, start_date):
        """Test GoalSet with both goal types."""
        goals = [
            IntermediateGoal(
                date=date(2025, 7, 1),
                account="Conservative",
                threshold=3_000_000,
                confidence=0.90
            ),
            TerminalGoal(
                account="Aggressive",
                threshold=30_000_000,
                confidence=0.80
            ),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert len(goal_set.intermediate_goals) == 1
        assert len(goal_set.terminal_goals) == 1

    def test_multiple_intermediate_goals(self, accounts, start_date):
        """Test GoalSet with multiple intermediate goals."""
        goals = [
            IntermediateGoal(date=date(2025, 7, 1), account="Conservative", threshold=3_000_000, confidence=0.80),
            IntermediateGoal(date=date(2026, 1, 1), account="Aggressive", threshold=5_000_000, confidence=0.85),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert len(goal_set.intermediate_goals) == 2


class TestGoalSetAccountResolution:
    """Test GoalSet account resolution."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def start_date(self):
        """Create test start date."""
        return date(2025, 1, 1)

    def test_resolve_by_name(self, accounts, start_date):
        """Test resolving account by name."""
        goals = [
            TerminalGoal(account="Aggressive", threshold=10_000_000, confidence=0.80)
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        idx = goal_set.get_account_index(goals[0])
        assert idx == 1

    def test_resolve_by_index(self, accounts, start_date):
        """Test resolving account by index."""
        goals = [
            TerminalGoal(account=0, threshold=10_000_000, confidence=0.80)
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        idx = goal_set.get_account_index(goals[0])
        assert idx == 0

    def test_invalid_account_name_raises(self, accounts, start_date):
        """Test that invalid account name raises ValueError."""
        goals = [
            TerminalGoal(account="NonExistent", threshold=10_000_000, confidence=0.80)
        ]
        with pytest.raises(ValueError, match="account"):
            GoalSet(goals, accounts, start_date)


class TestGoalSetTMin:
    """Test GoalSet T_min calculation."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def start_date(self):
        """Create test start date."""
        return date(2025, 1, 1)

    def test_T_min_no_intermediate_goals(self, accounts, start_date):
        """Test T_min returns 1 when no intermediate goals."""
        goals = [
            TerminalGoal(account="Conservative", threshold=10_000_000, confidence=0.80)
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert goal_set.T_min == 1

    def test_T_min_single_intermediate(self, accounts, start_date):
        """Test T_min equals intermediate goal month."""
        goals = [
            IntermediateGoal(date=date(2025, 7, 1), account="Conservative", threshold=3_000_000, confidence=0.80)
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert goal_set.T_min == 6  # July 1 is 6 months from Jan 1

    def test_T_min_multiple_intermediate_returns_max(self, accounts, start_date):
        """Test T_min returns maximum month."""
        goals = [
            IntermediateGoal(date=date(2025, 7, 1), account="Conservative", threshold=3_000_000, confidence=0.80),
            IntermediateGoal(date=date(2026, 1, 1), account="Aggressive", threshold=5_000_000, confidence=0.85),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert goal_set.T_min == 12  # Jan 2026 is 12 months from Jan 2025


class TestGoalSetIteration:
    """Test GoalSet iteration and access."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def start_date(self):
        """Create test start date."""
        return date(2025, 1, 1)

    def test_len(self, accounts, start_date):
        """Test len() returns total goal count."""
        goals = [
            IntermediateGoal(date=date(2025, 7, 1), account="Conservative", threshold=3_000_000, confidence=0.80),
            TerminalGoal(account="Aggressive", threshold=30_000_000, confidence=0.80),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert len(goal_set) == 2

    def test_goal_access(self, accounts, start_date):
        """Test accessing goals from GoalSet."""
        goals = [
            IntermediateGoal(date=date(2025, 7, 1), account="Conservative", threshold=3_000_000, confidence=0.80),
            TerminalGoal(account="Aggressive", threshold=30_000_000, confidence=0.80),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        # Access via properties
        assert len(goal_set.intermediate_goals) + len(goal_set.terminal_goals) == 2


class TestGoalSetEdgeCases:
    """Test GoalSet edge cases."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.04, 0.05),
            Account.from_annual("Aggressive", 0.14, 0.15),
        ]

    @pytest.fixture
    def start_date(self):
        """Create test start date."""
        return date(2025, 1, 1)

    def test_empty_goals_raises(self, accounts, start_date):
        """Test that empty goals list raises ValueError."""
        with pytest.raises(ValueError, match="goals list cannot be empty"):
            GoalSet([], accounts, start_date)

    def test_single_account_goals(self, start_date):
        """Test GoalSet with single account."""
        accounts = [Account.from_annual("Single", 0.08, 0.10)]
        goals = [
            TerminalGoal(account="Single", threshold=10_000_000, confidence=0.80)
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert goal_set.M == 1


# ============================================================================
# CHECK_GOALS DUAL METRICS TESTS
# ============================================================================

class TestCheckGoalsDualMetrics:
    """Test dual metric reporting in check_goals() for CVaR transparency."""

    @pytest.fixture
    def accounts(self):
        """Create test accounts."""
        return [
            Account.from_annual("Conservative", 0.08, 0.09, initial_wealth=0),
            Account.from_annual("Aggressive", 0.14, 0.15, initial_wealth=0)
        ]

    @pytest.fixture
    def start_date(self):
        """Create test start date."""
        return date(2025, 1, 1)

    def create_simulation_result(self, n_sims, T, M, wealth_values):
        """Helper to create SimulationResult with controlled wealth trajectories."""
        from finopt.model import SimulationResult

        # Create wealth array with specified values at final time
        wealth = np.zeros((n_sims, T + 1, M))
        wealth[:, -1, 0] = wealth_values  # Set terminal wealth for account 0

        # Dummy data for other required fields
        total_wealth = wealth.sum(axis=2)
        contributions = np.ones((n_sims, T))
        returns = np.zeros((n_sims, T, M))
        income = {
            "fixed": np.ones((n_sims, T)),
            "variable": np.zeros((n_sims, T)),
            "total": np.ones((n_sims, T))
        }
        allocation = np.array([[1.0, 0.0]] * T)  # All to first account

        return SimulationResult(
            wealth=wealth,
            total_wealth=total_wealth,
            contributions=contributions,
            returns=returns,
            income=income,
            allocation=allocation,
            withdrawals=None,
            T=T,
            n_sims=n_sims,
            M=M,
            start=date(2025, 1, 1),
            seed=42,
            account_names=["Conservative", "Aggressive"]
        )

    def test_dual_metrics_present(self, accounts, start_date):
        """Test that dual metrics are present in check_goals output."""
        from finopt.goals import check_goals

        # Create result where 90% of scenarios exceed threshold
        n_sims = 100
        wealth_values = np.concatenate([
            np.ones(90) * 6_000_000,   # 90 scenarios above threshold
            np.ones(10) * 4_000_000    # 10 scenarios below threshold
        ])
        result = self.create_simulation_result(n_sims, T=12, M=2, wealth_values=wealth_values)

        goals = [
            TerminalGoal(account="Conservative", threshold=5_000_000, confidence=0.80)
        ]

        status = check_goals(result, goals, accounts, start_date)

        # Verify new fields are present
        metrics = status[goals[0]]
        assert "empirical_probability" in metrics
        assert "confidence_gap" in metrics
        assert "note" in metrics

        # Verify backward compatibility (old fields still present)
        assert "satisfied" in metrics
        assert "violation_rate" in metrics
        assert "required_rate" in metrics
        assert "margin" in metrics
        assert "median_shortfall" in metrics
        assert "n_violations" in metrics

    def test_empirical_probability_computation(self, accounts, start_date):
        """Test that empirical_probability = 1 - violation_rate."""
        from finopt.goals import check_goals

        # Create result with known violation rate (20%)
        n_sims = 100
        wealth_values = np.concatenate([
            np.ones(80) * 6_000_000,   # 80 scenarios above threshold
            np.ones(20) * 4_000_000    # 20 scenarios below threshold
        ])
        result = self.create_simulation_result(n_sims, T=12, M=2, wealth_values=wealth_values)

        goals = [
            TerminalGoal(account="Conservative", threshold=5_000_000, confidence=0.85)
        ]

        status = check_goals(result, goals, accounts, start_date)
        metrics = status[goals[0]]

        # Empirical probability should be 80% (1 - 0.20)
        assert abs(metrics["empirical_probability"] - 0.80) < 1e-10
        assert abs(metrics["violation_rate"] - 0.20) < 1e-10
        assert abs(metrics["empirical_probability"] - (1.0 - metrics["violation_rate"])) < 1e-10

    def test_confidence_gap_computation(self, accounts, start_date):
        """Test that confidence_gap = empirical_probability - specified_confidence."""
        from finopt.goals import check_goals

        # Create result with 92% success rate
        n_sims = 100
        wealth_values = np.concatenate([
            np.ones(92) * 6_000_000,   # 92 scenarios above threshold
            np.ones(8) * 4_000_000     # 8 scenarios below threshold
        ])
        result = self.create_simulation_result(n_sims, T=12, M=2, wealth_values=wealth_values)

        # Goal specifies 85% confidence
        goals = [
            TerminalGoal(account="Conservative", threshold=5_000_000, confidence=0.85)
        ]

        status = check_goals(result, goals, accounts, start_date)
        metrics = status[goals[0]]

        # Confidence gap should be +7% (0.92 - 0.85)
        expected_gap = 0.92 - 0.85
        assert abs(metrics["confidence_gap"] - expected_gap) < 1e-10

    def test_significant_conservatism_note(self, accounts, start_date):
        """Test note generation for significant conservatism (gap > 1%)."""
        from finopt.goals import check_goals

        # Create result with 95% success rate (10% gap from 85% specified)
        n_sims = 100
        wealth_values = np.concatenate([
            np.ones(95) * 6_000_000,
            np.ones(5) * 4_000_000
        ])
        result = self.create_simulation_result(n_sims, T=12, M=2, wealth_values=wealth_values)

        goals = [
            TerminalGoal(account="Conservative", threshold=5_000_000, confidence=0.85)
        ]

        status = check_goals(result, goals, accounts, start_date)
        metrics = status[goals[0]]

        # Note should mention CVaR conservatism
        assert "CVaR optimization yields conservative estimates" in metrics["note"]
        assert "safety margin" in metrics["note"]

    def test_mild_conservatism_note(self, accounts, start_date):
        """Test note generation for mild conservatism (gap < 1%)."""
        from finopt.goals import check_goals

        # Create result with 85.5% success rate (0.5% gap from 85% specified)
        n_sims = 200
        wealth_values = np.concatenate([
            np.ones(171) * 6_000_000,   # 85.5% above
            np.ones(29) * 4_000_000
        ])
        result = self.create_simulation_result(n_sims, T=12, M=2, wealth_values=wealth_values)

        goals = [
            TerminalGoal(account="Conservative", threshold=5_000_000, confidence=0.85)
        ]

        status = check_goals(result, goals, accounts, start_date)
        metrics = status[goals[0]]

        # Note should be simpler for mild conservatism
        assert "CVaR constraint satisfied" in metrics["note"]
        assert "safety margin" not in metrics["note"]

    def test_violation_warning_note(self, accounts, start_date):
        """Test note generation for violation case (negative gap)."""
        from finopt.goals import check_goals

        # Create result with 80% success rate (5% below 85% specified)
        n_sims = 100
        wealth_values = np.concatenate([
            np.ones(80) * 6_000_000,
            np.ones(20) * 4_000_000
        ])
        result = self.create_simulation_result(n_sims, T=12, M=2, wealth_values=wealth_values)

        goals = [
            TerminalGoal(account="Conservative", threshold=5_000_000, confidence=0.85)
        ]

        status = check_goals(result, goals, accounts, start_date)
        metrics = status[goals[0]]

        # Note should warn about violation
        assert "Warning" in metrics["note"]
        assert "below specified confidence" in metrics["note"]
        assert not metrics["satisfied"]

    def test_intermediate_goal_dual_metrics(self, accounts, start_date):
        """Test dual metrics for IntermediateGoal."""
        from finopt.goals import check_goals

        # Create result with wealth at month 6
        n_sims = 100
        T = 12
        M = 2
        wealth = np.zeros((n_sims, T + 1, M))

        # Set wealth at month 6 (index 6 in 0-indexed array)
        wealth[:85, 6, 0] = 3_000_000  # 85 scenarios above threshold
        wealth[85:, 6, 0] = 1_500_000  # 15 scenarios below threshold

        from finopt.model import SimulationResult
        result = SimulationResult(
            wealth=wealth,
            total_wealth=wealth.sum(axis=2),
            contributions=np.ones((n_sims, T)),
            returns=np.zeros((n_sims, T, M)),
            income={
                "fixed": np.ones((n_sims, T)),
                "variable": np.zeros((n_sims, T)),
                "total": np.ones((n_sims, T))
            },
            allocation=np.array([[1.0, 0.0]] * T),
            withdrawals=None,
            T=T,
            n_sims=n_sims,
            M=M,
            start=start_date,
            seed=42,
            account_names=["Conservative", "Aggressive"]
        )

        # Goal at month 6 (July 1, 2025)
        goals = [
            IntermediateGoal(
                date=date(2025, 7, 1),
                account="Conservative",
                threshold=2_000_000,
                confidence=0.80
            )
        ]

        status = check_goals(result, goals, accounts, start_date)
        metrics = status[goals[0]]

        # Should have 85% empirical probability
        assert abs(metrics["empirical_probability"] - 0.85) < 1e-10
        assert abs(metrics["confidence_gap"] - 0.05) < 1e-10
        assert metrics["satisfied"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
