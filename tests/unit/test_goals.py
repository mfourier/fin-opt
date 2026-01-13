"""
Unit tests for goals.py module.

Tests IntermediateGoal, TerminalGoal, and GoalSet classes.
"""

import pytest
import numpy as np
from datetime import date

from src.portfolio import Account
from src.goals import IntermediateGoal, TerminalGoal, GoalSet


# ============================================================================
# INTERMEDIATEGOAL TESTS
# ============================================================================

class TestIntermediateGoalInstantiation:
    """Test IntermediateGoal initialization."""

    def test_basic_instantiation_with_month(self):
        """Test basic IntermediateGoal with month."""
        goal = IntermediateGoal(
            month=6,
            account="Conservative",
            threshold=5_000_000,
            confidence=0.80
        )

        assert goal.month == 6
        assert goal.account == "Conservative"
        assert goal.threshold == 5_000_000
        assert goal.confidence == 0.80

    def test_instantiation_with_date(self):
        """Test IntermediateGoal with date."""
        goal = IntermediateGoal(
            date=date(2025, 7, 1),
            account="Emergency",
            threshold=3_000_000,
            confidence=0.90
        )

        assert goal.date == date(2025, 7, 1)
        assert goal.month is None

    def test_account_by_index(self):
        """Test IntermediateGoal with account index."""
        goal = IntermediateGoal(
            month=12,
            account=0,
            threshold=10_000_000,
            confidence=0.85
        )

        assert goal.account == 0

    def test_frozen_dataclass(self):
        """Test that IntermediateGoal is immutable."""
        goal = IntermediateGoal(
            month=6,
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
                month=6,
                account="Test",
                threshold=1_000_000,
                confidence=-0.1
            )

    def test_confidence_above_one_raises(self):
        """Test that confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            IntermediateGoal(
                month=6,
                account="Test",
                threshold=1_000_000,
                confidence=1.5
            )

    def test_negative_threshold_raises(self):
        """Test that negative threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            IntermediateGoal(
                month=6,
                account="Test",
                threshold=-1_000_000,
                confidence=0.80
            )

    def test_zero_month_raises(self):
        """Test that month <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="month"):
            IntermediateGoal(
                month=0,
                account="Test",
                threshold=1_000_000,
                confidence=0.80
            )


class TestIntermediateGoalResolveMonth:
    """Test IntermediateGoal.resolve_month() method."""

    def test_resolve_month_from_month(self):
        """Test resolve_month when month is provided."""
        goal = IntermediateGoal(
            month=6,
            account="Test",
            threshold=1_000_000,
            confidence=0.80
        )

        resolved = goal.resolve_month(start_date=date(2025, 1, 1))
        assert resolved == 6

    def test_resolve_month_from_date(self):
        """Test resolve_month when date is provided."""
        goal = IntermediateGoal(
            date=date(2025, 7, 1),
            account="Test",
            threshold=1_000_000,
            confidence=0.80
        )

        resolved = goal.resolve_month(start_date=date(2025, 1, 1))
        assert resolved == 6  # July is 6 months from January


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
                month=6,
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
            IntermediateGoal(month=6, account="Conservative", threshold=3_000_000, confidence=0.80),
            IntermediateGoal(month=12, account="Aggressive", threshold=5_000_000, confidence=0.85),
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
            IntermediateGoal(month=6, account="Conservative", threshold=3_000_000, confidence=0.80)
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert goal_set.T_min == 6

    def test_T_min_multiple_intermediate_returns_max(self, accounts, start_date):
        """Test T_min returns maximum month."""
        goals = [
            IntermediateGoal(month=6, account="Conservative", threshold=3_000_000, confidence=0.80),
            IntermediateGoal(month=12, account="Aggressive", threshold=5_000_000, confidence=0.85),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert goal_set.T_min == 12


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
            IntermediateGoal(month=6, account="Conservative", threshold=3_000_000, confidence=0.80),
            TerminalGoal(account="Aggressive", threshold=30_000_000, confidence=0.80),
        ]
        goal_set = GoalSet(goals, accounts, start_date)

        assert len(goal_set) == 2

    def test_goal_access(self, accounts, start_date):
        """Test accessing goals from GoalSet."""
        goals = [
            IntermediateGoal(month=6, account="Conservative", threshold=3_000_000, confidence=0.80),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
