"""
Unit tests for rewards.py module.

Tests validation, scheduling, and policy logic for Reward and RewardSchedule.
"""

import pytest
import numpy as np
from datetime import date
from dataclasses import FrozenInstanceError

from src.rewards import Reward, RewardSchedule
from src.portfolio import Account


class TestReward:
    """Tests for Reward class."""

    def test_basic_creation(self):
        """Basic reward creation."""
        r = Reward(name="Laptop", amount=1_000_000, month=6)
        
        assert r.name == "Laptop"
        assert r.amount == 1_000_000
        assert r.month == 6
        assert r.optional is False
        assert r.priority == 1
        assert r.account is None

    def test_optional_reward(self):
        """Optional reward with priority."""
        r = Reward(
            name="Vacation",
            amount=2_000_000,
            month=12,
            optional=True,
            priority=3
        )
        
        assert r.optional is True
        assert r.priority == 3

    def test_specific_account(self):
        """Reward with specific account reference."""
        r_int = Reward(name="Test", amount=100_000, month=3, account=0)
        r_str = Reward(name="Test", amount=100_000, month=3, account="Cash")
        
        assert r_int.account == 0
        assert r_str.account == "Cash"

    def test_frozen_dataclass(self):
        """Reward is immutable."""
        r = Reward(name="Laptop", amount=1_000_000, month=6)
        with pytest.raises(FrozenInstanceError):
            r.amount = 2_000_000

    def test_negative_amount_validation(self):
        """Negative amount raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            Reward(name="Bad", amount=-100_000, month=6)

    def test_invalid_month_validation(self):
        """Month < 1 raises error."""
        with pytest.raises(ValueError, match="month must be >= 1"):
            Reward(name="Bad", amount=100_000, month=0)

    def test_invalid_priority_validation(self):
        """Priority < 1 raises error."""
        with pytest.raises(ValueError, match="priority must be >= 1"):
            Reward(name="Bad", amount=100_000, month=6, priority=0)


class TestRewardSchedule:
    """Tests for RewardSchedule class."""

    @pytest.fixture
    def sample_accounts(self):
        """Sample accounts for testing."""
        return [
            Account.from_annual("Cash", 0.03, 0.02, initial_wealth=5_000_000),
            Account.from_annual("Growth", 0.10, 0.15, initial_wealth=1_000_000),
        ]

    @pytest.fixture
    def sample_rewards(self):
        """Sample rewards list."""
        return [
            Reward(name="Laptop", amount=1_000_000, month=3),
            Reward(name="Vacation", amount=2_000_000, month=6),
            Reward(name="Car Upgrade", amount=500_000, month=9, optional=True, priority=2),
        ]

    def test_basic_creation(self, sample_rewards):
        """Basic schedule creation."""
        schedule = RewardSchedule(
            rewards=sample_rewards,
            withdrawal_policy="proportional"
        )
        
        assert len(schedule.rewards) == 3
        assert schedule.withdrawal_policy == "proportional"

    def test_empty_rewards_validation(self):
        """Empty rewards list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            RewardSchedule(rewards=[])

    def test_single_account_requires_default(self, sample_rewards):
        """Single account policy requires default_account."""
        with pytest.raises(ValueError, match="requires default_account"):
            RewardSchedule(
                rewards=sample_rewards,
                withdrawal_policy="single_account",
                default_account=None
            )

    def test_get_mandatory_rewards(self, sample_rewards):
        """Get mandatory rewards correctly."""
        schedule = RewardSchedule(rewards=sample_rewards)
        
        mandatory = schedule.get_mandatory_rewards()
        assert len(mandatory) == 2
        assert all(not r.optional for r in mandatory)

    def test_get_optional_rewards(self, sample_rewards):
        """Get optional rewards correctly."""
        schedule = RewardSchedule(rewards=sample_rewards)
        
        optional = schedule.get_optional_rewards()
        assert len(optional) == 1
        assert optional[0].name == "Car Upgrade"

    def test_total_mandatory_amount(self, sample_rewards):
        """Total mandatory amount calculated correctly."""
        schedule = RewardSchedule(rewards=sample_rewards)
        
        total = schedule.total_mandatory_amount()
        assert total == 1_000_000 + 2_000_000  # Laptop + Vacation

    def test_fixed_withdrawals_single_account(self, sample_accounts):
        """Fixed withdrawals with single_account policy."""
        rewards = [
            Reward(name="Laptop", amount=1_000_000, month=3),
            Reward(name="Vacation", amount=2_000_000, month=6),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        assert Y.shape == (12, 2)
        assert Y[2, 0] == 1_000_000  # Month 3 (0-indexed: 2), account 0
        assert Y[5, 0] == 2_000_000  # Month 6 (0-indexed: 5), account 0
        assert Y[:, 1].sum() == 0  # No withdrawals from account 1

    def test_fixed_withdrawals_proportional(self, sample_accounts):
        """Fixed withdrawals with proportional policy."""
        rewards = [
            Reward(name="Laptop", amount=1_000_000, month=3),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="proportional"
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        # Equal split across 2 accounts
        assert Y[2, 0] == 500_000
        assert Y[2, 1] == 500_000
        assert Y.sum() == 1_000_000

    def test_fixed_withdrawals_priority(self, sample_accounts):
        """Fixed withdrawals with priority policy (lowest volatility first)."""
        rewards = [
            Reward(name="Laptop", amount=1_000_000, month=3),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="priority"
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        # Cash has lower volatility (0.02) than Growth (0.15)
        assert Y[2, 0] == 1_000_000  # From Cash
        assert Y[2, 1] == 0  # Not from Growth

    def test_fixed_withdrawals_excludes_optional(self, sample_rewards, sample_accounts):
        """Optional rewards not included in fixed withdrawals."""
        schedule = RewardSchedule(
            rewards=sample_rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        # Optional Car Upgrade at month 9 should NOT be included
        assert Y[8, 0] == 0  # Month 9 (0-indexed: 8)
        # Only Laptop and Vacation
        assert Y.sum() == 1_000_000 + 2_000_000

    def test_fixed_withdrawals_outside_horizon(self, sample_accounts):
        """Rewards outside horizon are ignored."""
        rewards = [
            Reward(name="Early", amount=1_000_000, month=3),
            Reward(name="Late", amount=2_000_000, month=24),  # Beyond T=12
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        assert Y.sum() == 1_000_000  # Only Early included

    def test_fixed_withdrawals_accumulate(self, sample_accounts):
        """Multiple rewards at same month accumulate."""
        rewards = [
            Reward(name="Item1", amount=500_000, month=6),
            Reward(name="Item2", amount=300_000, month=6),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        assert Y[5, 0] == 800_000  # Both accumulated

    def test_specific_account_override(self, sample_accounts):
        """Reward-specific account overrides default."""
        rewards = [
            Reward(name="From Growth", amount=1_000_000, month=3, account=1),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0  # Default is Cash
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        assert Y[2, 0] == 0  # Not from default (Cash)
        assert Y[2, 1] == 1_000_000  # From Growth (override)

    def test_account_name_resolution(self, sample_accounts):
        """Account resolution by name works."""
        rewards = [
            Reward(name="Test", amount=1_000_000, month=3, account="Growth"),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account="Cash"
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        assert Y[2, 1] == 1_000_000  # Growth is index 1

    def test_to_optimization_params(self, sample_rewards, sample_accounts):
        """Optimization parameters generated correctly."""
        schedule = RewardSchedule(
            rewards=sample_rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        params = schedule.to_optimization_params(T=12, M=2, accounts=sample_accounts)
        
        assert "Y_fixed" in params
        assert "mandatory_months" in params
        assert "optional_rewards" in params
        assert "policy" in params
        
        assert params["Y_fixed"].shape == (12, 2)
        assert len(params["mandatory_months"]) == 2  # Laptop, Vacation
        assert len(params["optional_rewards"]) == 1  # Car Upgrade
        assert params["policy"] == "single_account"

    def test_validate_liquidity_feasible(self, sample_accounts):
        """Liquidity validation passes with sufficient wealth."""
        rewards = [
            Reward(name="Small", amount=100_000, month=3),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        # Wealth always >= 5M (Cash initial)
        W = np.full((10, 13, 2), 5_000_000)  # (n_sims, T+1, M)
        
        assert schedule.validate_liquidity(W, Y) is True

    def test_validate_liquidity_infeasible(self, sample_accounts):
        """Liquidity validation fails with insufficient wealth."""
        rewards = [
            Reward(name="Big", amount=10_000_000, month=3),  # More than initial
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        Y = schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
        
        # Wealth = 5M, but withdrawal = 10M
        W = np.full((10, 13, 2), 5_000_000)
        
        assert schedule.validate_liquidity(W, Y) is False

    def test_summary(self, sample_rewards):
        """Summary produces readable output."""
        schedule = RewardSchedule(
            rewards=sample_rewards,
            withdrawal_policy="proportional"
        )
        
        summary = schedule.summary()
        
        assert "RewardSchedule" in summary
        assert "Mandatory" in summary
        assert "Optional" in summary
        assert "Laptop" in summary
        assert "Vacation" in summary
        assert "Car Upgrade" in summary

    def test_invalid_account_index(self, sample_accounts):
        """Invalid account index raises error."""
        rewards = [
            Reward(name="Test", amount=100_000, month=3, account=99),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        with pytest.raises(ValueError, match="out of range"):
            schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)

    def test_invalid_account_name(self, sample_accounts):
        """Invalid account name raises error."""
        rewards = [
            Reward(name="Test", amount=100_000, month=3, account="NonExistent"),
        ]
        schedule = RewardSchedule(
            rewards=rewards,
            withdrawal_policy="single_account",
            default_account=0
        )
        
        with pytest.raises(ValueError, match="not found"):
            schedule.get_fixed_withdrawals(T=12, M=2, accounts=sample_accounts)
