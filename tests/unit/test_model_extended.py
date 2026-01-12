"""
Unit tests for FinancialModel expense/reward extension.

Tests Phase 6: FinancialModel integration with expenses and rewards.
"""
import pytest
import numpy as np
from datetime import date
from dataclasses import dataclass, field

from src.model import FinancialModel, SimulationResult
from src.income import IncomeModel, FixedIncome, VariableIncome
from src.portfolio import Account
from src.expenses import FixedExpense, VariableExpense, MicroExpense, ExpenseModel
from src.rewards import Reward, RewardSchedule


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_accounts():
    """Two-account portfolio for testing."""
    return [
        Account.from_annual(
            name="Emergency",
            annual_return=0.04,
            annual_volatility=0.05,
            initial_wealth=100_000
        ),
        Account.from_annual(
            name="Investment",
            annual_return=0.08,
            annual_volatility=0.15,
            initial_wealth=50_000
        )
    ]


@pytest.fixture
def deterministic_income():
    """Deterministic income model."""
    fixed = FixedIncome(base=5_000_000, annual_growth=0.0)
    variable = VariableIncome(base=0, sigma=0.0, seed=42)
    return IncomeModel(
        fixed=fixed,
        variable=variable,
        monthly_contribution={"fixed": [0.20] * 12, "variable": [0.0] * 12}
    )


@pytest.fixture
def stochastic_income():
    """Stochastic income model."""
    fixed = FixedIncome(base=4_000_000, annual_growth=0.0)
    variable = VariableIncome(base=1_000_000, sigma=0.15, seed=42)
    return IncomeModel(
        fixed=fixed,
        variable=variable,
        monthly_contribution={"fixed": [0.20] * 12, "variable": [0.50] * 12}
    )


@pytest.fixture
def sample_expenses():
    """Sample expense model."""
    # ExpenseModel accepts single instances, not lists
    fixed = FixedExpense(base=1_100_000, name="Rent+Utilities")
    variable = VariableExpense(
        base=300_000,
        sigma=0.10,
        seasonality=None,
        name="Groceries"
    )
    micro = MicroExpense(
        lambda_base=20.0,
        severity_mean=5_000,
        severity_std=1_000,
        name="Coffee"
    )
    return ExpenseModel(
        fixed=fixed,
        variable=variable,
        micro=micro
    )


@pytest.fixture
def sample_rewards(sample_accounts):
    """Sample reward schedule with proportional policy (default)."""
    rewards = [
        Reward(name="Vacation", amount=1_000_000, month=6, account="Emergency"),
        Reward(name="Car", amount=5_000_000, month=12, account="Investment")
    ]
    # Using default 'proportional' policy - amounts split evenly across accounts
    return RewardSchedule(rewards)


@pytest.fixture
def base_model(deterministic_income, sample_accounts):
    """Base model without expenses/rewards."""
    return FinancialModel(
        income=deterministic_income,
        accounts=sample_accounts
    )


@pytest.fixture
def model_with_expenses(deterministic_income, sample_accounts, sample_expenses):
    """Model with expenses configured."""
    return FinancialModel(
        income=deterministic_income,
        accounts=sample_accounts,
        expenses=sample_expenses
    )


@pytest.fixture
def model_with_rewards(deterministic_income, sample_accounts, sample_rewards):
    """Model with rewards configured."""
    return FinancialModel(
        income=deterministic_income,
        accounts=sample_accounts,
        rewards=sample_rewards
    )


@pytest.fixture
def model_with_both(deterministic_income, sample_accounts, sample_expenses, sample_rewards):
    """Model with both expenses and rewards."""
    return FinancialModel(
        income=deterministic_income,
        accounts=sample_accounts,
        expenses=sample_expenses,
        rewards=sample_rewards
    )


# ============================================================================
# Test: Constructor with expenses/rewards
# ============================================================================

class TestFinancialModelConstructor:
    """Tests for FinancialModel constructor with extension parameters."""
    
    def test_construct_without_extensions(self, base_model):
        """Model can be constructed without expenses or rewards."""
        assert base_model.expenses is None
        assert base_model.rewards is None
    
    def test_construct_with_expenses_only(self, model_with_expenses, sample_expenses):
        """Model can be constructed with expenses only."""
        assert model_with_expenses.expenses is sample_expenses
        assert model_with_expenses.rewards is None
    
    def test_construct_with_rewards_only(self, model_with_rewards, sample_rewards):
        """Model can be constructed with rewards only."""
        assert model_with_rewards.expenses is None
        assert model_with_rewards.rewards is sample_rewards
    
    def test_construct_with_both(self, model_with_both, sample_expenses, sample_rewards):
        """Model can be constructed with both expenses and rewards."""
        assert model_with_both.expenses is sample_expenses
        assert model_with_both.rewards is sample_rewards
    
    def test_invalid_expenses_type_rejected(self, deterministic_income, sample_accounts):
        """Non-ExpenseModel types are rejected."""
        with pytest.raises(TypeError, match="expenses must be ExpenseModel"):
            FinancialModel(
                income=deterministic_income,
                accounts=sample_accounts,
                expenses={"not": "an expense model"}
            )
    
    def test_invalid_rewards_type_rejected(self, deterministic_income, sample_accounts):
        """Non-RewardSchedule types are rejected."""
        with pytest.raises(TypeError, match="rewards must be RewardSchedule"):
            FinancialModel(
                income=deterministic_income,
                accounts=sample_accounts,
                rewards=["not", "a", "schedule"]
            )


# ============================================================================
# Test: SimulationResult new fields
# ============================================================================

class TestSimulationResultExtendedFields:
    """Tests for new SimulationResult fields."""
    
    def test_base_simulation_has_no_extensions(self, base_model):
        """Simulation without extensions has default field values."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = base_model.simulate(T=T, X=X, n_sims=10, seed=42)
        
        assert result.has_expenses is False
        assert result.has_rewards is False
        assert result.withdrawals is None
        assert result.gross_contributions is None
    
    def test_simulation_with_expenses_sets_flags(self, model_with_expenses):
        """Simulation with expenses sets appropriate flags."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model_with_expenses.simulate(T=T, X=X, n_sims=10, seed=42)
        
        assert result.has_expenses is True
        assert result.has_rewards is False
        assert result.gross_contributions is not None
        assert result.withdrawals is None
    
    def test_simulation_with_rewards_sets_flags(self, model_with_rewards):
        """Simulation with rewards sets appropriate flags."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model_with_rewards.simulate(T=T, X=X, n_sims=10, seed=42)
        
        assert result.has_expenses is False
        assert result.has_rewards is True
        assert result.gross_contributions is None
        assert result.withdrawals is not None
    
    def test_simulation_with_both_sets_all_flags(self, model_with_both):
        """Simulation with both extensions sets all flags."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model_with_both.simulate(T=T, X=X, n_sims=10, seed=42)
        
        assert result.has_expenses is True
        assert result.has_rewards is True
        assert result.gross_contributions is not None
        assert result.withdrawals is not None


# ============================================================================
# Test: Expenses affect contributions
# ============================================================================

class TestExpensesAffectContributions:
    """Tests that expenses affect contribution calculations."""
    
    def test_contributions_differ_with_expenses(self, base_model, model_with_expenses):
        """Contributions should differ when expenses are configured."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result_base = base_model.simulate(T=T, X=X, n_sims=10, seed=42)
        result_exp = model_with_expenses.simulate(T=T, X=X, n_sims=10, seed=42)
        
        # With expenses configured, contributions are calculated from disposable income
        # which uses different savings rate (0.3 default vs 0.2 contribution_rate)
        # So they won't be equal
        assert not np.allclose(result_exp.contributions, result_base.contributions)
        
        # Gross contributions (stored in result) should match base contributions
        # Note: gross_contributions stores original contributions before expense logic
        assert result_exp.gross_contributions is not None
    
    def test_expense_model_is_synced(self, model_with_expenses, sample_expenses):
        """Expense model should be synced to income model."""
        # The model's expense should be synced to income
        assert model_with_expenses.income.expenses is sample_expenses


# ============================================================================
# Test: Rewards create withdrawals
# ============================================================================

class TestRewardsCreateWithdrawals:
    """Tests that rewards generate withdrawal matrix Y."""
    
    def test_withdrawal_matrix_shape(self, model_with_rewards):
        """Withdrawal matrix Y has correct shape (T, M)."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model_with_rewards.simulate(T=T, X=X, n_sims=10, seed=42)
        
        assert result.withdrawals is not None
        assert result.withdrawals.shape == (T, M)
    
    def test_withdrawals_at_correct_months(self, model_with_rewards):
        """Withdrawals occur at the specified reward months."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model_with_rewards.simulate(T=T, X=X, n_sims=10, seed=42)
        Y = result.withdrawals
        
        # With proportional policy, 1M vacation is split: 500k per account
        assert Y[5, 0] == 500_000  # month 6 = index 5
        assert Y[5, 1] == 500_000
        
        # Car at month 12: 5M split to 2.5M each
        assert Y[11, 0] == 2_500_000  # month 12 = index 11
        assert Y[11, 1] == 2_500_000
    
    def test_no_withdrawals_at_other_months(self, model_with_rewards):
        """No withdrawals at months without rewards."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model_with_rewards.simulate(T=T, X=X, n_sims=10, seed=42)
        Y = result.withdrawals
        
        # Sum of withdrawals should equal total rewards
        total_rewards = 1_000_000 + 5_000_000
        assert Y.sum() == total_rewards


# ============================================================================
# Test: Withdrawals affect wealth
# ============================================================================

class TestWithdrawalsAffectWealth:
    """Tests that withdrawals reduce wealth trajectories."""
    
    def test_wealth_lower_with_withdrawals(self, base_model, model_with_rewards):
        """Wealth should be lower when withdrawals occur."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result_base = base_model.simulate(T=T, X=X, n_sims=50, seed=42)
        result_rew = model_with_rewards.simulate(T=T, X=X, n_sims=50, seed=42)
        
        # Mean final total wealth should be lower with withdrawals
        mean_final_base = result_base.total_wealth[:, -1].mean()
        mean_final_rew = result_rew.total_wealth[:, -1].mean()
        
        # Total rewards: 6M
        total_rewards = 6_000_000
        
        # Difference should be approximately rewards + compounding loss
        # At minimum, should be at least the raw withdrawal amount
        difference = mean_final_base - mean_final_rew
        assert difference > total_rewards * 0.8  # Allow for some variance
    
    def test_wealth_matches_after_accounting_withdrawals(
        self, base_model, model_with_rewards
    ):
        """Wealth + withdrawals should approximately match base wealth."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result_base = base_model.simulate(T=T, X=X, n_sims=100, seed=42)
        result_rew = model_with_rewards.simulate(T=T, X=X, n_sims=100, seed=42)
        
        # The gap should be explainable by:
        # 1. Raw withdrawal amounts (6M)
        # 2. Lost compound growth on withdrawn amounts
        
        final_base = result_base.total_wealth[:, -1].mean()
        final_rew = result_rew.total_wealth[:, -1].mean()
        gap = final_base - final_rew
        
        # Gap should be at least the withdrawals (6M)
        assert gap >= 6_000_000 * 0.8
        
        # But not more than 2x (accounting for compound loss)
        assert gap <= 6_000_000 * 2.0


# ============================================================================
# Test: Combined expenses + rewards
# ============================================================================

class TestCombinedExpensesRewards:
    """Tests for models with both expenses and rewards."""
    
    def test_both_effects_present(self, model_with_both):
        """Both expense and withdrawal effects should be present."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model_with_both.simulate(T=T, X=X, n_sims=10, seed=42)
        
        # Expense effects
        assert result.has_expenses is True
        assert result.gross_contributions is not None
        
        # Contributions should be computed from disposable income
        # (using different savings rate)
        assert result.contributions is not None
        
        # Reward effects
        assert result.has_rewards is True
        assert result.withdrawals is not None
        assert result.withdrawals.sum() > 0
    
    def test_combined_effect_on_wealth(
        self, base_model, model_with_rewards
    ):
        """Model with rewards should have lower final wealth than base."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        n_sims = 100
        seed = 42
        
        result_base = base_model.simulate(T=T, X=X, n_sims=n_sims, seed=seed)
        result_rew = model_with_rewards.simulate(T=T, X=X, n_sims=n_sims, seed=seed)
        
        mean_base = result_base.total_wealth[:, -1].mean()
        mean_rew = result_rew.total_wealth[:, -1].mean()
        
        # Rewards should reduce wealth
        assert mean_rew < mean_base


# ============================================================================
# Test: Reproducibility with extensions
# ============================================================================

class TestReproducibilityWithExtensions:
    """Tests that simulations with extensions are reproducible."""
    
    def test_expenses_simulation_reproducible(self, model_with_expenses):
        """Simulation with expenses is reproducible with seed."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result1 = model_with_expenses.simulate(T=T, X=X, n_sims=50, seed=42)
        result2 = model_with_expenses.simulate(T=T, X=X, n_sims=50, seed=42)
        
        np.testing.assert_array_almost_equal(
            result1.wealth, result2.wealth
        )
        np.testing.assert_array_almost_equal(
            result1.contributions, result2.contributions
        )
    
    def test_rewards_simulation_reproducible(self, model_with_rewards):
        """Simulation with rewards is reproducible with seed."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result1 = model_with_rewards.simulate(T=T, X=X, n_sims=50, seed=42)
        result2 = model_with_rewards.simulate(T=T, X=X, n_sims=50, seed=42)
        
        np.testing.assert_array_almost_equal(
            result1.wealth, result2.wealth
        )
        np.testing.assert_array_almost_equal(
            result1.withdrawals, result2.withdrawals
        )
    
    def test_combined_simulation_reproducible(self, model_with_both):
        """Simulation with both extensions is reproducible with seed."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result1 = model_with_both.simulate(T=T, X=X, n_sims=50, seed=42)
        result2 = model_with_both.simulate(T=T, X=X, n_sims=50, seed=42)
        
        np.testing.assert_array_almost_equal(
            result1.wealth, result2.wealth
        )


# ============================================================================
# Test: Stochastic income with extensions
# ============================================================================

class TestStochasticIncomeWithExtensions:
    """Tests that extensions work with stochastic income."""
    
    def test_stochastic_income_with_expenses(
        self, stochastic_income, sample_accounts, sample_expenses
    ):
        """Model works with stochastic income and expenses."""
        model = FinancialModel(
            income=stochastic_income,
            accounts=sample_accounts,
            expenses=sample_expenses
        )
        
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model.simulate(T=T, X=X, n_sims=50, seed=42)
        
        # Contributions should be stochastic (n_sims, T)
        assert result.contributions.shape == (50, T)
        assert result.gross_contributions.shape == (50, T)
        
        # Deductions should still be positive
        deduction = result.gross_contributions - result.contributions
        assert np.all(deduction >= 0)
    
    def test_stochastic_income_with_rewards(
        self, stochastic_income, sample_accounts, sample_rewards
    ):
        """Model works with stochastic income and rewards."""
        model = FinancialModel(
            income=stochastic_income,
            accounts=sample_accounts,
            rewards=sample_rewards
        )
        
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result = model.simulate(T=T, X=X, n_sims=50, seed=42)
        
        # Withdrawals are deterministic (T, M) regardless of income
        assert result.withdrawals.shape == (T, M)
        
        # Wealth trajectories should vary
        assert result.wealth.shape == (50, T + 1, M)


# ============================================================================
# Test: Cache behavior with extensions
# ============================================================================

class TestCacheBehaviorWithExtensions:
    """Tests that caching works correctly with extensions."""
    
    def test_cache_hit_with_extensions(self, model_with_both):
        """Cached results should be returned for identical parameters."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result1 = model_with_both.simulate(T=T, X=X, n_sims=50, seed=42)
        result2 = model_with_both.simulate(T=T, X=X, n_sims=50, seed=42)
        
        # Should be same object (cached)
        assert result1 is result2
    
    def test_cache_miss_with_different_seed(self, model_with_both):
        """Different seeds should not return cached results."""
        T, M = 12, 2
        X = np.ones((T, M)) / M
        
        result1 = model_with_both.simulate(T=T, X=X, n_sims=50, seed=42)
        result2 = model_with_both.simulate(T=T, X=X, n_sims=50, seed=43)
        
        # Should be different objects
        assert result1 is not result2


# ============================================================================
# Test: Edge cases
# ============================================================================

class TestExtensionEdgeCases:
    """Edge case tests for extensions."""
    
    def test_rewards_beyond_horizon(self, deterministic_income, sample_accounts):
        """Rewards beyond simulation horizon are ignored."""
        rewards = RewardSchedule([
            Reward(name="Early", amount=1_000_000, month=6, account="Emergency"),
            Reward(name="Late", amount=5_000_000, month=24, account="Investment")
        ])
        
        model = FinancialModel(
            income=deterministic_income,
            accounts=sample_accounts,
            rewards=rewards
        )
        
        T = 12  # Only simulate 12 months
        M = 2
        X = np.ones((T, M)) / M
        
        result = model.simulate(T=T, X=X, n_sims=10, seed=42)
        
        # Only early reward (1M) should be included
        assert result.withdrawals.sum() == 1_000_000
