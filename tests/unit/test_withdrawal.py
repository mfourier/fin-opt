"""
Unit tests for withdrawal module.

Tests all components:
- WithdrawalEvent: Single withdrawal specification
- WithdrawalSchedule: Collection of deterministic withdrawals
- StochasticWithdrawal: Withdrawals with uncertainty
- WithdrawalModel: Unified facade combining both types
"""

import pytest
import numpy as np
from datetime import date

from src.withdrawal import (
    WithdrawalEvent,
    WithdrawalSchedule,
    StochasticWithdrawal,
    WithdrawalModel
)
from src.portfolio import Account


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def accounts():
    """Sample accounts for testing."""
    return [
        Account.from_annual("Conservador", annual_return=0.06, annual_volatility=0.08),
        Account.from_annual("Agresivo", annual_return=0.12, annual_volatility=0.15)
    ]


@pytest.fixture
def start_date():
    """Standard start date for tests."""
    return date(2025, 1, 1)


# ---------------------------------------------------------------------------
# WithdrawalEvent Tests
# ---------------------------------------------------------------------------

class TestWithdrawalEvent:
    """Tests for WithdrawalEvent."""

    def test_creation_with_valid_params(self):
        """Test creating event with valid parameters."""
        event = WithdrawalEvent(
            account="Conservador",
            amount=400_000,
            date=date(2025, 6, 1),
            description="Compra bicicleta"
        )
        assert event.account == "Conservador"
        assert event.amount == 400_000
        assert event.date == date(2025, 6, 1)
        assert event.description == "Compra bicicleta"

    def test_creation_fails_with_negative_amount(self):
        """Test that negative amounts raise ValueError."""
        with pytest.raises(ValueError, match="amount must be positive"):
            WithdrawalEvent(
                account="Conservador",
                amount=-100_000,
                date=date(2025, 6, 1)
            )

    def test_creation_fails_with_zero_amount(self):
        """Test that zero amounts raise ValueError."""
        with pytest.raises(ValueError, match="amount must be positive"):
            WithdrawalEvent(
                account="Conservador",
                amount=0,
                date=date(2025, 6, 1)
            )

    def test_resolve_month_forward(self, start_date):
        """Test month resolution for future dates (1-indexed)."""
        event = WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
        assert event.resolve_month(start_date) == 6  # June = month 6 (1-indexed)

    def test_resolve_month_same_month(self, start_date):
        """Test month resolution for same month (1-indexed)."""
        event = WithdrawalEvent("Conservador", 400_000, date(2025, 1, 1))
        assert event.resolve_month(start_date) == 1  # January = month 1 (1-indexed)

    def test_resolve_month_backward(self, start_date):
        """Test month resolution for past dates (negative offset)."""
        event = WithdrawalEvent("Conservador", 400_000, date(2024, 11, 1))
        assert event.resolve_month(start_date) == -1  # Nov 2024 is 1 month before Jan 2025 (1-indexed: 0, then -1)

    def test_resolve_month_next_year(self, start_date):
        """Test month resolution across year boundary (1-indexed)."""
        event = WithdrawalEvent("Conservador", 400_000, date(2026, 3, 1))
        assert event.resolve_month(start_date) == 15  # March 2026 = month 15 (1-indexed)

    def test_repr(self):
        """Test string representation."""
        event = WithdrawalEvent(
            account="Conservador",
            amount=400_000,
            date=date(2025, 6, 1),
            description="Compra bicicleta"
        )
        repr_str = repr(event)
        assert "WithdrawalEvent" in repr_str
        assert "400,000" in repr_str
        assert "2025-06-01" in repr_str
        assert "Compra bicicleta" in repr_str


# ---------------------------------------------------------------------------
# WithdrawalSchedule Tests
# ---------------------------------------------------------------------------

class TestWithdrawalSchedule:
    """Tests for WithdrawalSchedule."""

    def test_empty_schedule(self, accounts, start_date):
        """Test empty schedule returns zeros."""
        schedule = WithdrawalSchedule(events=[])
        D = schedule.to_array(T=12, start_date=start_date, accounts=accounts)
        
        assert D.shape == (12, 2)
        assert np.allclose(D, 0)

    def test_single_withdrawal_by_name(self, accounts, start_date):
        """Test single withdrawal using account name."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
        ])
        D = schedule.to_array(T=12, start_date=start_date, accounts=accounts)
        
        assert D.shape == (12, 2)
        assert D[5, 0] == 400_000  # June (month 5), account 0
        assert D[5, 1] == 0
        assert np.sum(D) == 400_000

    def test_single_withdrawal_by_index(self, accounts, start_date):
        """Test single withdrawal using account index."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent(1, 2_000_000, date(2025, 12, 1))
        ])
        D = schedule.to_array(T=12, start_date=start_date, accounts=accounts)
        
        assert D[11, 1] == 2_000_000  # December (month 11), account 1
        assert D[11, 0] == 0

    def test_multiple_withdrawals(self, accounts, start_date):
        """Test multiple withdrawals from different accounts."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1)),
            WithdrawalEvent("Agresivo", 2_000_000, date(2025, 12, 1))
        ])
        D = schedule.to_array(T=24, start_date=start_date, accounts=accounts)
        
        assert D[5, 0] == 400_000
        assert D[11, 1] == 2_000_000
        assert np.sum(D) == 2_400_000

    def test_multiple_withdrawals_same_month_same_account(self, accounts, start_date):
        """Test that multiple withdrawals on same month/account are summed."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1)),
            WithdrawalEvent("Conservador", 100_000, date(2025, 6, 1))
        ])
        D = schedule.to_array(T=12, start_date=start_date, accounts=accounts)
        
        assert D[5, 0] == 500_000  # Sum of both

    def test_withdrawal_before_start_ignored(self, accounts, start_date):
        """Test that withdrawals before start_date are ignored with warning."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2024, 11, 1))
        ])
        
        with pytest.warns(UserWarning, match="before simulation start"):
            D = schedule.to_array(T=12, start_date=start_date, accounts=accounts)
        
        assert np.allclose(D, 0)

    def test_withdrawal_beyond_horizon_ignored(self, accounts, start_date):
        """Test that withdrawals beyond T are ignored with warning."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2027, 1, 1))
        ])
        
        with pytest.warns(UserWarning, match="beyond horizon"):
            D = schedule.to_array(T=12, start_date=start_date, accounts=accounts)
        
        assert np.allclose(D, 0)

    def test_invalid_account_name_raises(self, accounts, start_date):
        """Test that invalid account name raises ValueError."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("NonExistent", 400_000, date(2025, 6, 1))
        ])
        
        with pytest.raises(ValueError, match="Account name.*not found"):
            schedule.to_array(T=12, start_date=start_date, accounts=accounts)

    def test_invalid_account_index_raises(self, accounts, start_date):
        """Test that invalid account index raises ValueError."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent(5, 400_000, date(2025, 6, 1))
        ])
        
        with pytest.raises(ValueError, match="Account index.*out of range"):
            schedule.to_array(T=12, start_date=start_date, accounts=accounts)

    def test_total_by_account(self, accounts):
        """Test total_by_account computes correct sums."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1)),
            WithdrawalEvent("Conservador", 100_000, date(2025, 9, 1)),
            WithdrawalEvent("Agresivo", 2_000_000, date(2025, 12, 1))
        ])
        
        totals = schedule.total_by_account(accounts)
        assert totals["Conservador"] == 500_000
        assert totals["Agresivo"] == 2_000_000

    def test_get_events_for_account_by_name(self, accounts):
        """Test filtering events by account name."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1)),
            WithdrawalEvent("Agresivo", 2_000_000, date(2025, 12, 1)),
            WithdrawalEvent("Conservador", 100_000, date(2025, 9, 1))
        ])
        
        events = schedule.get_events_for_account("Conservador")
        assert len(events) == 2
        assert all(e.account == "Conservador" for e in events)

    def test_serialization_roundtrip(self, accounts):
        """Test to_dict/from_dict roundtrip."""
        schedule = WithdrawalSchedule(events=[
            WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1), "Bicicleta")
        ])
        
        # Serialize
        payload = schedule.to_dict()
        
        # Deserialize
        restored = WithdrawalSchedule.from_dict(payload)
        
        # Verify
        assert len(restored.events) == 1
        assert restored.events[0].account == "Conservador"
        assert restored.events[0].amount == 400_000
        assert restored.events[0].date == date(2025, 6, 1)
        assert restored.events[0].description == "Bicicleta"


# ---------------------------------------------------------------------------
# StochasticWithdrawal Tests
# ---------------------------------------------------------------------------

class TestStochasticWithdrawal:
    """Tests for StochasticWithdrawal."""

    def test_creation_with_month(self):
        """Test creating stochastic withdrawal with month."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            floor=200_000,
            cap=500_000,
            seed=42
        )
        assert withdrawal.month == 6
        assert withdrawal.date is None

    def test_creation_with_date(self):
        """Test creating stochastic withdrawal with date."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            date=date(2025, 9, 1),
            seed=42
        )
        assert withdrawal.date == date(2025, 9, 1)
        assert withdrawal.month is None

    def test_creation_fails_with_both_month_and_date(self):
        """Test that specifying both month and date raises ValueError."""
        with pytest.raises(ValueError, match="Specify either month or date, not both"):
            StochasticWithdrawal(
                account="Conservador",
                base_amount=300_000,
                sigma=50_000,
                month=6,
                date=date(2025, 9, 1)
            )

    def test_creation_fails_with_neither_month_nor_date(self):
        """Test that omitting both month and date raises ValueError."""
        with pytest.raises(ValueError, match="Must specify either month or date"):
            StochasticWithdrawal(
                account="Conservador",
                base_amount=300_000,
                sigma=50_000
            )

    def test_sample_shape(self):
        """Test that sample returns correct shape."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            seed=42
        )
        samples = withdrawal.sample(n_sims=100)
        assert samples.shape == (100,)

    def test_sample_respects_floor(self):
        """Test that samples respect floor constraint."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            floor=200_000,
            seed=42
        )
        samples = withdrawal.sample(n_sims=1000)
        assert (samples >= 200_000).all()

    def test_sample_respects_cap(self):
        """Test that samples respect cap constraint."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            cap=400_000,
            seed=42
        )
        samples = withdrawal.sample(n_sims=1000)
        assert (samples <= 400_000).all()

    def test_sample_respects_both_constraints(self):
        """Test that samples respect both floor and cap."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            floor=200_000,
            cap=400_000,
            seed=42
        )
        samples = withdrawal.sample(n_sims=1000)
        assert (samples >= 200_000).all()
        assert (samples <= 400_000).all()

    def test_sample_approximately_correct_mean(self):
        """Test that sample mean is approximately base_amount (without truncation)."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            seed=42
        )
        samples = withdrawal.sample(n_sims=10000)
        # Mean should be close to base_amount (within 3 sigma/sqrt(n))
        assert abs(samples.mean() - 300_000) < 3 * 50_000 / np.sqrt(10000)

    def test_sample_reproducibility_with_seed(self):
        """Test that same seed produces same samples."""
        withdrawal1 = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            seed=42
        )
        withdrawal2 = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6,
            seed=42
        )
        
        samples1 = withdrawal1.sample(n_sims=100)
        samples2 = withdrawal2.sample(n_sims=100)
        
        assert np.allclose(samples1, samples2)

    def test_resolve_month_with_month_param(self, start_date):
        """Test resolve_month when using month parameter."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            month=6
        )
        assert withdrawal.resolve_month(start_date) == 6

    def test_resolve_month_with_date_param(self, start_date):
        """Test resolve_month when using date parameter (1-indexed)."""
        withdrawal = StochasticWithdrawal(
            account="Conservador",
            base_amount=300_000,
            sigma=50_000,
            date=date(2025, 9, 1)
        )
        assert withdrawal.resolve_month(start_date) == 9  # September = month 9 (1-indexed)


# ---------------------------------------------------------------------------
# WithdrawalModel Tests
# ---------------------------------------------------------------------------

class TestWithdrawalModel:
    """Tests for WithdrawalModel."""

    def test_empty_model(self, accounts, start_date):
        """Test empty model returns zeros."""
        model = WithdrawalModel()
        D = model.to_array(T=12, start_date=start_date, accounts=accounts, n_sims=10)
        
        assert D.shape == (10, 12, 2)
        assert np.allclose(D, 0)

    def test_scheduled_only(self, accounts, start_date):
        """Test model with only scheduled withdrawals."""
        model = WithdrawalModel(
            scheduled=WithdrawalSchedule(events=[
                WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
            ])
        )
        D = model.to_array(T=12, start_date=start_date, accounts=accounts, n_sims=10, seed=42)
        
        assert D.shape == (10, 12, 2)
        # All scenarios should have same withdrawal (deterministic)
        assert np.allclose(D[:, 5, 0], 400_000)
        assert D[:, 5, 0].std() == 0  # No variance

    def test_stochastic_only(self, accounts, start_date):
        """Test model with only stochastic withdrawals."""
        model = WithdrawalModel(
            stochastic=[
                StochasticWithdrawal(
                    account="Conservador",
                    base_amount=300_000,
                    sigma=50_000,
                    date=date(2025, 9, 1),
                    seed=42
                )
            ]
        )
        D = model.to_array(T=12, start_date=start_date, accounts=accounts, n_sims=100, seed=42)
        
        assert D.shape == (100, 12, 2)
        # Should have variance across scenarios
        assert D[:, 8, 0].std() > 0
        # Mean should be close to base_amount
        assert abs(D[:, 8, 0].mean() - 300_000) < 10_000

    def test_combined_scheduled_and_stochastic(self, accounts, start_date):
        """Test model combining both types of withdrawals."""
        model = WithdrawalModel(
            scheduled=WithdrawalSchedule(events=[
                WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
            ]),
            stochastic=[
                StochasticWithdrawal(
                    account="Conservador",
                    base_amount=300_000,
                    sigma=50_000,
                    date=date(2025, 9, 1),
                    seed=42
                )
            ]
        )
        D = model.to_array(T=12, start_date=start_date, accounts=accounts, n_sims=100, seed=42)
        
        assert D.shape == (100, 12, 2)
        # June: deterministic
        assert np.allclose(D[:, 5, 0], 400_000)
        # September: stochastic
        assert D[:, 8, 0].std() > 0

    def test_total_expected(self, accounts):
        """Test total_expected computes correct values."""
        model = WithdrawalModel(
            scheduled=WithdrawalSchedule(events=[
                WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
            ]),
            stochastic=[
                StochasticWithdrawal(
                    account="Conservador",
                    base_amount=300_000,
                    sigma=50_000,
                    month=8
                )
            ]
        )
        
        totals = model.total_expected(accounts)
        assert totals["Conservador"] == 700_000  # 400k + 300k
        assert totals["Agresivo"] == 0

    def test_serialization_roundtrip(self, accounts):
        """Test to_dict/from_dict roundtrip."""
        model = WithdrawalModel(
            scheduled=WithdrawalSchedule(events=[
                WithdrawalEvent("Conservador", 400_000, date(2025, 6, 1))
            ]),
            stochastic=[
                StochasticWithdrawal(
                    account="Conservador",
                    base_amount=300_000,
                    sigma=50_000,
                    date=date(2025, 9, 1),
                    floor=200_000,
                    cap=400_000,
                    seed=42
                )
            ]
        )
        
        # Serialize
        payload = model.to_dict()
        
        # Deserialize
        restored = WithdrawalModel.from_dict(payload)
        
        # Verify scheduled
        assert len(restored.scheduled.events) == 1
        assert restored.scheduled.events[0].amount == 400_000
        
        # Verify stochastic
        assert len(restored.stochastic) == 1
        assert restored.stochastic[0].base_amount == 300_000
        assert restored.stochastic[0].sigma == 50_000
        assert restored.stochastic[0].floor == 200_000
        assert restored.stochastic[0].cap == 400_000


# ---------------------------------------------------------------------------
# Integration Tests: FinancialModel + Withdrawals
# ---------------------------------------------------------------------------

class TestFinancialModelWithdrawalIntegration:
    """Integration tests for FinancialModel.simulate() with withdrawals."""

    @pytest.fixture
    def financial_model(self):
        """Create a FinancialModel for integration tests."""
        from src.income import IncomeModel, FixedIncome
        from src.model import FinancialModel

        income = IncomeModel(
            fixed=FixedIncome(base=500_000, annual_growth=0.0)
        )
        accounts = [
            Account.from_annual("Conservador", annual_return=0.06, annual_volatility=0.08),
            Account.from_annual("Agresivo", annual_return=0.12, annual_volatility=0.15)
        ]
        return FinancialModel(income, accounts, enable_cache=False)

    def test_simulate_without_withdrawals(self, financial_model, start_date):
        """Test that simulation works without withdrawals (backward compat)."""
        T = 12
        M = 2
        X = np.tile([0.5, 0.5], (T, 1))

        result = financial_model.simulate(
            T=T, X=X, n_sims=10, seed=42, start=start_date
        )

        assert result.wealth.shape == (10, T + 1, M)
        assert result.withdrawals is None

    def test_simulate_with_scheduled_withdrawals(self, financial_model, start_date):
        """Test simulation with scheduled withdrawals reduces wealth."""
        T = 12
        M = 2
        X = np.tile([0.5, 0.5], (T, 1))

        # No withdrawals
        result_no_d = financial_model.simulate(
            T=T, X=X, n_sims=10, seed=42, start=start_date
        )

        # With withdrawal from account 0 in month 5
        withdrawals = WithdrawalModel(
            scheduled=WithdrawalSchedule(events=[
                WithdrawalEvent("Conservador", 200_000, date(2025, 6, 1))
            ])
        )
        result_with_d = financial_model.simulate(
            T=T, X=X, n_sims=10, seed=42, start=start_date,
            withdrawals=withdrawals
        )

        # Verify withdrawal array is present
        assert result_with_d.withdrawals is not None
        assert result_with_d.withdrawals.shape == (10, T, M)

        # Withdrawal in month 5 account 0 should be 200_000 for all scenarios
        assert np.allclose(result_with_d.withdrawals[:, 5, 0], 200_000)

        # Wealth after withdrawal month should be lower
        # Compare mean wealth at final time
        mean_final_no_d = result_no_d.total_wealth[:, -1].mean()
        mean_final_with_d = result_with_d.total_wealth[:, -1].mean()

        assert mean_final_with_d < mean_final_no_d

    def test_simulate_with_stochastic_withdrawals(self, financial_model, start_date):
        """Test simulation with stochastic withdrawals has variance."""
        T = 12
        M = 2
        X = np.tile([0.5, 0.5], (T, 1))

        withdrawals = WithdrawalModel(
            stochastic=[
                StochasticWithdrawal(
                    account="Agresivo",
                    base_amount=100_000,
                    sigma=20_000,
                    date=date(2025, 9, 1),  # Month 8
                    seed=42
                )
            ]
        )

        result = financial_model.simulate(
            T=T, X=X, n_sims=100, seed=42, start=start_date,
            withdrawals=withdrawals
        )

        assert result.withdrawals is not None
        # Month 8, account 1 (Agresivo) should have variance
        withdrawal_values = result.withdrawals[:, 8, 1]
        assert withdrawal_values.std() > 0
        # Mean should be close to base_amount
        assert abs(withdrawal_values.mean() - 100_000) < 10_000

    def test_simulate_result_attributes(self, financial_model, start_date):
        """Test that SimulationResult has correct attributes with withdrawals."""
        T = 12
        M = 2
        X = np.tile([0.5, 0.5], (T, 1))

        withdrawals = WithdrawalModel(
            scheduled=WithdrawalSchedule(events=[
                WithdrawalEvent("Conservador", 100_000, date(2025, 3, 1))
            ])
        )

        result = financial_model.simulate(
            T=T, X=X, n_sims=10, seed=42, start=start_date,
            withdrawals=withdrawals
        )

        # Check all expected attributes
        assert hasattr(result, 'wealth')
        assert hasattr(result, 'total_wealth')
        assert hasattr(result, 'contributions')
        assert hasattr(result, 'returns')
        assert hasattr(result, 'income')
        assert hasattr(result, 'allocation')
        assert hasattr(result, 'withdrawals')
        assert hasattr(result, 'T')
        assert hasattr(result, 'n_sims')
        assert hasattr(result, 'M')
        assert hasattr(result, 'start')
        assert hasattr(result, 'seed')
        assert hasattr(result, 'account_names')

        assert result.T == T
        assert result.n_sims == 10
        assert result.M == M
        assert result.withdrawals is not None

    def test_cache_key_differs_with_withdrawals(self, financial_model, start_date):
        """Test that different withdrawal models produce different cache keys."""
        T = 12
        M = 2
        X = np.tile([0.5, 0.5], (T, 1))

        # Enable caching for this test
        financial_model._cache_enabled = True
        financial_model.clear_cache()

        # Simulate without withdrawals
        result1 = financial_model.simulate(
            T=T, X=X, n_sims=10, seed=42, start=start_date
        )

        # Simulate with withdrawals
        withdrawals = WithdrawalModel(
            scheduled=WithdrawalSchedule(events=[
                WithdrawalEvent("Conservador", 100_000, date(2025, 3, 1))
            ])
        )
        result2 = financial_model.simulate(
            T=T, X=X, n_sims=10, seed=42, start=start_date,
            withdrawals=withdrawals
        )

        # Results should be different (not from cache)
        assert result1 is not result2
        assert result1.withdrawals is None
        assert result2.withdrawals is not None


# ---------------------------------------------------------------------------
# Integration Tests: Portfolio + Withdrawals (Direct)
# ---------------------------------------------------------------------------

class TestPortfolioWithdrawalIntegration:
    """Test Portfolio.simulate() with withdrawals directly."""

    def test_portfolio_simulate_with_D_deterministic(self, accounts, start_date):
        """Test Portfolio.simulate() with deterministic D array."""
        from src.portfolio import Portfolio
        from src.returns import ReturnModel

        portfolio = Portfolio(accounts)
        returns = ReturnModel(accounts)

        T = 12
        M = 2
        n_sims = 10

        A = np.full((n_sims, T), 500_000.0)  # (n_sims, T)
        R = returns.generate(T=T, n_sims=n_sims, seed=42)  # (n_sims, T, M)
        X = np.tile([0.5, 0.5], (T, 1))  # (T, M)

        # Deterministic withdrawal: (T, M)
        D = np.zeros((T, M))
        D[5, 0] = 200_000  # 200k from account 0 in month 5

        result_with_D = portfolio.simulate(A=A, R=R, X=X, D=D)
        result_no_D = portfolio.simulate(A=A, R=R, X=X)

        # Wealth with withdrawal should be lower
        assert result_with_D["total_wealth"][:, -1].mean() < result_no_D["total_wealth"][:, -1].mean()

    def test_portfolio_simulate_with_D_stochastic(self, accounts, start_date):
        """Test Portfolio.simulate() with stochastic D array."""
        from src.portfolio import Portfolio
        from src.returns import ReturnModel

        portfolio = Portfolio(accounts)
        returns = ReturnModel(accounts)

        T = 12
        M = 2
        n_sims = 10

        A = np.full((n_sims, T), 500_000.0)
        R = returns.generate(T=T, n_sims=n_sims, seed=42)
        X = np.tile([0.5, 0.5], (T, 1))

        # Stochastic withdrawal: (n_sims, T, M)
        rng = np.random.default_rng(42)
        D = np.zeros((n_sims, T, M))
        D[:, 5, 0] = rng.normal(200_000, 20_000, size=n_sims)  # Variable withdrawal

        result = portfolio.simulate(A=A, R=R, X=X, D=D)

        # Check that variance in withdrawals leads to variance in final wealth
        assert result["total_wealth"][:, -1].std() > 0
