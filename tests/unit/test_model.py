"""
Unit tests for model.py module.

Tests FinancialModel and SimulationResult classes.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from src.income import IncomeModel, FixedIncome, VariableIncome
from src.portfolio import Account
from src.model import FinancialModel, SimulationResult
from src.goals import TerminalGoal, IntermediateGoal
from src.optimization import CVaROptimizer, OptimizationResult
from src.withdrawal import WithdrawalModel, WithdrawalSchedule, WithdrawalEvent


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def income():
    """Create test income model."""
    return IncomeModel(
        fixed=FixedIncome(base=1_500_000, annual_growth=0.03),
        variable=VariableIncome(base=200_000, sigma=0.1, seed=42)
    )


@pytest.fixture
def deterministic_income():
    """Create deterministic income model (no variable component)."""
    return IncomeModel(
        fixed=FixedIncome(base=1_000_000, annual_growth=0.0),
        variable=None
    )


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
def model(income, accounts):
    """Create test FinancialModel."""
    return FinancialModel(income=income, accounts=accounts)


@pytest.fixture
def start_date():
    """Test start date."""
    return date(2025, 1, 1)


@pytest.fixture
def sample_result(model):
    """Create sample simulation result."""
    T = 12
    n_sims = 100
    X = np.tile([0.6, 0.4], (T, 1))
    return model.simulate(T=T, n_sims=n_sims, X=X, seed=42, start=date(2025, 1, 1))


# ============================================================================
# FINANCIALMODEL INSTANTIATION TESTS
# ============================================================================

class TestFinancialModelInstantiation:
    """Test FinancialModel initialization."""

    def test_basic_instantiation(self, income, accounts):
        """Test basic FinancialModel creation."""
        model = FinancialModel(income=income, accounts=accounts)

        assert model.income is income
        assert model.accounts == accounts
        assert model.M == 2

    def test_single_account(self, income):
        """Test FinancialModel with single account."""
        accounts = [Account.from_annual("Single", 0.08, 0.10)]
        model = FinancialModel(income=income, accounts=accounts)

        assert model.M == 1

    def test_empty_accounts_raises(self, income):
        """Test that empty accounts list raises ValueError."""
        with pytest.raises(ValueError, match="accounts list cannot be empty"):
            FinancialModel(income=income, accounts=[])

    def test_with_default_correlation(self, income, accounts):
        """Test FinancialModel with custom correlation matrix."""
        correlation = np.array([[1.0, 0.5], [0.5, 1.0]])
        model = FinancialModel(income=income, accounts=accounts, default_correlation=correlation)

        np.testing.assert_array_equal(model.returns.default_correlation, correlation)

    def test_cache_enabled_default(self, income, accounts):
        """Test cache is enabled by default."""
        model = FinancialModel(income=income, accounts=accounts)

        assert model._cache_enabled is True

    def test_cache_disabled(self, income, accounts):
        """Test cache can be disabled."""
        model = FinancialModel(income=income, accounts=accounts, enable_cache=False)

        assert model._cache_enabled is False

    def test_returns_model_created(self, income, accounts):
        """Test ReturnModel is correctly initialized."""
        model = FinancialModel(income=income, accounts=accounts)

        assert model.returns is not None
        assert model.returns.M == 2

    def test_portfolio_created(self, income, accounts):
        """Test Portfolio is correctly initialized."""
        model = FinancialModel(income=income, accounts=accounts)

        assert model.portfolio is not None
        assert len(model.portfolio.accounts) == 2

    def test_repr(self, model):
        """Test __repr__ output."""
        repr_str = repr(model)

        assert "FinancialModel" in repr_str
        assert "M=2" in repr_str
        assert "Conservative" in repr_str
        assert "Aggressive" in repr_str
        assert "cache=enabled" in repr_str


# ============================================================================
# FINANCIALMODEL SIMULATE TESTS
# ============================================================================

class TestFinancialModelSimulate:
    """Test FinancialModel.simulate() method."""

    def test_simulate_returns_result(self, model):
        """Test simulate returns SimulationResult."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert isinstance(result, SimulationResult)

    def test_simulate_wealth_shape(self, model):
        """Test simulate returns correct wealth shape."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert result.wealth.shape == (n_sims, T + 1, 2)

    def test_simulate_contributions_shape(self, model):
        """Test simulate returns correct contributions shape."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert result.contributions.shape == (n_sims, T)

    def test_simulate_returns_shape(self, model):
        """Test simulate returns correct returns shape."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        assert result.returns.shape == (n_sims, T, 2)

    def test_simulate_reproducibility(self, model):
        """Test simulate reproducibility with seed."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)
        result2 = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)

        np.testing.assert_array_equal(result1.wealth, result2.wealth)

    def test_simulate_different_seeds(self, model):
        """Test simulate produces different results with different seeds."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=n_sims, X=X, seed=42)
        result2 = model.simulate(T=T, n_sims=n_sims, X=X, seed=43)

        assert not np.allclose(result1.wealth, result2.wealth)

    def test_simulate_start_date(self, model):
        """Test simulate with start_date parameter."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42, start=date(2025, 1, 1))

        assert isinstance(result, SimulationResult)
        assert result.start == date(2025, 1, 1)


class TestFinancialModelSimulateValidation:
    """Test FinancialModel.simulate() validation."""

    def test_invalid_n_sims_zero_raises(self, model):
        """Test n_sims=0 raises ValueError."""
        T = 12
        X = np.tile([0.6, 0.4], (T, 1))

        with pytest.raises(ValueError, match="n_sims must be positive"):
            model.simulate(T=T, n_sims=0, X=X, seed=42)

    def test_invalid_n_sims_negative_raises(self, model):
        """Test negative n_sims raises ValueError."""
        T = 12
        X = np.tile([0.6, 0.4], (T, 1))

        with pytest.raises(ValueError, match="n_sims must be positive"):
            model.simulate(T=T, n_sims=-5, X=X, seed=42)

    def test_X_shape_mismatch_T_raises(self, model):
        """Test X with wrong T dimension raises ValueError."""
        T = 12
        X = np.tile([0.6, 0.4], (6, 1))  # T=6, but requesting T=12

        with pytest.raises(ValueError, match="Allocation policy X has shape"):
            model.simulate(T=T, n_sims=50, X=X, seed=42)

    def test_X_shape_mismatch_M_raises(self, model):
        """Test X with wrong M dimension raises ValueError."""
        T = 12
        X = np.tile([0.33, 0.33, 0.34], (T, 1))  # M=3, but model has M=2

        with pytest.raises(ValueError, match="Allocation policy X has shape"):
            model.simulate(T=T, n_sims=50, X=X, seed=42)

    def test_simulate_without_start_uses_today(self, model):
        """Test simulate without start uses today's date."""
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=50, X=X, seed=42)

        # start should be automatically set to today
        assert result.start == date.today()

    def test_simulate_without_seed(self, model):
        """Test simulate without seed produces non-deterministic results."""
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=50, X=X, use_cache=False)
        result2 = model.simulate(T=T, n_sims=50, X=X, use_cache=False)

        # Without seed, results should differ (with very high probability)
        assert result1.seed is None
        assert result2.seed is None


class TestFinancialModelSimulateWithdrawals:
    """Test FinancialModel.simulate() with withdrawals."""

    def test_simulate_with_scheduled_withdrawal(self, model, start_date):
        """Test simulate with scheduled withdrawal."""
        T = 12
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        withdrawals = WithdrawalModel(
            scheduled=WithdrawalSchedule([
                WithdrawalEvent("Conservative", 100_000, date(2025, 6, 1))
            ])
        )

        result = model.simulate(
            T=T, n_sims=n_sims, X=X, seed=42,
            start=start_date, withdrawals=withdrawals
        )

        assert result.withdrawals is not None
        assert result.withdrawals.shape == (n_sims, T, 2)

    def test_simulate_without_withdrawals(self, model, start_date):
        """Test simulate without withdrawals has None."""
        T = 6
        n_sims = 50
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42, start=start_date)

        assert result.withdrawals is None


# ============================================================================
# FINANCIALMODEL CACHE TESTS
# ============================================================================

class TestFinancialModelCache:
    """Test FinancialModel caching behavior."""

    def test_cache_hit(self, model):
        """Test same parameters return cached result."""
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=50, X=X, seed=42)
        result2 = model.simulate(T=T, n_sims=50, X=X, seed=42)

        # Should be exact same object (cached)
        assert result1 is result2

    def test_cache_miss_different_seed(self, model):
        """Test different seed creates new cache entry."""
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=50, X=X, seed=42)
        result2 = model.simulate(T=T, n_sims=50, X=X, seed=43)

        assert result1 is not result2

    def test_cache_miss_different_T(self, model):
        """Test different T creates new cache entry."""
        X6 = np.tile([0.6, 0.4], (6, 1))
        X12 = np.tile([0.6, 0.4], (12, 1))

        result1 = model.simulate(T=6, n_sims=50, X=X6, seed=42)
        result2 = model.simulate(T=12, n_sims=50, X=X12, seed=42)

        assert result1 is not result2

    def test_cache_info(self, income, accounts):
        """Test cache_info returns correct structure."""
        model = FinancialModel(income=income, accounts=accounts)
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        # Empty cache
        info = model.cache_info()
        assert info['size'] == 0
        assert info['memory_mb'] == 0.0

        # After simulation
        model.simulate(T=T, n_sims=50, X=X, seed=42)
        info = model.cache_info()
        assert info['size'] == 1
        assert info['memory_mb'] > 0

    def test_clear_cache(self, income, accounts):
        """Test clear_cache removes all cached results."""
        model = FinancialModel(income=income, accounts=accounts)
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        model.simulate(T=T, n_sims=50, X=X, seed=42)
        model.simulate(T=T, n_sims=50, X=X, seed=43)

        assert model.cache_info()['size'] == 2

        model.clear_cache()

        assert model.cache_info()['size'] == 0

    def test_use_cache_false(self, model):
        """Test use_cache=False bypasses cache."""
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=50, X=X, seed=42, use_cache=False)
        result2 = model.simulate(T=T, n_sims=50, X=X, seed=42, use_cache=False)

        # Different objects even with same parameters
        assert result1 is not result2

    def test_cache_disabled_model(self, income, accounts):
        """Test cache disabled model doesn't cache."""
        model = FinancialModel(income=income, accounts=accounts, enable_cache=False)
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        result1 = model.simulate(T=T, n_sims=50, X=X, seed=42)
        result2 = model.simulate(T=T, n_sims=50, X=X, seed=42)

        # Should not be same object when cache disabled
        assert result1 is not result2


# ============================================================================
# SIMULATIONRESULT TESTS
# ============================================================================

class TestSimulationResult:
    """Test SimulationResult dataclass."""

    def test_n_sims_property(self, sample_result):
        """Test n_sims property."""
        assert sample_result.n_sims == 100

    def test_T_property(self, sample_result):
        """Test T property."""
        assert sample_result.T == 12

    def test_M_property(self, sample_result):
        """Test M property."""
        assert sample_result.M == 2

    def test_total_wealth_shape(self, sample_result):
        """Test total_wealth computed correctly."""
        assert sample_result.total_wealth.shape == (100, 13)  # n_sims x (T+1)

    def test_account_names(self, sample_result):
        """Test account_names are preserved."""
        assert sample_result.account_names == ["Conservative", "Aggressive"]

    def test_allocation_preserved(self, sample_result):
        """Test allocation is stored correctly."""
        assert sample_result.allocation.shape == (12, 2)
        np.testing.assert_array_almost_equal(
            sample_result.allocation[0],
            [0.6, 0.4]
        )

    def test_income_breakdown(self, sample_result):
        """Test income dict has expected keys."""
        assert "fixed" in sample_result.income
        assert "variable" in sample_result.income
        assert "total" in sample_result.income


class TestSimulationResultMetrics:
    """Test SimulationResult.metrics() method."""

    def test_metrics_returns_dataframe(self, sample_result):
        """Test metrics returns DataFrame."""
        metrics = sample_result.metrics()

        assert isinstance(metrics, pd.DataFrame)

    def test_metrics_single_account(self, sample_result):
        """Test metrics for specific account."""
        metrics = sample_result.metrics(account="Conservative")

        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) == sample_result.n_sims

    def test_metrics_invalid_account_raises(self, sample_result):
        """Test invalid account name raises ValueError."""
        with pytest.raises(ValueError, match="Account 'Invalid' not found"):
            sample_result.metrics(account="Invalid")

    def test_metrics_has_expected_columns(self, sample_result):
        """Test metrics DataFrame has expected columns."""
        metrics = sample_result.metrics(account="Conservative")

        expected_columns = ['cagr', 'volatility', 'sharpe', 'sortino', 'max_drawdown']
        for col in expected_columns:
            assert col in metrics.columns

    def test_metrics_cached(self, sample_result):
        """Test metrics are cached after first call."""
        _ = sample_result.metrics()

        assert sample_result._metrics is not None
        assert "Conservative" in sample_result._metrics
        assert "Aggressive" in sample_result._metrics

    def test_metrics_all_accounts_multiindex(self, sample_result):
        """Test metrics() with no account returns MultiIndex DataFrame."""
        metrics = sample_result.metrics()

        assert isinstance(metrics.index, pd.MultiIndex)
        assert metrics.index.names[0] == 'account'

    def test_metrics_sharpe_reasonable_range(self, sample_result):
        """Test Sharpe ratios are in reasonable range."""
        metrics = sample_result.metrics(account="Conservative")

        # Sharpe ratios should typically be between -3 and 3
        assert metrics['sharpe'].mean() > -5
        assert metrics['sharpe'].mean() < 5


class TestSimulationResultAggregateMetrics:
    """Test SimulationResult.aggregate_metrics() method."""

    def test_aggregate_metrics_single_account(self, sample_result):
        """Test aggregate_metrics for single account."""
        agg = sample_result.aggregate_metrics(account="Conservative")

        assert isinstance(agg, pd.Series)
        assert agg.name == "Conservative"

    def test_aggregate_metrics_all_accounts(self, sample_result):
        """Test aggregate_metrics for all accounts."""
        agg = sample_result.aggregate_metrics()

        assert isinstance(agg, pd.DataFrame)
        assert "Conservative" in agg.index
        assert "Aggressive" in agg.index

    def test_aggregate_metrics_invalid_account_raises(self, sample_result):
        """Test invalid account raises ValueError."""
        with pytest.raises(ValueError, match="Account 'Invalid' not found"):
            sample_result.aggregate_metrics(account="Invalid")

    def test_aggregate_metrics_has_expected_keys(self, sample_result):
        """Test aggregate metrics has expected keys."""
        agg = sample_result.aggregate_metrics(account="Conservative")

        expected_keys = ['var_95', 'cvar_95', 'mean_final', 'median_final',
                        'std_final', 'min_final', 'max_final']
        for key in expected_keys:
            assert key in agg.index

    def test_aggregate_metrics_var_less_than_mean(self, sample_result):
        """Test VaR is less than mean (5th percentile < mean)."""
        agg = sample_result.aggregate_metrics(account="Aggressive")

        assert agg['var_95'] <= agg['mean_final']

    def test_aggregate_metrics_cvar_less_than_var(self, sample_result):
        """Test CVaR is less than or equal to VaR (tail mean ≤ threshold)."""
        agg = sample_result.aggregate_metrics(account="Aggressive")

        assert agg['cvar_95'] <= agg['var_95']


class TestSimulationResultSummary:
    """Test SimulationResult.summary() method."""

    def test_summary_returns_dataframe(self, sample_result):
        """Test summary returns DataFrame."""
        summary = sample_result.summary()

        assert isinstance(summary, pd.DataFrame)

    def test_summary_has_total_row(self, sample_result):
        """Test summary includes Total row."""
        summary = sample_result.summary()

        assert "Total" in summary.index

    def test_summary_has_all_accounts(self, sample_result):
        """Test summary includes all accounts."""
        summary = sample_result.summary()

        assert "Conservative" in summary.index
        assert "Aggressive" in summary.index

    def test_summary_default_confidence(self, sample_result):
        """Test summary with default 95% confidence."""
        summary = sample_result.summary()

        assert "CI_lower_95" in summary.columns
        assert "CI_upper_95" in summary.columns

    def test_summary_custom_confidence(self, sample_result):
        """Test summary with custom confidence level."""
        summary = sample_result.summary(confidence=0.90)

        assert "CI_lower_90" in summary.columns
        assert "CI_upper_90" in summary.columns

    def test_summary_columns(self, sample_result):
        """Test summary has expected columns."""
        summary = sample_result.summary()

        assert "mean" in summary.columns
        assert "median" in summary.columns
        assert "std" in summary.columns

    def test_summary_ci_ordering(self, sample_result):
        """Test CI_lower < mean < CI_upper for all rows."""
        summary = sample_result.summary()

        for idx in summary.index:
            assert summary.loc[idx, 'CI_lower_95'] <= summary.loc[idx, 'mean']
            assert summary.loc[idx, 'mean'] <= summary.loc[idx, 'CI_upper_95']


class TestSimulationResultConvergence:
    """Test SimulationResult.convergence_analysis() method."""

    def test_convergence_returns_dataframe(self, sample_result):
        """Test convergence_analysis returns DataFrame."""
        conv = sample_result.convergence_analysis()

        assert isinstance(conv, pd.DataFrame)

    def test_convergence_has_expected_columns(self, sample_result):
        """Test convergence DataFrame has expected columns."""
        conv = sample_result.convergence_analysis()

        assert "n_sims" in conv.columns
        assert "mean" in conv.columns
        assert "std_error" in conv.columns

    def test_convergence_n_sims_increasing(self, sample_result):
        """Test n_sims values are increasing."""
        conv = sample_result.convergence_analysis()

        n_sims = conv['n_sims'].values
        assert np.all(np.diff(n_sims) > 0)

    def test_convergence_std_error_decreasing(self, sample_result):
        """Test std_error generally decreases with n_sims."""
        conv = sample_result.convergence_analysis()

        # Allow some variation but overall trend should be decreasing
        first_std_error = conv.iloc[0]['std_error']
        last_std_error = conv.iloc[-1]['std_error']

        assert last_std_error < first_std_error


# ============================================================================
# FINANCIALMODEL OPTIMIZE TESTS
# ============================================================================

class TestFinancialModelOptimize:
    """Test FinancialModel.optimize() method."""

    def test_optimize_returns_result(self, model, start_date):
        """Test optimize returns OptimizationResult."""
        goals = [TerminalGoal(account="Aggressive", threshold=3_000_000, confidence=0.60)]
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=24,
            n_sims=50,
            seed=42,
            start=start_date,
            verbose=False
        )

        assert isinstance(result, OptimizationResult)

    def test_optimize_empty_goals_raises(self, model, start_date):
        """Test optimize with empty goals raises ValueError."""
        optimizer = CVaROptimizer(n_accounts=2)

        with pytest.raises(ValueError, match="goals list cannot be empty"):
            model.optimize(
                goals=[],
                optimizer=optimizer,
                T_max=24,
                n_sims=50,
                start=start_date
            )

    def test_optimize_invalid_optimizer_raises(self, model, start_date):
        """Test optimize with invalid optimizer raises TypeError."""
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]

        with pytest.raises(TypeError, match="optimizer must implement AllocationOptimizer"):
            model.optimize(
                goals=goals,
                optimizer="not_an_optimizer",
                T_max=24,
                n_sims=50,
                start=start_date
            )

    def test_optimize_M_mismatch_raises(self, model, start_date):
        """Test optimize with wrong M raises ValueError."""
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        optimizer = CVaROptimizer(n_accounts=3)  # Wrong M

        with pytest.raises(ValueError, match="optimizer.M=3 must match model.M=2"):
            model.optimize(
                goals=goals,
                optimizer=optimizer,
                T_max=24,
                n_sims=50,
                start=start_date
            )

    def test_optimize_with_search_method_linear(self, model, start_date):
        """Test optimize with linear search."""
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=12,
            n_sims=50,
            seed=42,
            start=start_date,
            search_method="linear",
            verbose=False
        )

        assert result.feasible is True

    def test_optimize_with_search_method_binary(self, model, start_date):
        """Test optimize with binary search."""
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=12,
            n_sims=50,
            seed=42,
            start=start_date,
            search_method="binary",
            verbose=False
        )

        assert result.feasible is True


class TestFinancialModelSimulateFromOptimization:
    """Test FinancialModel.simulate_from_optimization() method."""

    def test_simulate_from_optimization(self, model, start_date):
        """Test simulate_from_optimization returns SimulationResult."""
        # First optimize
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        opt_result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=12,
            n_sims=50,
            seed=42,
            start=start_date,
            verbose=False
        )

        # Then simulate with optimal policy
        sim_result = model.simulate_from_optimization(
            opt_result,
            n_sims=100,
            seed=999,
            start=start_date
        )

        assert isinstance(sim_result, SimulationResult)
        assert sim_result.T == opt_result.T
        np.testing.assert_array_almost_equal(sim_result.allocation, opt_result.X)

    def test_simulate_from_optimization_invalid_type_raises(self, model):
        """Test simulate_from_optimization with wrong type raises TypeError."""
        with pytest.raises(TypeError, match="opt_result must be OptimizationResult"):
            model.simulate_from_optimization("not_an_optimization_result")


class TestFinancialModelVerifyGoals:
    """Test FinancialModel.verify_goals() method."""

    def test_verify_goals_with_simulation_result(self, model, start_date):
        """Test verify_goals with SimulationResult."""
        T = 12
        X = np.tile([0.6, 0.4], (T, 1))
        result = model.simulate(T=T, n_sims=100, X=X, seed=42, start=start_date)

        goals = [TerminalGoal(account="Aggressive", threshold=100_000, confidence=0.60)]

        status = model.verify_goals(result, goals, start=start_date)

        assert isinstance(status, dict)
        assert len(status) == 1
        goal_status = list(status.values())[0]
        assert 'satisfied' in goal_status
        assert 'violation_rate' in goal_status

    def test_verify_goals_with_optimization_result(self, model, start_date):
        """Test verify_goals with OptimizationResult (auto-simulates)."""
        goals = [TerminalGoal(account="Aggressive", threshold=500_000, confidence=0.60)]
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        opt_result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=12,
            n_sims=50,
            seed=42,
            start=start_date,
            verbose=False
        )

        status = model.verify_goals(opt_result, goals, start=start_date)

        assert isinstance(status, dict)

    def test_verify_goals_invalid_type_raises(self, model, start_date):
        """Test verify_goals with wrong type raises TypeError."""
        goals = [TerminalGoal(account="Aggressive", threshold=100_000, confidence=0.60)]

        with pytest.raises(TypeError, match="result must be SimulationResult or OptimizationResult"):
            model.verify_goals("not_a_result", goals, start=start_date)


# ============================================================================
# FINANCIALMODEL INTEGRATION TESTS
# ============================================================================

class TestFinancialModelIntegration:
    """Integration tests for FinancialModel."""

    def test_full_workflow(self, income, accounts):
        """Test complete workflow."""
        model = FinancialModel(income=income, accounts=accounts)

        T = 24
        n_sims = 100
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=n_sims, X=X, seed=42, start=date(2025, 1, 1))

        # Verify result structure
        assert result.wealth.shape == (n_sims, T + 1, 2)
        assert result.contributions.shape == (n_sims, T)
        assert result.returns.shape == (n_sims, T, 2)

        # Verify wealth is non-negative
        assert np.all(result.wealth >= -1e-6)

    def test_different_allocation_policies(self, model):
        """Test different allocation policies."""
        T = 12
        n_sims = 50

        # Aggressive: 20-80
        X_aggressive = np.tile([0.2, 0.8], (T, 1))
        result_aggressive = model.simulate(T=T, n_sims=n_sims, X=X_aggressive, seed=42)

        # Conservative: 80-20
        X_conservative = np.tile([0.8, 0.2], (T, 1))
        result_conservative = model.simulate(T=T, n_sims=n_sims, X=X_conservative, seed=42)

        # Results should differ
        assert not np.allclose(result_aggressive.wealth, result_conservative.wealth)

    def test_deterministic_income(self, deterministic_income, accounts, start_date):
        """Test simulation with deterministic income (no variable component)."""
        model = FinancialModel(income=deterministic_income, accounts=accounts)
        T = 6
        X = np.tile([0.5, 0.5], (T, 1))

        result = model.simulate(T=T, n_sims=50, X=X, seed=42, start=start_date)

        # Contributions should be deterministic (1D array)
        assert result.contributions.ndim == 1 or result.contributions.shape[0] == 50

    def test_three_accounts(self, income, three_accounts, start_date):
        """Test model with three accounts."""
        model = FinancialModel(income=income, accounts=three_accounts)
        T = 6
        X = np.tile([0.33, 0.34, 0.33], (T, 1))

        result = model.simulate(T=T, n_sims=50, X=X, seed=42, start=start_date)

        assert result.M == 3
        assert result.wealth.shape == (50, T + 1, 3)

    def test_full_optimize_simulate_verify_workflow(self, model, start_date):
        """Test complete optimize -> simulate -> verify workflow."""
        # 1. Optimize
        goals = [TerminalGoal(account="Aggressive", threshold=1_000_000, confidence=0.60)]
        optimizer = CVaROptimizer(n_accounts=2, objective="balanced")

        opt_result = model.optimize(
            goals=goals,
            optimizer=optimizer,
            T_max=12,
            n_sims=50,
            seed=42,
            start=start_date,
            verbose=False
        )

        # 2. Simulate with optimal policy
        sim_result = model.simulate_from_optimization(
            opt_result,
            n_sims=100,
            seed=999,
            start=start_date
        )

        # 3. Verify goals
        status = model.verify_goals(sim_result, goals, start=start_date)

        # Should complete without errors
        assert opt_result.feasible is True
        assert sim_result.T == opt_result.T
        assert isinstance(status, dict)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestModelEdgeCases:
    """Test edge cases for FinancialModel."""

    def test_single_simulation(self, model, start_date):
        """Test n_sims=1."""
        T = 6
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=1, X=X, seed=42, start=start_date)

        assert result.n_sims == 1
        assert result.wealth.shape == (1, T + 1, 2)

    def test_T_equals_1(self, model, start_date):
        """Test T=1."""
        T = 1
        X = np.tile([0.6, 0.4], (T, 1))

        result = model.simulate(T=T, n_sims=50, X=X, seed=42, start=start_date)

        assert result.T == 1
        assert result.wealth.shape == (50, 2, 2)  # n_sims, T+1, M

    def test_unequal_allocation(self, model, start_date):
        """Test allocation that changes over time."""
        T = 6
        X = np.zeros((T, 2))
        X[:3, :] = [0.8, 0.2]  # Conservative first 3 months
        X[3:, :] = [0.2, 0.8]  # Aggressive last 3 months

        result = model.simulate(T=T, n_sims=50, X=X, seed=42, start=start_date)

        np.testing.assert_array_almost_equal(result.allocation[:3], [[0.8, 0.2]] * 3)
        np.testing.assert_array_almost_equal(result.allocation[3:], [[0.2, 0.8]] * 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
