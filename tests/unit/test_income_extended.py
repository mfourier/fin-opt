"""
Unit tests for IncomeModel expense integration.

Tests the new methods: gross_income, net_income, disposable_income, 
contributions_from_disposable.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date

from src.income import FixedIncome, VariableIncome, IncomeModel
from src.expenses import FixedExpense, VariableExpense, MicroExpense, ExpenseModel


class TestIncomeModelGrossIncome:
    """Tests for IncomeModel.gross_income()."""

    @pytest.fixture
    def simple_model(self):
        """Simple income model without expenses."""
        return IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, sigma=0.0)
        )

    def test_gross_income_basic(self, simple_model):
        """Gross income equals sum of fixed + variable."""
        gross = simple_model.gross_income(12)
        
        assert gross.shape == (12,)
        assert np.allclose(gross, 1_200_000)  # 1M fixed + 200k variable

    def test_gross_income_multi_sims(self, simple_model):
        """Gross income with multiple simulations."""
        gross = simple_model.gross_income(12, n_sims=100)
        
        assert gross.shape == (100, 12)

    def test_gross_income_equals_project(self, simple_model):
        """Gross income matches project() total."""
        gross = simple_model.gross_income(12, seed=42)
        result = simple_model.project(12, seed=42, output="array")
        
        assert np.allclose(gross, result["total"])


class TestIncomeModelNetIncome:
    """Tests for IncomeModel.net_income()."""

    @pytest.fixture
    def model_with_expenses(self):
        """Income model with fixed expense."""
        return IncomeModel(
            fixed=FixedIncome(base=1_500_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                fixed=FixedExpense(base=500_000)
            )
        )

    def test_net_income_deducts_expenses(self, model_with_expenses):
        """Net income = gross - expenses."""
        net = model_with_expenses.net_income(12)
        
        # Gross = 1.5M, Expenses = 500k, Net = 1M
        assert net.shape == (12,)
        assert np.allclose(net, 1_000_000)

    def test_net_income_no_expenses(self):
        """Net income equals gross when no expenses."""
        model = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, sigma=0)
        )
        
        gross = model.gross_income(12)
        net = model.net_income(12)
        
        assert np.allclose(net, gross)

    def test_net_income_can_be_negative(self):
        """Net income can be negative when expenses exceed income."""
        model = IncomeModel(
            fixed=FixedIncome(base=500_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                fixed=FixedExpense(base=700_000)  # Exceeds income
            )
        )
        
        net = model.net_income(12)
        
        assert np.all(net < 0)
        assert np.allclose(net, -200_000)

    def test_net_income_multi_sims(self, model_with_expenses):
        """Net income with multiple simulations."""
        net = model_with_expenses.net_income(12, n_sims=100)
        
        assert net.shape == (100, 12)


class TestIncomeModelDisposableIncome:
    """Tests for IncomeModel.disposable_income()."""

    def test_disposable_income_non_negative(self):
        """Disposable income clamped to zero."""
        model = IncomeModel(
            fixed=FixedIncome(base=500_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                fixed=FixedExpense(base=700_000)  # Exceeds income
            )
        )
        
        disposable = model.disposable_income(12)
        
        assert np.all(disposable >= 0)
        assert np.allclose(disposable, 0)  # All clamped to zero

    def test_disposable_income_positive_case(self):
        """Disposable income when net is positive."""
        model = IncomeModel(
            fixed=FixedIncome(base=1_500_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                fixed=FixedExpense(base=500_000)
            )
        )
        
        disposable = model.disposable_income(12)
        
        assert np.all(disposable > 0)
        assert np.allclose(disposable, 1_000_000)

    def test_disposable_equals_max_zero_net(self):
        """Disposable = max(0, net)."""
        model = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=100_000, sigma=0.3, seed=42),
            expenses=ExpenseModel(
                variable=VariableExpense(base=800_000, sigma=0.3, seed=43)
            )
        )
        
        net = model.net_income(12, n_sims=100, seed=42)
        disposable = model.disposable_income(12, n_sims=100, seed=42)
        
        expected = np.maximum(net, 0)
        assert np.allclose(disposable, expected)

    def test_disposable_income_multi_sims(self):
        """Disposable income with multiple simulations."""
        model = IncomeModel(
            fixed=FixedIncome(base=1_500_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(fixed=FixedExpense(base=500_000))
        )
        
        disposable = model.disposable_income(12, n_sims=100)
        
        assert disposable.shape == (100, 12)
        assert np.all(disposable >= 0)


class TestIncomeModelContributionsFromDisposable:
    """Tests for IncomeModel.contributions_from_disposable()."""

    @pytest.fixture
    def model_with_expenses(self):
        """Income model with expenses."""
        return IncomeModel(
            fixed=FixedIncome(base=1_500_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                fixed=FixedExpense(base=500_000)
            )
        )

    def test_contributions_constant_rate(self, model_with_expenses):
        """Contributions with constant savings rate."""
        # Disposable = 1M per month
        contrib = model_with_expenses.contributions_from_disposable(12, savings_rate=0.3)
        
        assert contrib.shape == (12,)
        assert np.allclose(contrib, 300_000)  # 30% of 1M

    def test_contributions_zero_when_no_disposable(self):
        """Zero contributions when expenses exceed income."""
        model = IncomeModel(
            fixed=FixedIncome(base=500_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(fixed=FixedExpense(base=700_000))
        )
        
        contrib = model.contributions_from_disposable(12, savings_rate=0.5)
        
        assert np.allclose(contrib, 0)

    def test_contributions_rotating_rate(self, model_with_expenses):
        """Contributions with 12-month rotating rates."""
        # Higher savings rate in December
        rates = [0.3] * 11 + [0.5]
        contrib = model_with_expenses.contributions_from_disposable(
            12, savings_rate=np.array(rates), start=date(2025, 1, 1)
        )
        
        assert contrib[11] == pytest.approx(500_000)  # 50% of 1M in December
        assert contrib[0] == pytest.approx(300_000)   # 30% of 1M in January

    def test_contributions_multi_sims(self, model_with_expenses):
        """Contributions with multiple simulations."""
        contrib = model_with_expenses.contributions_from_disposable(
            12, savings_rate=0.3, n_sims=100
        )
        
        assert contrib.shape == (100, 12)

    def test_contributions_vs_original_method(self):
        """Compare with original contributions() when no expenses."""
        model = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=0, sigma=0)
        )
        
        # Original method: 30% of fixed by default
        original = model.contributions(12, output="array")
        
        # New method: same rate, no expenses
        new = model.contributions_from_disposable(12, savings_rate=0.3)
        
        # Should be identical (no expenses, same rate)
        assert np.allclose(original, new)


class TestIncomeModelBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_original_methods_still_work(self):
        """Original methods work without expenses parameter."""
        model = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000, sigma=0.1, seed=42)
        )
        
        # Original methods should work unchanged
        projection = model.project(12, output="dataframe")
        assert isinstance(projection, pd.DataFrame)
        assert "total" in projection.columns
        
        contrib = model.contributions(12, output="series")
        assert isinstance(contrib, pd.Series)

    def test_expenses_parameter_optional(self):
        """Expenses parameter is optional."""
        # Should work without expenses
        model = IncomeModel(
            fixed=FixedIncome(base=1_000_000),
            variable=VariableIncome(base=200_000)
        )
        
        assert model.expenses is None
        
        # New methods should work too
        gross = model.gross_income(12)
        net = model.net_income(12)
        
        assert np.allclose(gross, net)  # No expenses means gross == net


class TestIncomeModelWithStochasticExpenses:
    """Tests with stochastic expense models."""

    def test_stochastic_expenses_create_variance(self):
        """Stochastic expenses create variance in net income."""
        model = IncomeModel(
            fixed=FixedIncome(base=2_000_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                variable=VariableExpense(base=500_000, sigma=0.2, seed=42)
            )
        )
        
        net = model.net_income(12, n_sims=1000, seed=42)
        
        # Should have variance from stochastic expenses
        assert net.std() > 0

    def test_micro_expenses_impact(self):
        """Micro-expenses reduce disposable income."""
        model_no_micro = IncomeModel(
            fixed=FixedIncome(base=2_000_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                fixed=FixedExpense(base=500_000)
            )
        )
        
        model_with_micro = IncomeModel(
            fixed=FixedIncome(base=2_000_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(
                fixed=FixedExpense(base=500_000),
                micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500, seed=42)
            )
        )
        
        disposable_no_micro = model_no_micro.disposable_income(12, n_sims=100, seed=42)
        disposable_with_micro = model_with_micro.disposable_income(12, n_sims=100, seed=42)
        
        # Micro-expenses reduce average disposable income
        assert disposable_with_micro.mean() < disposable_no_micro.mean()

    def test_contributions_reduction_from_waste(self):
        """More expenses = less contributions."""
        model_low_expense = IncomeModel(
            fixed=FixedIncome(base=2_000_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(fixed=FixedExpense(base=500_000))
        )
        
        model_high_expense = IncomeModel(
            fixed=FixedIncome(base=2_000_000),
            variable=VariableIncome(base=0, sigma=0),
            expenses=ExpenseModel(fixed=FixedExpense(base=1_000_000))
        )
        
        contrib_low = model_low_expense.contributions_from_disposable(12, savings_rate=0.5)
        contrib_high = model_high_expense.contributions_from_disposable(12, savings_rate=0.5)
        
        # Higher expenses = lower contributions
        assert contrib_high.sum() < contrib_low.sum()
        assert np.allclose(contrib_low, 750_000)   # 50% of (2M - 500k)
        assert np.allclose(contrib_high, 500_000)  # 50% of (2M - 1M)
