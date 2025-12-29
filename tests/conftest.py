"""
Pytest configuration and fixtures for FinOpt test suite.

This module provides reusable fixtures for testing all FinOpt components.
Fixtures follow the principle of "arrange-act-assert" with clear separation.
"""

from datetime import date
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.income import FixedIncome, VariableIncome, IncomeModel
from src.portfolio import Account, Portfolio
from src.returns import ReturnModel
from src.goals import IntermediateGoal, TerminalGoal


# ---------------------------------------------------------------------------
# Date Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def start_date() -> date:
    """Standard start date for tests."""
    return date(2025, 1, 1)


@pytest.fixture
def months() -> int:
    """Standard simulation horizon for tests."""
    return 24


@pytest.fixture
def n_sims() -> int:
    """Standard number of Monte Carlo scenarios for tests."""
    return 100


@pytest.fixture
def seed() -> int:
    """Standard random seed for reproducibility."""
    return 42


# ---------------------------------------------------------------------------
# Income Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixed_income() -> FixedIncome:
    """
    Simple fixed income stream with annual growth.

    Base: 1,500,000 CLP/month
    Growth: 3% annually
    """
    return FixedIncome(base=1_500_000, annual_growth=0.03)


@pytest.fixture
def variable_income() -> VariableIncome:
    """
    Variable income stream with moderate volatility.

    Base: 200,000 CLP/month
    Volatility: 10%
    No seasonality, floor, or cap
    """
    return VariableIncome(base=200_000, sigma=0.10, seed=42)


@pytest.fixture
def variable_income_seasonal() -> VariableIncome:
    """
    Variable income with seasonality (higher in Q4, lower in Q1).
    """
    seasonality = np.array([
        0.8, 0.8, 0.9,  # Q1: low
        1.0, 1.0, 1.0,  # Q2: normal
        1.0, 1.1, 1.1,  # Q3: normal-high
        1.2, 1.3, 1.2,  # Q4: high (bonuses)
    ])
    return VariableIncome(
        base=200_000,
        sigma=0.10,
        seasonality=seasonality,
        seed=42,
    )


@pytest.fixture
def income_model(fixed_income, variable_income) -> IncomeModel:
    """
    Standard income model with fixed + variable streams.

    Contribution rates: 30% fixed, 100% variable
    """
    return IncomeModel(
        fixed=fixed_income,
        variable=variable_income,
        monthly_contribution={"fixed": 0.3, "variable": 1.0},
    )


# ---------------------------------------------------------------------------
# Account Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def account_conservative() -> Account:
    """
    Conservative account (emergency fund, low-risk bonds).

    Return: 4% annually
    Volatility: 5% annually
    """
    return Account.from_annual(
        name="Conservative",
        annual_return=0.04,
        annual_volatility=0.05,
        initial_wealth=0,
    )


@pytest.fixture
def account_aggressive() -> Account:
    """
    Aggressive account (stocks, high growth).

    Return: 14% annually
    Volatility: 15% annually
    """
    return Account.from_annual(
        name="Aggressive",
        annual_return=0.14,
        annual_volatility=0.15,
        initial_wealth=0,
    )


@pytest.fixture
def accounts(account_conservative, account_aggressive) -> List[Account]:
    """Standard two-account portfolio."""
    return [account_conservative, account_aggressive]


@pytest.fixture
def accounts_with_wealth() -> List[Account]:
    """Two accounts with initial wealth."""
    return [
        Account.from_annual("Conservative", 0.04, 0.05, initial_wealth=1_000_000),
        Account.from_annual("Aggressive", 0.14, 0.15, initial_wealth=500_000),
    ]


# ---------------------------------------------------------------------------
# Portfolio Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def portfolio(accounts) -> Portfolio:
    """Standard portfolio with two accounts."""
    return Portfolio(accounts)


@pytest.fixture
def correlation_matrix() -> np.ndarray:
    """Standard correlation matrix for two accounts (moderate positive correlation)."""
    return np.array([
        [1.0, 0.3],
        [0.3, 1.0],
    ])


@pytest.fixture
def return_model(accounts, correlation_matrix) -> ReturnModel:
    """Return model for two-account portfolio with correlation."""
    return ReturnModel(accounts, default_correlation=correlation_matrix)


# ---------------------------------------------------------------------------
# Simulation Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def contributions_2d(months, n_sims) -> np.ndarray:
    """
    Sample contribution array (2D) for testing.

    Shape: (n_sims, months)
    Values: 450,000 - 550,000 CLP/month (varying across scenarios)
    """
    np.random.seed(42)
    base = 500_000
    noise = np.random.normal(0, 50_000, size=(n_sims, months))
    return np.maximum(base + noise, 0)  # Ensure non-negative


@pytest.fixture
def returns_3d(months, n_sims) -> np.ndarray:
    """
    Sample return array (3D) for two accounts.

    Shape: (n_sims, months, 2)
    Account 0: 4% annual (~0.33% monthly)
    Account 1: 14% annual (~1.1% monthly)
    """
    np.random.seed(42)

    # Conservative account (low return, low vol)
    mu0, sigma0 = 0.0033, 0.0144
    R0 = np.random.normal(mu0, sigma0, size=(n_sims, months))

    # Aggressive account (high return, high vol)
    mu1, sigma1 = 0.0110, 0.0433
    R1 = np.random.normal(mu1, sigma1, size=(n_sims, months))

    return np.stack([R0, R1], axis=2)


@pytest.fixture
def allocation_policy(months) -> np.ndarray:
    """
    Standard allocation policy: 60% conservative, 40% aggressive.

    Shape: (months, 2)
    """
    return np.tile([0.6, 0.4], (months, 1))


# ---------------------------------------------------------------------------
# Goal Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def intermediate_goal() -> IntermediateGoal:
    """
    Intermediate goal: Emergency fund by month 6.

    Target: 3,000,000 CLP
    Confidence: 80%
    """
    return IntermediateGoal(
        month=6,
        account="Conservative",
        threshold=3_000_000,
        confidence=0.80,
    )


@pytest.fixture
def terminal_goal() -> TerminalGoal:
    """
    Terminal goal: Retirement wealth at end of horizon.

    Target: 30,000,000 CLP
    Confidence: 80%
    """
    return TerminalGoal(
        account="Aggressive",
        threshold=30_000_000,
        confidence=0.80,
    )


@pytest.fixture
def goals_mixed(intermediate_goal, terminal_goal) -> List:
    """Mixed goals: 1 intermediate + 1 terminal."""
    return [intermediate_goal, terminal_goal]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_array_properties():
    """Factory for array validation assertions."""
    def _assert(arr, shape=None, dtype=None, finite=True, non_negative=False):
        """
        Assert array has expected properties.

        Parameters
        ----------
        arr : np.ndarray
            Array to validate
        shape : tuple, optional
            Expected shape
        dtype : type, optional
            Expected dtype
        finite : bool
            Whether all values should be finite
        non_negative : bool
            Whether all values should be >= 0
        """
        assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)}"

        if shape is not None:
            assert arr.shape == shape, f"Expected shape {shape}, got {arr.shape}"

        if dtype is not None:
            assert arr.dtype == dtype, f"Expected dtype {dtype}, got {arr.dtype}"

        if finite:
            assert np.all(np.isfinite(arr)), "Array contains non-finite values"

        if non_negative:
            assert np.all(arr >= 0), f"Array contains negative values: min={arr.min()}"

    return _assert


@pytest.fixture
def assert_simplex():
    """Factory for simplex constraint validation."""
    def _assert(X, tol=1e-6):
        """
        Assert allocation policy satisfies simplex constraints.

        Parameters
        ----------
        X : np.ndarray, shape (T, M)
            Allocation policy matrix
        tol : float
            Tolerance for constraint violations
        """
        # Non-negativity
        assert np.all(X >= -tol), f"Negative allocations found: min={X.min()}"

        # Sum to 1 along accounts dimension
        row_sums = X.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=tol), \
            f"Row sums not equal to 1: {row_sums}"

    return _assert
