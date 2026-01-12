"""
Expense modeling module for FinOpt.

Purpose
-------
Models monthly expenses (gastos) that reduce disposable income before savings.
Mirrors the income module structure with three expense types:

- FixedExpense: Deterministic expenses (rent, utilities, loan payments)
- VariableExpense: Stochastic expenses with seasonality (food, transport)
- MicroExpense: Compound Poisson micro-expenses (coffee, impulse buys)

The total expense at time t is:
    C_t = C_t^fixed + C_t^variable + C_t^micro

Design principles
-----------------
- Mirrors income.py structure for consistency
- Frozen dataclasses for immutability
- Vectorized n_sims support for Monte Carlo
- Calendar-aware outputs with pandas Series/DataFrame

Example
-------
>>> from datetime import date
>>> from src.expenses import FixedExpense, VariableExpense, MicroExpense, ExpenseModel
>>> fe = FixedExpense(base=500_000, annual_inflation=0.04)
>>> ve = VariableExpense(base=200_000, sigma=0.1, seasonality=[1.0]*11 + [1.5])
>>> me = MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500)
>>> expenses = ExpenseModel(fixed=fe, variable=ve, micro=me)
>>> result = expenses.project(12, n_sims=1000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, Optional, Literal, Union, Tuple

import numpy as np
import pandas as pd

from .utils import (
    check_non_negative,
    annual_to_monthly,
    month_index,
    normalize_start_month,
)

__all__ = [
    "FixedExpense",
    "VariableExpense",
    "MicroExpense",
    "ExpenseModel",
    "ExpenseMetrics",
]


@dataclass(frozen=True)
class ExpenseMetrics:
    """Summary metrics for expense projections."""
    months: int
    total_fixed: float
    total_variable: float
    total_micro: float
    total_expenses: float
    mean_fixed: float
    mean_variable: float
    mean_micro: float
    mean_total: float
    std_variable: float
    std_micro: float
    std_total: float


# ---------------------------------------------------------------------------
# Expense Streams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedExpense:
    """
    Deterministic monthly expense with optional inflation and step changes.

    Purpose
    -------
    Models predictable, recurring expenses such as rent, utilities,
    loan payments, or subscriptions. Optionally applies an annual inflation
    rate, which is internally converted to a compounded monthly rate.

    Supports scheduled step changes at specific dates (e.g., rent increase).

    Parameters
    ----------
    base : float
        Monthly base expense at t=0 (e.g., 500_000 CLP for rent).
        Must be non-negative.
    annual_inflation : float, default 0.0
        Nominal annual inflation rate (e.g., 0.04 for 4%/year).
        Internally converted to monthly compounded rate:
            m = (1 + annual_inflation) ** (1/12) - 1
        The projected expense follows:
            c_t = base * (1 + m)^t + step_adjustments
    step_changes : Optional[Dict[date, float]], default None
        Dictionary mapping dates to absolute change amounts (additive).
        Changes are applied from the first month >= the specified date.
        Example: {date(2026, 1, 1): 50_000}  # rent increase
    name : str, default "fixed_expense"
        Identifier for labeling outputs.

    Methods
    -------
    project(months, start=None, output="array", n_sims=1)
        Returns deterministic monthly expense projection.
        When n_sims > 1, the projection is replicated (all identical).

    Examples
    --------
    >>> from datetime import date
    >>> fe = FixedExpense(
    ...     base=500_000,
    ...     annual_inflation=0.04,
    ...     step_changes={date(2025, 7, 1): 50_000}
    ... )
    >>> fe.project(12, start=date(2025, 1, 1))
    array([500000., 501636., ..., 556636., ...])  # step applied from month 6
    """
    base: float
    annual_inflation: float = 0.0
    step_changes: Optional[Dict[date, float]] = None
    name: str = "fixed_expense"

    def __post_init__(self) -> None:
        check_non_negative("base", self.base)
        if self.annual_inflation < -1.0:
            raise ValueError("annual_inflation must be >= -1.0")

    def project(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        output: Literal["array", "series"] = "array",
        n_sims: int = 1,
    ) -> Union[np.ndarray, pd.Series]:
        """
        Project deterministic monthly expense stream.

        Parameters
        ----------
        months : int
            Number of months to project. Must be >= 0.
        start : date, optional
            Start date for calendar alignment (used for step_changes).
            If None, step_changes are ignored.
        output : {"array", "series"}, default "array"
            Output format.
        n_sims : int, default 1
            Number of simulations. All will be identical (deterministic).

        Returns
        -------
        np.ndarray or pd.Series
            If n_sims=1 and output="array": shape (months,)
            If n_sims>1 and output="array": shape (n_sims, months)
            If output="series": pandas Series with DatetimeIndex (n_sims must be 1)
        """
        if months <= 0:
            if output == "series":
                return pd.Series([], dtype=float, name=self.name)
            return np.array([]) if n_sims == 1 else np.zeros((n_sims, 0))

        # Monthly compounded inflation rate
        m = annual_to_monthly(self.annual_inflation)
        t = np.arange(months)
        base_projection = self.base * (1 + m) ** t

        # Apply step changes if start date provided
        if self.step_changes and start is not None:
            step_adjustments = np.zeros(months)
            for change_date, amount in self.step_changes.items():
                # Calculate month offset from start
                month_offset = (change_date.year - start.year) * 12 + (change_date.month - start.month)
                if 0 <= month_offset < months:
                    step_adjustments[month_offset:] += amount
            base_projection = base_projection + step_adjustments

        # Ensure non-negative
        base_projection = np.maximum(base_projection, 0.0)

        if output == "series":
            if n_sims != 1:
                raise ValueError("Series output requires n_sims=1")
            idx = month_index(start, months)
            return pd.Series(base_projection, index=idx, name=self.name)

        if n_sims == 1:
            return base_projection
        else:
            # Replicate for n_sims (all identical)
            return np.tile(base_projection, (n_sims, 1))


@dataclass(frozen=True)
class VariableExpense:
    """
    Stochastic monthly expense with seasonality and noise.

    Purpose
    -------
    Models variable expenses such as food, transport, entertainment,
    or utilities that fluctuate month-to-month. Supports:
    - 12-month seasonality factors (e.g., higher in December)
    - Gaussian noise as fraction of mean
    - Floor/cap guardrails
    - Optional annual inflation

    Mathematical Model
    ------------------
    C_t^variable = base * (1+m)^t * s_{t mod 12} * (1 + epsilon_t)
    
    where:
    - m = monthly inflation rate
    - s_k = seasonality factor for month k (1-indexed calendar month)
    - epsilon_t ~ N(0, sigma^2)
    
    Result is clipped to [floor, cap].

    Parameters
    ----------
    base : float
        Baseline monthly expense before growth/seasonality.
    seasonality : Optional[Iterable[float]], default None
        12 multiplicative factors for each calendar month (Jan=0, Dec=11).
        If None, all factors are 1.0.
    sigma : float, default 0.0
        Gaussian noise std as fraction of mean. sigma=0.1 means ±10% noise.
    floor : Optional[float], default 0.0
        Minimum expense (cannot be negative by default).
    cap : Optional[float], default None
        Maximum expense. If None, no upper bound.
    annual_inflation : float, default 0.0
        Nominal annual growth rate.
    name : str, default "variable_expense"
    seed : Optional[int], default None
        Random seed for reproducibility.

    Methods
    -------
    project(months, start=None, seed=None, output="array", n_sims=1)
        Stochastic projection with shape (n_sims, months) or (months,).

    Examples
    --------
    >>> ve = VariableExpense(
    ...     base=200_000,
    ...     seasonality=[1.0]*11 + [1.5],  # December 50% higher
    ...     sigma=0.1,
    ...     seed=42
    ... )
    >>> ve.project(12, start=date(2025, 1, 1), n_sims=100).shape
    (100, 12)
    """
    base: float
    seasonality: Optional[Iterable[float]] = None
    sigma: float = 0.0
    floor: Optional[float] = 0.0
    cap: Optional[float] = None
    annual_inflation: float = 0.0
    name: str = "variable_expense"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        check_non_negative("base", self.base)
        check_non_negative("sigma", self.sigma)
        if self.floor is not None and self.floor < 0:
            raise ValueError("floor must be non-negative")
        if self.cap is not None and self.floor is not None and self.cap < self.floor:
            raise ValueError("cap must be >= floor")
        if self.seasonality is not None:
            s = list(self.seasonality)
            if len(s) != 12:
                raise ValueError("seasonality must have exactly 12 elements")
            if any(x < 0 for x in s):
                raise ValueError("seasonality factors must be non-negative")

    def project(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        output: Literal["array", "series"] = "array",
        n_sims: int = 1,
    ) -> Union[np.ndarray, pd.Series]:
        """
        Project stochastic monthly expense stream.

        Parameters
        ----------
        months : int
            Number of months to project.
        start : date, optional
            Start date for calendar alignment and seasonality.
        seed : int, optional
            Override instance seed for this projection.
        output : {"array", "series"}, default "array"
        n_sims : int, default 1
            Number of Monte Carlo simulations.

        Returns
        -------
        np.ndarray or pd.Series
            If n_sims=1 and output="array": shape (months,)
            If n_sims>1 and output="array": shape (n_sims, months)
            If output="series": mean across simulations with DatetimeIndex
        """
        if months <= 0:
            if output == "series":
                return pd.Series([], dtype=float, name=self.name)
            return np.array([]) if n_sims == 1 else np.zeros((n_sims, 0))

        rng_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(rng_seed)

        # Monthly inflation rate
        m = annual_to_monthly(self.annual_inflation)
        t = np.arange(months)
        growth = (1 + m) ** t  # shape (months,)

        # Seasonality factors
        if self.seasonality is not None:
            s = np.array(list(self.seasonality))
            offset = normalize_start_month(start)
            month_indices = (offset + t) % 12
            seasonality_factors = s[month_indices]  # shape (months,)
        else:
            seasonality_factors = np.ones(months)

        # Base mean path
        mean_path = self.base * growth * seasonality_factors  # shape (months,)

        # Generate noise
        if self.sigma > 0:
            epsilon = rng.normal(0, self.sigma, size=(n_sims, months))
            projection = mean_path[None, :] * (1 + epsilon)  # (n_sims, months)
        else:
            projection = np.tile(mean_path, (n_sims, 1))  # (n_sims, months)

        # Apply floor and cap
        if self.floor is not None:
            projection = np.maximum(projection, self.floor)
        if self.cap is not None:
            projection = np.minimum(projection, self.cap)

        # Format output
        if output == "series":
            idx = month_index(start, months)
            mean_proj = projection.mean(axis=0) if n_sims > 1 else projection[0]
            return pd.Series(mean_proj, index=idx, name=self.name)

        if n_sims == 1:
            return projection[0]
        return projection


@dataclass(frozen=True)
class MicroExpense:
    """
    Compound Poisson micro-expense process.

    Purpose
    -------
    Models frequent small purchases (coffee, snacks, impulse buys)
    as a compound Poisson process:
        C_t^micro = sum_{j=1}^{N_t} S_j

    where:
    - N_t ~ Poisson(lambda_t) is the number of events
    - S_j ~ severity distribution (lognormal or gamma)

    Parameters
    ----------
    lambda_base : float
        Base expected number of events per month (e.g., 30 for daily).
    severity_mean : float
        Mean expense per event (e.g., 2_000 CLP).
    severity_std : float
        Std of expense per event.
    severity_distribution : {"lognormal", "gamma"}, default "lognormal"
        Distribution for individual expense amounts.
    lambda_seasonality : Optional[Iterable[float]], default None
        12-month factors for event frequency (e.g., higher in December).
    name : str, default "micro_expense"
    seed : Optional[int], default None

    Methods
    -------
    project(months, start=None, seed=None, output="array", n_sims=1)
        Compound Poisson projection.
    expected_monthly()
        Returns E[C_micro] = lambda * E[S].

    Mathematical Notes
    ------------------
    For lognormal severity with mean mu and std sigma:
        sigma_ln^2 = log(1 + (sigma/mu)^2)
        mu_ln = log(mu) - sigma_ln^2 / 2
        S ~ LogNormal(mu_ln, sigma_ln)

    Expected value: E[C_micro] = lambda * E[S] = lambda * mu
    Variance: Var[C_micro] = lambda * (Var[S] + E[S]^2)

    Examples
    --------
    >>> me = MicroExpense(
    ...     lambda_base=30,
    ...     severity_mean=2_000,
    ...     severity_std=500,
    ...     seed=42
    ... )
    >>> me.expected_monthly()
    60000.0
    >>> me.project(12, n_sims=1000).mean()  # Should be close to 60000
    """
    lambda_base: float
    severity_mean: float
    severity_std: float
    severity_distribution: Literal["lognormal", "gamma"] = "lognormal"
    lambda_seasonality: Optional[Iterable[float]] = None
    name: str = "micro_expense"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        check_non_negative("lambda_base", self.lambda_base)
        check_non_negative("severity_mean", self.severity_mean)
        check_non_negative("severity_std", self.severity_std)
        if self.lambda_seasonality is not None:
            s = list(self.lambda_seasonality)
            if len(s) != 12:
                raise ValueError("lambda_seasonality must have exactly 12 elements")
            if any(x < 0 for x in s):
                raise ValueError("lambda_seasonality factors must be non-negative")

    def _get_lognormal_params(self) -> Tuple[float, float]:
        """Convert mean/std to lognormal mu_ln, sigma_ln parameters."""
        mu = self.severity_mean
        sigma = self.severity_std
        if mu <= 0:
            raise ValueError("severity_mean must be positive for lognormal")
        sigma_ln_sq = np.log(1 + (sigma / mu) ** 2)
        sigma_ln = np.sqrt(sigma_ln_sq)
        mu_ln = np.log(mu) - sigma_ln_sq / 2
        return mu_ln, sigma_ln

    def _get_gamma_params(self) -> Tuple[float, float]:
        """Convert mean/std to gamma shape (alpha) and scale (beta) parameters."""
        mu = self.severity_mean
        sigma = self.severity_std
        if sigma <= 0:
            # Degenerate case: constant severity
            return mu, 0.0
        alpha = (mu / sigma) ** 2  # shape
        beta = sigma ** 2 / mu  # scale
        return alpha, beta

    def expected_monthly(self) -> float:
        """Return expected monthly expense: E[C_micro] = lambda * E[S]."""
        return self.lambda_base * self.severity_mean

    def project(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        seed: Optional[int] = None,
        output: Literal["array", "series"] = "array",
        n_sims: int = 1,
    ) -> Union[np.ndarray, pd.Series]:
        """
        Project compound Poisson micro-expense stream.

        Parameters
        ----------
        months : int
            Number of months to project.
        start : date, optional
            Start date for calendar alignment and lambda_seasonality.
        seed : int, optional
            Override instance seed.
        output : {"array", "series"}, default "array"
        n_sims : int, default 1

        Returns
        -------
        np.ndarray or pd.Series
            Monthly aggregate micro-expenses.
        """
        if months <= 0:
            if output == "series":
                return pd.Series([], dtype=float, name=self.name)
            return np.array([]) if n_sims == 1 else np.zeros((n_sims, 0))

        rng_seed = seed if seed is not None else self.seed
        rng = np.random.default_rng(rng_seed)

        # Compute lambda for each month (with seasonality)
        if self.lambda_seasonality is not None:
            s = np.array(list(self.lambda_seasonality))
            offset = normalize_start_month(start)
            month_indices = (offset + np.arange(months)) % 12
            lambdas = self.lambda_base * s[month_indices]  # shape (months,)
        else:
            lambdas = np.full(months, self.lambda_base)

        # Get severity distribution parameters
        if self.severity_distribution == "lognormal":
            mu_ln, sigma_ln = self._get_lognormal_params()
        else:
            alpha, beta = self._get_gamma_params()

        # Generate compound Poisson for each (sim, month)
        result = np.zeros((n_sims, months), dtype=float)

        for t in range(months):
            # Number of events for all simulations
            N_t = rng.poisson(lambdas[t], size=n_sims)
            
            # For each simulation, sum severity of N_t events
            for i in range(n_sims):
                n_events = N_t[i]
                if n_events > 0:
                    if self.severity_distribution == "lognormal":
                        severities = rng.lognormal(mu_ln, sigma_ln, size=n_events)
                    else:
                        if beta > 0:
                            severities = rng.gamma(alpha, beta, size=n_events)
                        else:
                            severities = np.full(n_events, self.severity_mean)
                    result[i, t] = severities.sum()

        # Format output
        if output == "series":
            idx = month_index(start, months)
            mean_proj = result.mean(axis=0) if n_sims > 1 else result[0]
            return pd.Series(mean_proj, index=idx, name=self.name)

        if n_sims == 1:
            return result[0]
        return result


# ---------------------------------------------------------------------------
# Expense Model (Facade)
# ---------------------------------------------------------------------------

@dataclass
class ExpenseModel:
    """
    Unified expense model combining fixed, variable, and micro-expenses.

    Purpose
    -------
    Facade for combining multiple expense streams into a single projection.
    Provides aggregated metrics and visualization.

    Parameters
    ----------
    fixed : Optional[FixedExpense], default None
    variable : Optional[VariableExpense], default None
    micro : Optional[MicroExpense], default None

    Methods
    -------
    project(months, start=None, output="series", seed=None, n_sims=1)
        Total expense projection combining all streams.
    summary(months, start=None, n_sims=500, seed=None)
        Statistical summary of expenses.
    expected_monthly()
        Expected monthly expense (sum of all components).

    Examples
    --------
    >>> em = ExpenseModel(
    ...     fixed=FixedExpense(base=500_000),
    ...     variable=VariableExpense(base=200_000, sigma=0.1),
    ...     micro=MicroExpense(lambda_base=30, severity_mean=2_000, severity_std=500)
    ... )
    >>> result = em.project(12, n_sims=100, output="array")
    >>> result["total"].shape
    (100, 12)
    """
    fixed: Optional[FixedExpense] = None
    variable: Optional[VariableExpense] = None
    micro: Optional[MicroExpense] = None

    def __post_init__(self) -> None:
        if self.fixed is None and self.variable is None and self.micro is None:
            raise ValueError("At least one expense stream must be provided")

    def project(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        output: Literal["array", "dataframe", "series"] = "array",
        seed: Optional[int] = None,
        n_sims: int = 1,
    ) -> Union[Dict[str, np.ndarray], pd.DataFrame, pd.Series]:
        """
        Project total expenses combining all streams.

        Parameters
        ----------
        months : int
            Number of months to project.
        start : date, optional
            Start date for calendar alignment.
        output : {"array", "dataframe", "series"}, default "array"
            - "array": dict with "fixed", "variable", "micro", "total" keys
            - "dataframe": pandas DataFrame (n_sims must be 1, or returns mean)
            - "series": pandas Series of total (mean across simulations)
        seed : int, optional
            Random seed for stochastic components.
        n_sims : int, default 1
            Number of Monte Carlo simulations.

        Returns
        -------
        dict, pd.DataFrame, or pd.Series
            Expense projections in requested format.
        """
        if months <= 0:
            if output == "dataframe":
                return pd.DataFrame()
            if output == "series":
                return pd.Series([], dtype=float, name="total_expenses")
            return {"fixed": np.array([]), "variable": np.array([]), 
                    "micro": np.array([]), "total": np.array([])}

        # Manage seeds for reproducibility
        if seed is not None:
            seed_fixed = seed
            seed_variable = seed + 1
            seed_micro = seed + 2
        else:
            seed_fixed = None
            seed_variable = None
            seed_micro = None

        # Project each component
        if self.fixed is not None:
            fixed_proj = self.fixed.project(months, start=start, output="array", n_sims=n_sims)
            if n_sims == 1:
                fixed_proj = fixed_proj[None, :]  # (1, months)
        else:
            fixed_proj = np.zeros((n_sims, months))

        if self.variable is not None:
            variable_proj = self.variable.project(
                months, start=start, seed=seed_variable, output="array", n_sims=n_sims
            )
            if n_sims == 1:
                variable_proj = variable_proj[None, :]
        else:
            variable_proj = np.zeros((n_sims, months))

        if self.micro is not None:
            micro_proj = self.micro.project(
                months, start=start, seed=seed_micro, output="array", n_sims=n_sims
            )
            if n_sims == 1:
                micro_proj = micro_proj[None, :]
        else:
            micro_proj = np.zeros((n_sims, months))

        total_proj = fixed_proj + variable_proj + micro_proj

        # Format output
        if output == "array":
            if n_sims == 1:
                return {
                    "fixed": fixed_proj[0],
                    "variable": variable_proj[0],
                    "micro": micro_proj[0],
                    "total": total_proj[0],
                }
            return {
                "fixed": fixed_proj,
                "variable": variable_proj,
                "micro": micro_proj,
                "total": total_proj,
            }

        if output == "dataframe":
            idx = month_index(start, months)
            return pd.DataFrame({
                "fixed": fixed_proj.mean(axis=0),
                "variable": variable_proj.mean(axis=0),
                "micro": micro_proj.mean(axis=0),
                "total": total_proj.mean(axis=0),
            }, index=idx)

        if output == "series":
            idx = month_index(start, months)
            return pd.Series(total_proj.mean(axis=0), index=idx, name="total_expenses")

        raise ValueError(f"Unknown output format: {output}")

    def expected_monthly(self) -> float:
        """Return expected monthly expense (deterministic + stochastic means)."""
        total = 0.0
        if self.fixed is not None:
            total += self.fixed.base
        if self.variable is not None:
            total += self.variable.base
        if self.micro is not None:
            total += self.micro.expected_monthly()
        return total

    def summary(
        self,
        months: int,
        *,
        start: Optional[date] = None,
        n_sims: int = 500,
        seed: Optional[int] = None,
    ) -> ExpenseMetrics:
        """
        Compute statistical summary of expense projections.

        Parameters
        ----------
        months : int
            Projection horizon.
        start : date, optional
            Start date.
        n_sims : int, default 500
            Number of simulations for statistics.
        seed : int, optional
            Random seed.

        Returns
        -------
        ExpenseMetrics
            Summary statistics including totals, means, and standard deviations.
        """
        result = self.project(months, start=start, output="array", seed=seed, n_sims=n_sims)

        # Handle both single and multi-sim cases
        if isinstance(result["total"], np.ndarray) and result["total"].ndim == 2:
            fixed = result["fixed"]
            variable = result["variable"]
            micro = result["micro"]
            total = result["total"]
        else:
            fixed = result["fixed"][None, :]
            variable = result["variable"][None, :]
            micro = result["micro"][None, :]
            total = result["total"][None, :]

        return ExpenseMetrics(
            months=months,
            total_fixed=fixed.sum(axis=1).mean(),
            total_variable=variable.sum(axis=1).mean(),
            total_micro=micro.sum(axis=1).mean(),
            total_expenses=total.sum(axis=1).mean(),
            mean_fixed=fixed.mean(),
            mean_variable=variable.mean(),
            mean_micro=micro.mean(),
            mean_total=total.mean(),
            std_variable=variable.std(),
            std_micro=micro.std(),
            std_total=total.std(),
        )
