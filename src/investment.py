"""
Investment modeling module for FinOpt.

Purpose
-------
Models portfolio allocation, wealth evolution, and return sampling for the optimization
framework. This module implements the mathematical foundation for the wealth optimization
problem where we seek allocation policies X that distribute total contributions A_t
across multiple accounts to meet financial goals.

Key Mathematical Framework
--------------------------
- Total monthly contribution: A_t (from income.py)
- Allocation policy: X = {x_t^m} where x_t^m ≥ 0, Σ_m x_t^m = 1
- Account contribution: A_t^m = x_t^m * A_t
- Wealth evolution: W_{t+1}^m = (W_t^m + A_t^m)(1 + R_t^m)
- Affine representation: W_t^m = W_0^m * F_{0,t}^m + Σ_{s=0}^{t-1} A_s * x_s^m * F_{s,t}^m

Key components
--------------
- Account:
    Represents a single investment account (e.g., emergency, housing, brokerage)
    with return distribution parameters and risk characteristics.

- Portfolio: 
    Manages multiple accounts, handles cross-correlations, and provides
    the core optimization interface. Computes wealth evolution given
    allocation policies X and total contributions A_t, and can also
    generate Monte Carlo samples of monthly returns for each account.

Design principles
-----------------
- Optimization-centric: All methods designed to work with allocation matrices X
- Mathematical consistency: Implements exact formulas from optimization theory
- Efficient computation: Vectorized operations for Monte Carlo scenarios
- Clear separation: Accounts define returns, Portfolio handles allocation logic

Example
-------
>>> import numpy as np
>>> from datetime import date
>>> # Create accounts
>>> emergency = Account.from_gaussian("emergency", 0.03, 0.01, initial_wealth=100000)
>>> housing = Account.from_gaussian("housing", 0.08, 0.12, initial_wealth=50000)
>>> portfolio = Portfolio([emergency, housing])
>>> 
>>> # Define allocation policy (50-50 split)
>>> months = 12
>>> X = np.full((months, 2), 0.5)  # allocation matrix
>>> A = np.full(months, 100000.0)   # total contributions from income
>>> 
>>> # Sample returns and compute wealth evolution
>>> returns = portfolio.returns(months=12, seed=42)  # sample Monte Carlo returns
>>> wealth_path = portfolio.wealth(
...     contributions=A,
...     allocation_policy=X, 
...     returns=returns
... )
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, Literal, Sequence, Mapping
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

from .utils import (
    check_non_negative,
    annual_to_monthly,
)

__all__ = [
    "Account",
    "Portfolio", 
    "ReturnModel",
    "GaussianReturns",
    "StudentTReturns",
]


# ---------------------------------------------------------------------------
# Return Models
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class ReturnModel:
    """Base class for return distribution models."""
    
    def sample(self, months: int, scenarios: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample returns for given months and scenarios.
        
        Returns
        -------
        np.ndarray of shape (scenarios, months)
        """
        raise NotImplementedError


@dataclass(frozen=False)
class GaussianReturns(ReturnModel):
    """
    Gaussian (Normal) return model.
    
    Parameters
    ----------
    annual_return : float
        Expected annual return (e.g., 0.10 for 10%).
    annual_volatility : float  
        Annual volatility/standard deviation (e.g., 0.16 for 16%).
    """
    annual_return: float
    annual_volatility: float
    
    def __post_init__(self):
        check_non_negative("annual_volatility", self.annual_volatility)
    
    @property
    def monthly_return(self) -> float:
        """Convert annual return to monthly return (compounded)."""
        return annual_to_monthly(self.annual_return)
    
    @property 
    def monthly_volatility(self) -> float:
        """Convert annual volatility to monthly volatility."""
        return self.annual_volatility / np.sqrt(12)
    
    def sample(self, months: int, scenarios: int, rng: np.random.Generator) -> np.ndarray:
        """Sample Gaussian returns."""
        return rng.normal(
            loc=self.monthly_return,
            scale=self.monthly_volatility, 
            size=(scenarios, months)
        )


@dataclass(frozen=False)
class StudentTReturns(ReturnModel):
    """
    Student's t-distribution return model for heavy tails.
    
    Parameters
    ----------
    annual_return : float
        Expected annual return.
    annual_volatility : float
        Annual volatility.
    degrees_freedom : float
        Degrees of freedom for t-distribution. Must be > 2.
    """
    annual_return: float
    annual_volatility: float
    degrees_freedom: float
    
    def __post_init__(self):
        check_non_negative("annual_volatility", self.annual_volatility)
        if self.degrees_freedom <= 2:
            raise ValueError("degrees_freedom must be > 2 for finite variance")
    
    @property
    def monthly_return(self) -> float:
        return annual_to_monthly(self.annual_return)
    
    @property
    def monthly_volatility(self) -> float:
        # Scale volatility for t-distribution to match target
        scale_factor = np.sqrt(self.degrees_freedom / (self.degrees_freedom - 2))
        return (self.annual_volatility / np.sqrt(12)) / scale_factor
    
    def sample(self, months: int, scenarios: int, rng: np.random.Generator) -> np.ndarray:
        """Sample Student-t returns."""
        t_samples = rng.standard_t(df=self.degrees_freedom, size=(scenarios, months))
        return self.monthly_return + self.monthly_volatility * t_samples


# ---------------------------------------------------------------------------
# Account and Portfolio Classes
# ---------------------------------------------------------------------------

@dataclass(frozen=False)
class Account:
    """
    Single investment account with return characteristics.
    
    Parameters
    ----------
    name : str
        Account identifier (e.g., "emergency", "housing", "brokerage").
    return_model : ReturnModel
        Statistical model for sampling returns R_t^m.
    initial_wealth : float, default 0.0
        Starting wealth W_0^m in this account.
    """
    name: str
    return_model: ReturnModel
    initial_wealth: float = 0.0
    
    def __post_init__(self):
        check_non_negative("initial_wealth", self.initial_wealth)
    
    @classmethod
    def from_gaussian(
        cls,
        name: str,
        annual_return: float,
        annual_volatility: float,
        initial_wealth: float = 0.0
    ) -> Account:
        """Convenience constructor for Gaussian returns."""
        return cls(
            name=name,
            return_model=GaussianReturns(annual_return, annual_volatility),
            initial_wealth=initial_wealth
        )
    
    @classmethod
    def from_student_t(
        cls,
        name: str, 
        annual_return: float,
        annual_volatility: float,
        degrees_freedom: float,
        initial_wealth: float = 0.0
    ) -> Account:
        """Convenience constructor for Student-t returns."""
        return cls(
            name=name,
            return_model=StudentTReturns(annual_return, annual_volatility, degrees_freedom),
            initial_wealth=initial_wealth
        )


@dataclass(frozen=False)
class Portfolio:
    """
    Collection of accounts with correlation structure and allocation logic.
    
    This is the main interface for the optimization framework. It handles:
    - Sampling correlated returns R_t^m across accounts
    - Computing wealth evolution given allocation policies X and contributions A_t
    - Providing optimization-ready representations (affine form, gradients)
    
    Parameters
    ----------
    accounts : list[Account]
        List of Account instances representing different investment accounts.
    correlation : Optional[np.ndarray], default None
        Correlation matrix between account returns. Must be symmetric positive definite.
        If None, assumes uncorrelated accounts (identity matrix).
    name : str, default "portfolio"
        Identifier for the portfolio.
    """
    accounts: list[Account]
    correlation: Optional[np.ndarray] = None
    name: str = "portfolio"
    
    def __post_init__(self):
        if not self.accounts:
            raise ValueError("accounts list cannot be empty")
            
        n_accounts = len(self.accounts)
        
        # Initialize correlation matrix if not provided
        if self.correlation is None:
            self.correlation = np.eye(n_accounts)
        else:
            self.correlation = np.asarray(self.correlation)
            if self.correlation.shape != (n_accounts, n_accounts):
                raise ValueError(f"correlation matrix shape {self.correlation.shape} "
                               f"does not match number of accounts {n_accounts}")
            
            # Check if correlation matrix is valid
            if not np.allclose(self.correlation, self.correlation.T):
                raise ValueError("correlation matrix must be symmetric")
            
            eigenvals = np.linalg.eigvals(self.correlation)
            if np.any(eigenvals <= 0):
                raise ValueError("correlation matrix must be positive definite")
    
    @property
    def n_accounts(self) -> int:
        """Number of accounts in the portfolio."""
        return len(self.accounts)
    
    @property
    def account_names(self) -> list[str]:
        """List of account names."""
        return [acc.name for acc in self.accounts]
    
    @property
    def initial_wealth_vector(self) -> np.ndarray:
        """Initial wealth W_0^m across all accounts."""
        return np.array([acc.initial_wealth for acc in self.accounts])
    
    def returns(self, months: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample correlated returns for all accounts over a specified number of months.
        
        Parameters
        ----------
        months : int
            Number of months to simulate. Must be positive.
        seed : Optional[int], default None
            Random seed for reproducibility.
            
        Returns
        -------
        np.ndarray of shape (months, n_accounts)
            Sampled return matrix R_t^m for each month and account.
        
        Notes
        -----
        - Each account's return is sampled according to its `return_model`.
        - Correlations between accounts are applied via Cholesky decomposition.
        
        Example
        -------
        >>> portfolio.returns(months=12, seed=42)
        array([[0.01, 0.03],
            [0.02, 0.01],
            ...])
        """
        if not isinstance(months, int) or months <= 0:
            raise ValueError(f"'months' must be a positive integer, got {months}")
        
        rng = np.random.default_rng(seed)
        n_accounts = self.n_accounts
        
        # Step 1: Sample uncorrelated returns from each account model (vectorized)
        uncorrelated = np.column_stack([
            account.return_model.sample(months, 1, rng)[0] for account in self.accounts
        ])  # shape: (months, n_accounts)
        
        # Step 2: Apply correlation if needed
        if not np.allclose(self.correlation, np.eye(n_accounts)):
            L = cholesky(self.correlation.astype(np.float64), lower=True)
            uncorrelated = (L @ uncorrelated.T).T  # shape: (months, n_accounts)
    
        return uncorrelated

    def wealth(
        self,
        contributions: np.ndarray,
        allocation_policy: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """
        Compute wealth using affine representation for optimization.
        
        Implements the closed-form affine formula:
        W_t^m = W_0^m * F_{0,t}^m + Σ_{s=0}^{t-1} A_s * x_s^m * F_{s,t}^m
        
        where F_{s,t}^m = Π_{r=s}^{t-1} (1 + R_r^m) are accumulation factors.
        
        This representation is linear in the allocation policy X, making it
        suitable for convex optimization problems.
        
        Parameters
        ----------
        contributions : np.ndarray of shape (months,)
            Total monthly contributions A_t from income streams.
        allocation_policy : np.ndarray of shape (months, n_accounts)
            Allocation policy matrix X with x_t^m ≥ 0, Σ_m x_t^m = 1.
        returns : np.ndarray of shape (months, n_accounts)
            Monthly returns R_t^m for each month and account (single realization).
            
        Returns
        -------
        np.ndarray of shape (months+1, n_accounts)
            Wealth trajectory W_t^m using affine representation.
            
        Notes
        -----
        - This method is mathematically equivalent to wealth_recursive() but uses
          the closed-form affine representation instead of recursive evolution.
        - The affine form makes gradients with respect to allocation policy
          immediate: ∂W_t^m/∂x_s^m = A_s * F_{s,t}^m
        - Preferred method for optimization problems due to analytical tractability.
        """
        months, n_accounts = returns.shape
        
        # Validate inputs (same as wealth_recursive)
        contributions = np.asarray(contributions).flatten()
        if len(contributions) != months:
            raise ValueError(f"contributions length {len(contributions)} "
                           f"must match months {months}")
        
        allocation_policy = np.asarray(allocation_policy)
        if allocation_policy.shape != (months, n_accounts):
            raise ValueError(f"allocation_policy shape {allocation_policy.shape} "
                           f"must be ({months}, {n_accounts})")
        
        # Validate allocation policy constraints
        if np.any(allocation_policy < 0):
            raise ValueError("allocation_policy must be non-negative")
        
        row_sums = np.sum(allocation_policy, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-6):
            bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-6)[0][:5]
            raise ValueError(f"allocation_policy rows must sum to 1. "
                           f"Bad rows: {bad_rows.tolist()}")
        
        initial_wealth = self.initial_wealth_vector
        
        # Compute accumulation factors F_{s,t}^m
        accumulation_factors = self.compute_accumulation_factors(returns)
        
        # Initialize wealth array: (months+1, n_accounts)
        wealth = np.zeros((months + 1, n_accounts))
        
        # Set initial wealth W_0^m
        wealth[0, :] = initial_wealth
        
        # Apply affine formula for each time t and account m
        for t in range(1, months + 1):  # t = 1, 2, ..., months
            for m in range(n_accounts):
                # Initial wealth contribution: W_0^m * F_{0,t}^m
                wealth[t, m] = initial_wealth[m] * accumulation_factors[0, t, m]
                
                # Contribution terms: Σ_{s=0}^{t-1} A_s * x_s^m * F_{s,t}^m
                for s in range(t):
                    contribution_term = (
                        contributions[s] * 
                        allocation_policy[s, m] * 
                        accumulation_factors[s, t, m]
                    )
                    wealth[t, m] += contribution_term
        
        return wealth
    
    def compute_accumulation_factors(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute accumulation factors F_{s,t}^m for affine wealth representation.
        
        F_{s,t}^m = ∏_{r=s}^{t-1}(1+R_r^m) represents the growth factor
        from month s to month t for account m.
        
        Parameters
        ----------
        returns : np.ndarray of shape (months, n_accounts)
            Monthly returns R_t^m for a single realization.
            
        Returns
        -------
        np.ndarray of shape (months+1, months+1, n_accounts)
            Accumulation factors F_{s,t}^m where F[s,t,m] = ∏_{r=s}^{t-1}(1+R_r^m)
            from month s to month t, account m.
        """
        months, n_accounts = returns.shape
        
        # Initialize factors array
        factors = np.ones((months + 1, months + 1, n_accounts))
        
        # Compute accumulation factors
        for s in range(months + 1):
            for t in range(s, months + 1):
                if t > s:
                    # F_{s,t}^m = ∏_{r=s}^{t-1}(1+R_r^m)
                    factors[s, t, :] = np.prod(1 + returns[s:t, :], axis=0)
        
        return factors

    def wealth_statistics(
        self,
        wealth_paths: np.ndarray,
        percentiles: Optional[list[float]] = None
    ) -> dict:
        """
        Compute comprehensive statistics for wealth paths.
        
        Parameters
        ----------
        wealth_paths : np.ndarray of shape (scenarios, months+1, n_accounts)
            Wealth trajectories from Portoflio.wealth
        percentiles : Optional[list[float]], default None
            Percentiles to compute. If None, uses [5, 25, 50, 75, 95].
            
        Returns
        -------
        dict
            Dictionary containing statistics:
            - 'mean': mean wealth across scenarios
            - 'std': standard deviation across scenarios  
            - 'percentiles': percentile values
            - 'final_wealth': statistics for final period wealth
            - 'account_statistics': per-account statistics
        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        
        scenarios, months_plus_1, n_accounts = wealth_paths.shape
        
        # Overall statistics
        mean_wealth = np.mean(wealth_paths, axis=0)  # (months+1, n_accounts)
        std_wealth = np.std(wealth_paths, axis=0)    # (months+1, n_accounts)
        
        # Percentile statistics
        pct_wealth = np.percentile(wealth_paths, percentiles, axis=0)  # (len(percentiles), months+1, n_accounts)
        
        # Final wealth statistics (last time period)
        final_wealth = wealth_paths[:, -1, :]  # (scenarios, n_accounts)
        final_stats = {
            'mean': np.mean(final_wealth, axis=0),
            'std': np.std(final_wealth, axis=0),
            'percentiles': {p: np.percentile(final_wealth, p, axis=0) for p in percentiles}
        }
        
        # Total portfolio wealth (sum across accounts)
        total_wealth = np.sum(wealth_paths, axis=2)  # (scenarios, months+1)
        total_stats = {
            'mean': np.mean(total_wealth, axis=0),
            'std': np.std(total_wealth, axis=0),
            'percentiles': {p: np.percentile(total_wealth, p, axis=0) for p in percentiles}
        }
        
        # Per-account statistics
        account_stats = {}
        for i, account_name in enumerate(self.account_names):
            account_wealth = wealth_paths[:, :, i]  # (scenarios, months+1)
            account_stats[account_name] = {
                'mean': np.mean(account_wealth, axis=0),
                'std': np.std(account_wealth, axis=0),
                'final_mean': np.mean(account_wealth[:, -1]),
                'final_std': np.std(account_wealth[:, -1]),
                'percentiles': {p: np.percentile(account_wealth, p, axis=0) for p in percentiles}
            }
        
        return {
            'mean': mean_wealth,
            'std': std_wealth,
            'percentiles': {p: pct_wealth[i] for i, p in enumerate(percentiles)},
            'final_wealth': final_stats,
            'total_portfolio': total_stats,
            'account_statistics': account_stats,
            'scenarios': scenarios,
            'months': months_plus_1 - 1,
            'n_accounts': n_accounts
        }

    def wealth_recursive(
            self,
            contributions: np.ndarray,
            allocation_policy: np.ndarray,
            returns: np.ndarray,
        ) -> np.ndarray:
            """
            Compute wealth evolution given allocation policy and total contributions.
            
            Implements the core optimization equation:
            W_{t+1}^m = (W_t^m + A_t^m)(1 + R_t^m)
            where A_t^m = x_t^m * A_t
            
            Parameters
            ----------
            contributions : np.ndarray of shape (months,)
                Total monthly contributions A_t from income streams.
            allocation_policy : np.ndarray of shape (months, n_accounts)
                Allocation policy matrix X with x_t^m ≥ 0, Σ_m x_t^m = 1.
            returns : np.ndarray of shape (months, n_accounts) 
                Monthly returns R_t^m for each month and account (single realization).
                
            Returns
            -------
            np.ndarray of shape (months+1, n_accounts)
                Wealth trajectory W_t^m starting from t=0.
            """
            months, n_accounts = returns.shape
            
            # Validate inputs
            contributions = np.asarray(contributions).flatten()
            if len(contributions) != months:
                raise ValueError(f"contributions length {len(contributions)} "
                            f"must match months {months}")
            
            allocation_policy = np.asarray(allocation_policy)
            if allocation_policy.shape != (months, n_accounts):
                raise ValueError(f"allocation_policy shape {allocation_policy.shape} "
                            f"must be ({months}, {n_accounts})")
            
            # Validate allocation policy constraints
            if np.any(allocation_policy < 0):
                raise ValueError("allocation_policy must be non-negative")
            
            row_sums = np.sum(allocation_policy, axis=1)
            if not np.allclose(row_sums, 1.0, rtol=1e-6):
                bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-6)[0][:5]
                raise ValueError(f"allocation_policy rows must sum to 1. "
                            f"Bad rows: {bad_rows.tolist()}")
            
            initial_wealth = self.initial_wealth_vector

            # Compute account contributions: A_t^m = x_t^m * A_t
            account_contributions = allocation_policy * contributions.reshape(-1, 1)
            
            # Initialize wealth array: (months+1, n_accounts)
            wealth = np.zeros((months + 1, n_accounts))
            wealth[0, :] = initial_wealth  # Set initial wealth
            
            # Recursive evolution: W_{t+1}^m = (W_t^m + A_t^m) * (1 + R_t^m)
            for t in range(months):
                wealth[t+1, :] = (wealth[t, :] + account_contributions[t, :]) * (1 + returns[t, :])
            
            return wealth