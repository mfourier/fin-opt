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

    def returns_batch(
        self,
        months: int,
        n_scenarios: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample multiple correlated return scenarios efficiently.
        
        Generates multiple independent realizations of return paths for Monte Carlo
        simulation. More efficient than calling returns() in a loop when n_scenarios
        is large (>100) due to vectorized operations.
        
        Parameters
        ----------
        months : int
            Number of months to simulate. Must be positive.
        n_scenarios : int
            Number of independent Monte Carlo scenarios. Must be positive.
        seed : Optional[int], default None
            Random seed for reproducibility.
            
        Returns
        -------
        np.ndarray of shape (n_scenarios, months, n_accounts)
            Batch of sampled return matrices R_t^m for each scenario, month, and account.
            
        Notes
        -----
        - Each scenario is an independent realization from the same distribution.
        - Correlations between accounts are preserved within each scenario.
        - Approximately 3-5x faster than looping over returns() due to vectorization.
        
        Example
        -------
        >>> returns_batch = portfolio.returns_batch(months=12, n_scenarios=1000, seed=42)
        >>> returns_batch.shape
        (1000, 12, 3)
        """
        if not isinstance(months, int) or months <= 0:
            raise ValueError(f"months must be a positive integer, got {months}")
        if not isinstance(n_scenarios, int) or n_scenarios <= 0:
            raise ValueError(f"n_scenarios must be a positive integer, got {n_scenarios}")
        
        rng = np.random.default_rng(seed)
        n_accounts = self.n_accounts
        
        # Sample all scenarios at once: (n_scenarios, months, n_accounts)
        samples = np.zeros((n_scenarios, months, n_accounts))
        for i, account in enumerate(self.accounts):
            samples[:, :, i] = account.return_model.sample(months, n_scenarios, rng)
        
        # Apply correlation to each scenario
        if not np.allclose(self.correlation, np.eye(n_accounts)):
            L = cholesky(self.correlation.astype(np.float64), lower=True)
            for s in range(n_scenarios):
                samples[s] = (L @ samples[s].T).T
        
        return samples


    def wealth_monte_carlo(
        self,
        contributions: np.ndarray,
        allocation_policy: np.ndarray,
        n_scenarios: int,
        seed: Optional[int] = None,
        method: Literal["affine", "recursive"] = "affine"
    ) -> np.ndarray:
        """
        Monte Carlo simulation of wealth evolution under return uncertainty.
        
        Generates multiple wealth trajectories by sampling returns from their distributions
        and computing wealth paths for each realization. Essential for Sample Average
        Approximation (SAA) in stochastic optimization:
        
            E[W_t^m(X)] ≈ (1/N) Σ_{i=1}^N W_t^m(X; ω^(i))
        
        Parameters
        ----------
        contributions : np.ndarray of shape (months,)
            Total monthly contributions A_t from income streams.
        allocation_policy : np.ndarray of shape (months, n_accounts)
            Allocation policy matrix X with x_t^m ≥ 0, Σ_m x_t^m = 1.
        n_scenarios : int
            Number of Monte Carlo scenarios. Typical values: 500-5000.
            Larger values → better approximation but slower computation.
        seed : Optional[int], default None
            Random seed for reproducibility.
        method : Literal["affine", "recursive"], default "affine"
            Computation method:
            - "affine": closed-form using accumulation factors (preferred for optimization)
            - "recursive": iterative evolution (simpler, slightly slower)
            
        Returns
        -------
        np.ndarray of shape (n_scenarios, months+1, n_accounts)
            Wealth trajectories W_t^m for each scenario, time period, and account.
            
        Notes
        -----
        - Encapsulates the common pattern of looping over scenarios, sampling returns,
        and computing wealth paths.
        - For optimization problems, use this output with evaluate_goal_probability()
        or wealth_statistics() to validate constraints.
        - Memory usage: O(n_scenarios × months × n_accounts) floats (~30MB for 1000 scenarios,
        24 months, 3 accounts).
        
        Example
        -------
        >>> # Basic Monte Carlo
        >>> wealth_paths = portfolio.wealth_monte_carlo(
        ...     contributions=A,
        ...     allocation_policy=X,
        ...     n_scenarios=1000,
        ...     seed=42
        ... )
        >>> wealth_paths.shape
        (1000, 25, 3)
        
        >>> # Compute statistics
        >>> stats = portfolio.wealth_statistics(wealth_paths)
        >>> final_mean = stats['final_wealth']['mean']
        """
        if not isinstance(n_scenarios, int) or n_scenarios <= 0:
            raise ValueError(f"n_scenarios must be a positive integer, got {n_scenarios}")
        
        months = len(contributions)
        rng = np.random.default_rng(seed)
        
        # Preallocate output array
        wealth_paths = np.zeros((n_scenarios, months + 1, self.n_accounts))
        
        # Choose computation method
        wealth_func = self.wealth if method == "affine" else self.wealth_recursive
        
        # Generate scenarios
        for i in range(n_scenarios):
            R = self.returns(months, seed=rng.integers(0, 2**31))
            wealth_paths[i] = wealth_func(contributions, allocation_policy, R)
        
        return wealth_paths


    def evaluate_goal_probability(
        self,
        wealth_paths: np.ndarray,
        t: int,
        account: int,
        threshold: float
    ) -> float:
        """
        Evaluate probabilistic goal satisfaction from Monte Carlo samples.
        
        Computes the empirical probability that wealth in a specific account at time t
        exceeds a threshold, using Monte Carlo samples:
        
            P̂(W_t^m ≥ b) = (1/N) Σ_{i=1}^N 1{W_t^m(ω^(i)) ≥ b}
        
        Used to validate chance constraints in stochastic optimization:
            P(W_t^m(X) ≥ b_t^m) ≥ 1 - ε_t^m
        
        Parameters
        ----------
        wealth_paths : np.ndarray of shape (n_scenarios, months+1, n_accounts)
            Monte Carlo wealth trajectories from wealth_monte_carlo().
        t : int
            Target time period (0 ≤ t ≤ months). t=0 is initial wealth.
        account : int
            Target account index (0 ≤ account < n_accounts).
        threshold : float
            Wealth threshold b_t^m to evaluate against.
            
        Returns
        -------
        float
            Empirical probability in [0, 1]. Interpretation:
            - 0.95 → 95% of scenarios meet the goal
            - 0.50 → only 50% of scenarios meet the goal
            
        Raises
        ------
        ValueError
            If t or account indices are out of bounds.
            
        Notes
        -----
        - Standard error of estimate: SE ≈ sqrt(p(1-p)/N) where p is true probability
        - For 95% CI with width ±0.02, need N ≈ 2400 scenarios
        - Use this to check if allocation policy X satisfies goals before returning
        from optimization inner loop
        
        Example
        -------
        >>> # Check emergency fund goal: 2M at month 12 with 95% confidence
        >>> wealth_mc = portfolio.wealth_monte_carlo(A, X, n_scenarios=1000, seed=42)
        >>> prob = portfolio.evaluate_goal_probability(
        ...     wealth_paths=wealth_mc,
        ...     t=12,
        ...     account=0,  # emergency fund
        ...     threshold=2_000_000
        ... )
        >>> print(f"Goal satisfaction probability: {prob:.2%}")
        Goal satisfaction probability: 96.30%
        >>> 
        >>> # Validate constraint: require P(W >= b) >= 0.95
        >>> is_feasible = prob >= 0.95
        """
        n_scenarios, months_plus_1, n_accounts = wealth_paths.shape
        
        # Validate indices
        if not (0 <= t < months_plus_1):
            raise ValueError(f"t must be in [0, {months_plus_1-1}], got {t}")
        if not (0 <= account < n_accounts):
            raise ValueError(f"account must be in [0, {n_accounts-1}], got {account}")
        
        # Extract wealth at target time and account
        wealth_at_t = wealth_paths[:, t, account]
        
        # Compute empirical probability
        success = wealth_at_t >= threshold
        probability = np.mean(success)
        
        return float(probability)


    def wealth_gradient(
        self,
        contributions: np.ndarray,
        returns: np.ndarray,
        account: int
    ) -> np.ndarray:
        """
        Compute gradient of wealth with respect to allocation policy.
        
        For the affine wealth representation:
            W_t^m = W_0^m * F_{0,t}^m + Σ_{s=0}^{t-1} A_s * x_s^m * F_{s,t}^m
        
        The gradient with respect to allocation x_s^m is:
            ∂W_t^m/∂x_s^m = A_s * F_{s,t}^m    for s < t
        
        This provides analytical gradients for gradient-based optimization methods
        (scipy.optimize, gradient descent, Newton methods).
        
        Parameters
        ----------
        contributions : np.ndarray of shape (months,)
            Total monthly contributions A_t.
        returns : np.ndarray of shape (months, n_accounts)
            Return realization R_t^m for a single scenario.
        account : int
            Target account index m (0 ≤ account < n_accounts).
            
        Returns
        -------
        np.ndarray of shape (months+1, months)
            Gradient matrix where gradient[t, s] = ∂W_t^m/∂x_s^m
            - gradient[t, s] = A_s * F_{s,t}^m for s < t
            - gradient[t, s] = 0 for s ≥ t (causal structure)
            
        Notes
        -----
        - The gradient is **linear** in contributions A_s and **multiplicative** in
        accumulation factors F_{s,t}^m.
        - For optimization at fixed horizon T, use gradient[T, :] to get the gradient
        of final wealth W_T^m with respect to entire allocation sequence.
        - Useful for:
        * scipy.optimize.minimize with method='L-BFGS-B' or 'trust-constr'
        * Projected gradient descent on allocation simplex
        * Sensitivity analysis
        
        Example
        -------
        >>> # Gradient of final wealth for account 0
        >>> R = portfolio.returns(months=24, seed=42)
        >>> grad = portfolio.wealth_gradient(
        ...     contributions=A,
        ...     returns=R,
        ...     account=0
        ... )
        >>> # Gradient at final time T=24
        >>> grad_final = grad[24, :]  # shape (24,)
        >>> print(f"Most influential month: {np.argmax(grad_final)}")
        Most influential month: 0
        
        >>> # Use in scipy.optimize
        >>> def objective(X_flat):
        ...     X = X_flat.reshape(months, n_accounts)
        ...     W = portfolio.wealth(A, X, R)
        ...     return -W[-1, 0]  # maximize final wealth in account 0
        >>> 
        >>> def objective_gradient(X_flat):
        ...     grad_matrix = portfolio.wealth_gradient(A, R, account=0)
        ...     grad_final = grad_matrix[-1, :]  # ∂W_T^0/∂x_s^0
        ...     # For other accounts, gradient is zero (constraint: Σ_m x_t^m = 1)
        ...     grad_full = np.zeros((months, n_accounts))
        ...     grad_full[:, 0] = -grad_final
        ...     return grad_full.flatten()
        """
        months = len(contributions)
        
        # Validate account index
        if not (0 <= account < self.n_accounts):
            raise ValueError(f"account must be in [0, {self.n_accounts-1}], got {account}")
        
        # Compute accumulation factors F_{s,t}^m
        factors = self.compute_accumulation_factors(returns)
        
        # Initialize gradient matrix
        gradient = np.zeros((months + 1, months))
        
        # Compute gradient: ∂W_t^m/∂x_s^m = A_s * F_{s,t}^m for s < t
        for t in range(1, months + 1):
            for s in range(t):
                gradient[t, s] = contributions[s] * factors[s, t, account]
        
        return gradient

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