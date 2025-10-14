"""
Portfolio investment modeling module for FinOpt.

Purpose
-------
Defines portfolio structure and wealth dynamics executor. This module implements
the mathematical foundation for wealth evolution under allocation policies,
designed for integration with the optimization framework.

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
    Metadata container for investment account (name, initial wealth, return strategy).
    Return generation is delegated to returns.py (separation of concerns).
    Use Account.from_annual() for annual parameters (recommended).

- Portfolio:
    Executor for wealth dynamics. Receives pre-generated contributions A and returns R,
    applies allocation policy X, and computes wealth trajectories W.
    Supports both recursive and affine (closed-form) computation methods.

Design principles
-----------------
- Separation of concerns: Portfolio executes dynamics, does NOT generate returns
- Vectorized computation: Processes full Monte Carlo batches (n_sims, T, M)
- Optimization-ready: Affine representation exposes gradients ∂W/∂X analytically
- Matching income.py pattern: Same batch processing structure
- Annual parameters by default: Use .from_annual() for user-friendly API

Example
-------
>>> from datetime import date
>>> import numpy as np
>>> from finopt.src.portfolio import Account, Portfolio
>>> from finopt.src.returns import ReturnModel
>>> from finopt.src.income import IncomeModel, FixedIncome, VariableIncome
>>> 
>>> # 1. Define accounts with annual parameters (recommended)
>>> accounts = [
...     Account.from_annual("Emergency", annual_return=0.04, 
...                         annual_volatility=0.05, initial_wealth=0),
...     Account.from_annual("Housing", annual_return=0.07,
...                         annual_volatility=0.12, initial_wealth=0)
... ]
>>> 
>>> # 2. Create portfolio executor
>>> portfolio = Portfolio(accounts)
>>> 
>>> # 3. Generate stochastic inputs externally
>>> income = IncomeModel(
...     fixed=FixedIncome(base=1_400_000, annual_growth=0.03),
...     variable=VariableIncome(base=200_000, sigma=0.10)
... )
>>> returns = ReturnModel(accounts, default_correlation=np.eye(2))
>>> 
>>> T, n_sims = 24, 500
>>> A_sims = income.contributions(T, start=date(2025,1,1), n_sims=n_sims)  # (500, 24)
>>> R_sims = returns.generate(T, n_sims=n_sims, seed=42)  # (500, 24, 2)
>>> 
>>> # 4. Define allocation policy
>>> X = np.tile([0.6, 0.4], (T, 1))  # 60-40 split
>>> 
>>> # 5. Execute wealth dynamics
>>> result = portfolio.simulate(A=A_sims, R=R_sims, X=X)
>>> W = result["wealth"]  # (500, 25, 2)
>>> W_total = result["total_wealth"]  # (500, 25)
>>>
>>> # 6. Visualize
>>> portfolio.plot(result=result, X=X, save_path="portfolio_analysis.png")
"""

from __future__ import annotations
from typing import List, Literal, Optional
from dataclasses import dataclass

import numpy as np

from .utils import check_non_negative, annual_to_monthly, monthly_to_annual

__all__ = [
    "Account",
    "Portfolio",
]


# ---------------------------------------------------------------------------
# Account (Metadata Container)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Account:
    """
    Portfolio account metadata with dual temporal parameter access.
    
    Represents a single investment account with return characteristics.
    Return generation is delegated to returns.py (ReturnModel consumes this metadata).
    
    Internal storage uses monthly parameters (canonical form), but provides
    seamless access to both monthly and annual representations via properties.
    
    Parameters
    ----------
    name : str
        Account identifier (e.g., "Emergency", "Housing", "Brokerage").
    initial_wealth : float
        Starting balance W_0^m (non-negative).
    return_strategy : dict
        Expected return and volatility in **monthly arithmetic** space:
        {"mu": float, "sigma": float}
        
        - mu: monthly expected return (e.g., 0.0033 for ~4% annual)
        - sigma: monthly volatility (e.g., 0.0144 for ~5% annual)
        
        For annual parameters, use Account.from_annual() instead (recommended).
        For monthly parameters, use Account.from_monthly() (advanced/explicit).
    
    Properties
    ----------
    monthly_params : Dict[str, float]
        Monthly return parameters {"mu": float, "sigma": float}.
    annual_params : Dict[str, float]
        Annualized return parameters {"return": float, "volatility": float}.
    
    Methods
    -------
    from_annual(name, annual_return, annual_volatility, initial_wealth=0.0)
        Create account from annual parameters (recommended API).
    from_monthly(name, monthly_mu, monthly_sigma, initial_wealth=0.0)
        Create account from monthly parameters (advanced/explicit API).
        
    Examples
    --------
    # Recommended: Annual parameters (user-friendly)
    >>> emergency = Account.from_annual("Emergency", annual_return=0.04,
    ...                                 annual_volatility=0.05, initial_wealth=0)
    >>> print(emergency)
    Account('Emergency': 4.0%/year, σ=5.0%, W₀=$0)
    
    # Introspection: dual temporal views
    >>> emergency.annual_params
    {'return': 0.04, 'volatility': 0.05}
    >>> emergency.monthly_params
    {'mu': 0.0032737, 'sigma': 0.0144338}
    
    # Advanced: Monthly parameters (explicit control)
    >>> custom = Account.from_monthly("Custom", monthly_mu=0.0058, 
    ...                               monthly_sigma=0.0347, initial_wealth=50_000)
    >>> custom.annual_params
    {'return': 0.0719..., 'volatility': 0.1201...}
    
    # Low-level: Direct construction (deserialization/internal use)
    >>> acc = Account("Legacy", 10_000, {"mu": 0.005, "sigma": 0.03})
    """
    name: str
    initial_wealth: float
    return_strategy: dict  # {"mu": monthly, "sigma": monthly}

    def __post_init__(self):
        if self.initial_wealth < 0:
            raise ValueError(f"initial_wealth must be non-negative, got {self.initial_wealth}")
        if "mu" not in self.return_strategy or "sigma" not in self.return_strategy:
            raise ValueError("return_strategy must contain 'mu' and 'sigma' keys")
        if self.return_strategy["sigma"] < 0:
            raise ValueError(f"sigma must be non-negative, got {self.return_strategy['sigma']}")
    
    @classmethod
    def from_annual(
        cls,
        name: str,
        annual_return: float,
        annual_volatility: float,
        initial_wealth: float = 0.0,
    ) -> "Account":
        """
        Create account from annual return parameters (recommended API).
        
        Converts annual parameters to monthly representation internally,
        following the same pattern as FixedIncome.annual_growth and
        VariableIncome.annual_growth in income.py.
        
        Parameters
        ----------
        name : str
            Account identifier (e.g., "Emergency", "Housing").
        annual_return : float
            Expected annual return (e.g., 0.09 for 9%/year).
        annual_volatility : float
            Annual volatility (e.g., 0.15 for 15%/year).
        initial_wealth : float, default 0.0
            Starting balance W_0^m (non-negative).
        
        Returns
        -------
        Account
            Account instance with monthly parameters in return_strategy.
        
        Notes
        -----
        Conversion formulas:
        - mu_monthly = (1 + annual_return)^(1/12) - 1  [compounded]
        - sigma_monthly = annual_volatility / sqrt(12)  [time scaling]
        
        Examples
        --------
        >>> # Conservative emergency fund: 4% annual return, 5% vol
        >>> emergency = Account.from_annual("Emergency", annual_return=0.04,
        ...                                 annual_volatility=0.05)
        >>> emergency.annual_params
        {'return': 0.04, 'volatility': 0.05}
        
        >>> # Aggressive growth account: 12% annual return, 20% vol
        >>> growth = Account.from_annual("Growth", annual_return=0.12,
        ...                              annual_volatility=0.20,
        ...                              initial_wealth=100_000)
        
        >>> # Verify round-trip conversion
        >>> monthly_to_annual(emergency.return_strategy["mu"])
        0.04  # recovers annual_return
        >>> emergency.return_strategy["sigma"] * np.sqrt(12)
        0.05  # recovers annual_volatility
        """
        mu_monthly = annual_to_monthly(annual_return)
        sigma_monthly = annual_volatility / np.sqrt(12)
        
        return cls(
            name=name,
            initial_wealth=initial_wealth,
            return_strategy={"mu": mu_monthly, "sigma": sigma_monthly}
        )
    
    @classmethod
    def from_monthly(
        cls,
        name: str,
        monthly_mu: float,
        monthly_sigma: float,
        initial_wealth: float = 0.0,
    ) -> "Account":
        """
        Create account from monthly return parameters (advanced/explicit API).
        
        For most use cases, prefer from_annual() which uses more intuitive
        annualized inputs. Use this method when you have explicit monthly
        parameters from external sources or require precise control.
        
        Parameters
        ----------
        name : str
            Account identifier.
        monthly_mu : float
            Monthly expected return (arithmetic).
        monthly_sigma : float
            Monthly volatility (arithmetic).
        initial_wealth : float, default 0.0
            Starting balance W_0^m (non-negative).
        
        Returns
        -------
        Account
            Account instance with monthly parameters in return_strategy.
        
        Examples
        --------
        >>> # Explicit monthly parameters
        >>> acc = Account.from_monthly("Tactical", monthly_mu=0.0058,
        ...                            monthly_sigma=0.0347)
        >>> acc.annual_params
        {'return': 0.0719..., 'volatility': 0.1201...}
        
        >>> # Deserialization use case
        >>> saved_params = {"mu": 0.0065, "sigma": 0.04}
        >>> acc = Account.from_monthly("Restored", **saved_params)
        """
        return cls(
            name=name,
            initial_wealth=initial_wealth,
            return_strategy={"mu": monthly_mu, "sigma": monthly_sigma}
        )
    
    @property
    def monthly_params(self) -> Dict[str, float]:
        """
        Monthly return parameters (canonical storage format).
        
        Returns
        -------
        Dict[str, float]
            {"mu": float, "sigma": float} in monthly arithmetic space.
        
        Examples
        --------
        >>> acc = Account.from_annual("Test", 0.08, 0.12)
        >>> acc.monthly_params
        {'mu': 0.006434..., 'sigma': 0.034641...}
        """
        return dict(self.return_strategy)  # Copy to prevent mutation
    
    @property
    def annual_params(self) -> Dict[str, float]:
        """
        Annualized return parameters (user-friendly representation).
        
        Converts internal monthly parameters to annualized equivalents
        using standard financial formulas:
        - Return: geometric compounding (1+μ_m)^12 - 1
        - Volatility: time-scaling σ_m * sqrt(12)
        
        Returns
        -------
        Dict[str, float]
            {"return": float, "volatility": float} in annual space.
        
        Examples
        --------
        >>> acc = Account.from_annual("Test", annual_return=0.08,
        ...                           annual_volatility=0.12)
        >>> acc.annual_params
        {'return': 0.08, 'volatility': 0.12}  # Round-trip recovery
        
        >>> # Conversion from monthly
        >>> acc2 = Account.from_monthly("Test2", monthly_mu=0.005,
        ...                             monthly_sigma=0.03)
        >>> acc2.annual_params
        {'return': 0.0616..., 'volatility': 0.1039...}
        """
        mu_annual = monthly_to_annual(self.return_strategy["mu"])
        sigma_annual = self.return_strategy["sigma"] * np.sqrt(12)
        return {
            "return": mu_annual,
            "volatility": sigma_annual
        }
    
    def __repr__(self) -> str:
        """
        String representation showing annualized parameters (user-friendly).
        
        Examples
        --------
        >>> acc = Account.from_annual("Emergency", 0.04, 0.05, 10_000)
        >>> print(acc)
        Account('Emergency': 4.0%/year, σ=5.0%, W₀=$10,000)
        """
        ap = self.annual_params
        return (
            f"Account('{self.name}': {ap['return']:.1%}/year, "
            f"σ={ap['volatility']:.1%}, W₀=${self.initial_wealth:,.0f})"
        )

# ---------------------------------------------------------------------------
# Portfolio (Wealth Dynamics Executor)
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Multi-account portfolio executor with allocation policy support.
    
    Executes wealth dynamics W_{t+1}^m = (W_t^m + A_t^m)(1 + R_t^m) given:
    - Pre-generated contributions A (from income.py)
    - Pre-generated returns R (from returns.py)
    - Allocation policy X (from user or optimizer)
    
    Does NOT generate stochastic processes (delegated to income.py and returns.py).
    
    Parameters
    ----------
    accounts : List[Account]
        Portfolio accounts with metadata (no embedded return models).
    
    Methods
    -------
    simulate(A, R, X, method="recursive") -> dict
        Execute wealth dynamics for given contributions, returns, and allocations.
        Supports batch processing of Monte Carlo samples.
    compute_accumulation_factors(R) -> np.ndarray
        Compute F_{s,t}^m = ∏_{r=s}^{t-1} (1 + R_r^m) for affine wealth representation.
    plot(result, X, **kwargs)
        Visualize wealth trajectories, composition, and allocation policy.
    
    Notes
    -----
    - Portfolio does NOT import ReturnModel (loose coupling by design)
    - User must import income.py and returns.py separately
    - Portfolio only executes dynamics, never generates stochastic processes
    - Supports both recursive and affine (closed-form) wealth computation
    - Vectorized: processes full batches (n_sims, T, M) without Python loops
    
    Examples
    --------
    >>> # Complete workflow
    >>> accounts = [
    ...     Account.from_annual("Emergency", annual_return=0.04, annual_volatility=0.05),
    ...     Account.from_annual("Housing", annual_return=0.07, annual_volatility=0.12)
    ... ]
    >>> portfolio = Portfolio(accounts)
    >>> 
    >>> # Generate inputs
    >>> income = IncomeModel(...)
    >>> returns = ReturnModel(accounts, default_correlation=np.eye(2))
    >>> A = income.contributions(24, start=date(2025,1,1), n_sims=500)
    >>> R = returns.generate(T=24, n_sims=500, seed=42)
    >>> X = np.tile([0.6, 0.4], (24, 1))
    >>> 
    >>> # Execute
    >>> result = portfolio.simulate(A=A, R=R, X=X)
    >>> result["wealth"].shape
    (500, 25, 2)
    >>>
    >>> # Visualize
    >>> portfolio.plot(result, X)
    """
    
    def __init__(self, accounts: List[Account]):
        if not accounts:
            raise ValueError("Portfolio requires at least one account")
        self.accounts = accounts
        self.M = len(accounts)
    
    @property
    def account_names(self) -> List[str]:
        """List of account names."""
        return [acc.name for acc in self.accounts]
    
    @property
    def initial_wealth_vector(self) -> np.ndarray:
        """Initial wealth W_0^m across all accounts, shape (M,)."""
        return np.array([acc.initial_wealth for acc in self.accounts])
    
    def simulate(
        self,
        A: np.ndarray,  # Contributions: (T,) or (n_sims, T)
        R: np.ndarray,  # Returns: (n_sims, T, M)
        X: np.ndarray,  # Allocations: (T, M)
        method: Literal["recursive", "affine"] = "affine"
    ) -> dict:
        """
        Execute wealth dynamics W_{t+1}^m = (W_t^m + A_t^m)(1 + R_t^m).
        
        Supports batch processing of Monte Carlo samples with automatic broadcasting.
        
        Parameters
        ----------
        A : np.ndarray
            Total monthly contributions.
            - Shape (T,): single deterministic path (broadcast across simulations)
            - Shape (n_sims, T): multiple stochastic paths
        R : np.ndarray, shape (n_sims, T, M)
            Monthly returns for each simulation, period, and account.
        X : np.ndarray, shape (T, M)
            Allocation policy: X[t, m] = fraction of A_t to account m.
            Must satisfy: X[t, :].sum() = 1, X[t, m] ≥ 0.
        method : {"recursive", "affine"}, default "affine"
            Computation method:
            - "recursive": iterative W_{t+1} = (W_t + A_t)(1+R_t)
            - "affine": closed-form W_t = W_0 F_{0,t} + ∑_s A_s x_s F_{s,t}
        
        Returns
        -------
        dict with keys:
            - "wealth": np.ndarray, shape (n_sims, T+1, M)
                Wealth trajectories including W_0.
            - "total_wealth": np.ndarray, shape (n_sims, T+1)
                Sum across accounts: ∑_m W_t^m.
        
        Raises
        ------
        ValueError
            If shapes are incompatible or allocation policy violates constraints.
        
        Notes
        -----
        - Initial wealth W_0^m from self.accounts[m].initial_wealth
        - Contributions split: A_t^m = A_t * X[t, m]
        - For deterministic A (shape (T,)), broadcast across simulations
        - Vectorized: no Python-level loops over simulations
        
        Algorithm Complexity
        --------------------
        - Recursive: O(n_sims * T * M) via vectorized operations
        - Affine: O(T² * M * n_sims) for factor computation + application
        
        Examples
        --------
        # Deterministic contributions, stochastic returns
        >>> A = np.full(24, 100_000.0)  # (24,)
        >>> R = returns.generate(T=24, n_sims=500, seed=42)  # (500, 24, 2)
        >>> X = np.tile([0.6, 0.4], (24, 1))  # (24, 2)
        >>> result = portfolio.simulate(A, R, X)
        >>> result["wealth"].shape
        (500, 25, 2)
        
        # Stochastic contributions and returns
        >>> A = income.contributions(24, start=date(2025,1,1), n_sims=500)  # (500, 24)
        >>> result = portfolio.simulate(A, R, X)
        
        # Single simulation edge case
        >>> R_single = returns.generate(T=24, n_sims=1, seed=42)  # (1, 24, 2)
        >>> result = portfolio.simulate(A, R_single, X)
        >>> result["wealth"].shape
        (1, 25, 2)
        """
        n_sims, T, M = R.shape
        
        # Validate dimensions
        if M != self.M:
            raise ValueError(f"R has {M} accounts, expected {self.M}")
        if X.shape != (T, M):
            raise ValueError(f"X shape {X.shape} != expected ({T}, {M})")
        
        # Validate allocation policy constraints
        if np.any(X < 0):
            raise ValueError("X must be non-negative")
        row_sums = X.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-6):
            bad_rows = np.where(np.abs(row_sums - 1.0) > 1e-6)[0][:5]
            raise ValueError(
                f"X rows must sum to 1. Bad rows: {bad_rows.tolist()}"
            )
        
        # Handle A broadcasting
        if A.ndim == 1:
            if A.shape[0] != T:
                raise ValueError(f"A shape {A.shape} != expected ({T},)")
            A_expanded = np.tile(A, (n_sims, 1))  # (n_sims, T)
        elif A.ndim == 2:
            if A.shape != (n_sims, T):
                raise ValueError(f"A shape {A.shape} != expected ({n_sims}, {T})")
            A_expanded = A
        else:
            raise ValueError(f"A must be 1D or 2D, got shape {A.shape}")
        
        # Dispatch to computation method
        if method == "recursive":
            W = self._simulate_recursive(A_expanded, R, X)
        elif method == "affine":
            W = self._simulate_affine(A_expanded, R, X)
        else:
            raise ValueError(f"method must be 'recursive' or 'affine', got {method}")
        
        # Total wealth across accounts
        total_wealth = W.sum(axis=2)  # (n_sims, T+1)
        
        return {
            "wealth": W,
            "total_wealth": total_wealth
        }
    
    def _simulate_recursive(
        self,
        A: np.ndarray,  # (n_sims, T)
        R: np.ndarray,  # (n_sims, T, M)
        X: np.ndarray   # (T, M)
    ) -> np.ndarray:
        """
        Recursive wealth dynamics: W_{t+1}^m = (W_t^m + A_t^m)(1 + R_t^m).
        
        Vectorized over simulations (no Python loops over n_sims).
        
        Returns
        -------
        W : np.ndarray, shape (n_sims, T+1, M)
        """
        n_sims, T, M = R.shape
        W0 = self.initial_wealth_vector  # (M,)
        
        # Initialize wealth
        W = np.zeros((n_sims, T + 1, M))
        W[:, 0, :] = W0  # Broadcast W0 to all simulations
        
        # Vectorized recursive evolution
        for t in range(T):
            # A_t^m = A_t * X[t, m]
            # Broadcasting: (n_sims, 1) * (1, M) → (n_sims, M)
            A_allocated = A[:, t, None] * X[t, :]
            
            # W_{t+1} = (W_t + A_t) * (1 + R_t)
            # All operations vectorized over n_sims
            W[:, t + 1, :] = (W[:, t, :] + A_allocated) * (1 + R[:, t, :])
        
        return W
    
    def _simulate_affine(
        self,
        A: np.ndarray,  # (n_sims, T)
        R: np.ndarray,  # (n_sims, T, M)
        X: np.ndarray   # (T, M)
    ) -> np.ndarray:
        """
        Affine wealth representation: W_t^m = W_0^m F_{0,t}^m + ∑_s A_s x_s^m F_{s,t}^m.
        
        Closed-form formula, useful for optimization (wealth is linear in X).
        
        Returns
        -------
        W : np.ndarray, shape (n_sims, T+1, M)
        """
        n_sims, T, M = R.shape
        W0 = self.initial_wealth_vector  # (M,)
        
        # Compute accumulation factors: (n_sims, T+1, T+1, M)
        F = self.compute_accumulation_factors(R)
        
        # Initialize wealth
        W = np.zeros((n_sims, T + 1, M))
        W[:, 0, :] = W0
        
        # Apply affine formula for each time t
        for t in range(1, T + 1):
            # Initial wealth term: W_0^m F_{0,t}^m
            # Broadcasting: (M,) * (n_sims, M) → (n_sims, M)
            W[:, t, :] = W0 * F[:, 0, t, :]
            
            # Contribution term: ∑_{s=0}^{t-1} A_s x_s^m F_{s,t}^m
            for s in range(t):
                # Allocate: A_s * X[s,:]: (n_sims, 1) * (1, M) → (n_sims, M)
                contrib = A[:, s, None] * X[s, :]
                # contrib * F_{s,t}^m: (n_sims, M) * (n_sims, M) → (n_sims, M)
                W[:, t, :] += contrib * F[:, s, t, :]
        
        return W
    
    def compute_accumulation_factors(self, R: np.ndarray) -> np.ndarray:
        """
        Compute accumulation factors F_{s,t}^m = ∏_{r=s}^{t-1} (1 + R_r^m).
        
        Vectorized computation over all simulations simultaneously.
        
        Parameters
        ----------
        R : np.ndarray, shape (n_sims, T, M)
            Return matrix.
        
        Returns
        -------
        F : np.ndarray, shape (n_sims, T+1, T+1, M)
            Accumulation factors where F[i, s, t, m] = ∏_{r=s}^{t-1} (1 + R[i,r,m])
            By convention: F[:, s, s, :] = 1 (no accumulation over empty interval)
        
        Notes
        -----
        Primary use in optimization framework:
        
        1. **Affine wealth gradient**: ∂W_t^m / ∂X[s,m] = A_s F_{s,t}^m
           Enables analytical computation of sensitivities for convex solvers.
           
        2. **Chance constraint reformulation**: For goals P(W_t^m ≥ b) ≥ 1-ε,
           Sample Average Approximation uses: (1/N)∑_i 1{W_t^i(X) ≥ b} ≥ 1-ε
           where each W_t^i is computed via affine formula using F.
           
        3. **CVaR optimization**: For risk-adjusted objectives like
           max E[W_T] - λ·CVaR_α(losses), gradients flow through F factors.
        
        Complexity & Memory
        -------------------
        Time: O(n_sims * T² * M)
        Memory: O(n_sims * T² * M) floats (8 bytes each)
        
        Memory estimates:
        - n_sims=500, T=24, M=2: ~115 MB
        - n_sims=500, T=120, M=5: ~14 GB
        - n_sims=1000, T=240, M=10: ~221 GB (infeasible on most systems)
        
        Warnings
        --------
        For T > 100, memory usage can exceed RAM. Consider:
        - Using recursive method instead of affine (no F precomputation)
        - Processing simulations in batches (chunking n_sims)
        - Computing gradients on-the-fly (store only needed F_{s,t} pairs)
        - Using sparse storage if only specific time points are constrained
        
        Examples
        --------
        >>> R = returns.generate(T=24, n_sims=500, seed=42)
        >>> F = portfolio.compute_accumulation_factors(R)
        >>> F.shape
        (500, 25, 25, 2)
        
        # Sample mean of cumulative factors (close to (1+μ)^T for low vol)
        >>> F[:, 0, 24, :].mean(axis=0)
        array([1.073, 1.184])
        
        # Note: With nonzero sigma, sample mean > (1+μ)^T due to Jensen's inequality
        # E[exp(X)] > exp(E[X]) for lognormal returns
        >>> # Expected: (1.003)^24 ≈ 1.074, (1.007)^24 ≈ 1.184
        >>> # Observed mean slightly higher due to convexity
        
        # Use in optimization: gradient computation
        >>> # For constraint W_24^0 >= b, gradient w.r.t. X[10, 0]:
        >>> # Assuming deterministic A (shape (24,)):
        >>> A_val = 100_000.0  # contribution at month 10
        >>> grad_X_10_0 = A_val * F[:, 10, 24, 0].mean()
        """
        n_sims, T, M = R.shape
        F = np.ones((n_sims, T + 1, T + 1, M))
        
        gross_returns = 1.0 + R  # (n_sims, T, M)
        
        # Vectorized computation over simulations
        # F[i, s, t, m] = ∏_{r=s}^{t-1} gross_returns[i, r, m]
        for s in range(T):
            for t in range(s + 1, T + 1):
                # Product over axis=1 (time dimension) for slice [s:t]
                # gross_returns[:, s:t, :]: (n_sims, t-s, M)
                # prod(axis=1): (n_sims, M)
                F[:, s, t, :] = np.prod(gross_returns[:, s:t, :], axis=1)
        
        return F
    
    def plot(
        self,
        result: dict,
        X: np.ndarray,
        start: Optional[date] = None,
        figsize: tuple = (16, 8),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        show_trajectories: bool = True,
        trajectory_alpha: float = 0.05,
        colors: Optional[dict] = None,
        hist_bins: int = 30,
        hist_color: str = 'mediumseagreen',
    ):
        """
        Visualize portfolio wealth dynamics with 4 panels + lateral histogram.
        
        Panel layout:
        - Top-left: Wealth per account (time series with trajectories)
        - Top-right: Total portfolio wealth (time series with trajectories + lateral histogram)
        - Bottom-left: Portfolio composition over time (stacked area)
        - Bottom-right: Allocation policy heatmap (X matrix)
        
        Parameters
        ----------
        result : dict
            Output from simulate() containing:
            - "wealth": np.ndarray, shape (n_sims, T+1, M)
            - "total_wealth": np.ndarray, shape (n_sims, T+1)
        X : np.ndarray, shape (T, M)
            Allocation policy used in the simulation.
        start : Optional[date], default None
            Start date for temporal axis. If None, uses numeric month indices (0, 1, 2, ...).
            If provided, x-axis shows calendar dates (first-of-month).
            Aligns with income.py temporal representation convention.
        figsize : tuple, default (16, 10)
            Figure size (width, height).
        title : str, optional
            Main title for the figure.
        save_path : str, optional
            Path to save figure.
        return_fig_ax : bool, default False
            If True, returns (fig, axes_dict).
        show_trajectories : bool, default True
            Whether to show individual simulation paths.
        trajectory_alpha : float, default 0.05
            Transparency for trajectory lines.
        colors : dict, optional
            Custom colors for accounts. Keys are account names or indices.
        hist_bins : int, default 30
            Number of bins for final wealth distribution histogram.
        hist_color : str, default 'mediumseagreen'
            Color for the lateral histogram.
        
        Returns
        -------
        None or (fig, axes_dict)
            If return_fig_ax=True, returns figure and dict of axes.
        
        Notes
        -----
        The lateral histogram on the total wealth panel shows the distribution
        of final wealth W_T across all Monte Carlo simulations, providing
        immediate visual feedback on outcome uncertainty.
        
        When start is provided, months are converted to calendar dates for
        improved readability. This matches the temporal representation used
        in income.plot_income() and income.plot_contributions().
        
        Examples
        --------
        >>> from datetime import date
        >>> result = portfolio.simulate(A, R, X)
        >>> 
        >>> # Numeric time axis (default)
        >>> portfolio.plot(result, X, title="Portfolio Analysis")
        >>> 
        >>> # Calendar time axis
        >>> portfolio.plot(result, X, start=date(2025, 1, 1),
        ...                title="Portfolio Analysis")
        >>> 
        >>> # Custom colors and histogram
        >>> portfolio.plot(result, X, start=date(2025, 1, 1),
        ...                colors={"Emergency": "green", "Housing": "blue"},
        ...                hist_bins=40, hist_color='coral')
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        
        W = result["wealth"]  # (n_sims, T+1, M)
        W_total = result["total_wealth"]  # (n_sims, T+1)
        n_sims, T_plus_1, M = W.shape
        T = T_plus_1 - 1
        
        if X.shape != (T, M):
            raise ValueError(f"X shape {X.shape} != expected ({T}, {M})")
        
        # ========== Construcción del eje temporal ==========
        if start is not None:
            from .utils import month_index
            time_axis = month_index(start, T_plus_1)
            xlabel = "Date"
        else:
            time_axis = np.arange(T_plus_1)
            xlabel = "Month"
        
        # Setup colors
        if colors is None:
            colors = {}
        default_colors = plt.cm.tab10(np.linspace(0, 1, M))
        account_colors = []
        for i, acc in enumerate(self.accounts):
            if acc.name in colors:
                account_colors.append(colors[acc.name])
            elif i in colors:
                account_colors.append(colors[i])
            else:
                account_colors.append(default_colors[i])
        
        # Setup figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax_accounts = fig.add_subplot(gs[0, 0])
        ax_total = fig.add_subplot(gs[0, 1])
        ax_composition = fig.add_subplot(gs[1, 0])
        ax_policy = fig.add_subplot(gs[1, 1])
        
        # ========== Panel 1: Wealth per account ==========
        if show_trajectories:
            for m in range(M):
                for i in range(n_sims):
                    ax_accounts.plot(
                        time_axis,
                        W[i, :, m],
                        color=account_colors[m],
                        alpha=trajectory_alpha,
                        linewidth=0.8,
                        label=self.accounts[m].name if i == 0 else '_nolegend_'
                    )
        
        # Mean trajectories (thicker)
        for m in range(M):
            mean_wealth = W[:, :, m].mean(axis=0)
            ax_accounts.plot(
                time_axis,
                mean_wealth,
                color=account_colors[m],
                linewidth=2.5,
                label=f"{self.accounts[m].name} (mean)"
            )
        
        ax_accounts.set_xlabel(xlabel)
        ax_accounts.set_ylabel("Wealth (CLP)")
        ax_accounts.set_title(r'Wealth by Account $(W_t^m)$')
        ax_accounts.legend(loc='best', fontsize=8)
        ax_accounts.grid(True, alpha=0.3)
        
        # ========== Panel 2: Total wealth ==========
        if show_trajectories:
            for i in range(n_sims):
                ax_total.plot(
                    time_axis,
                    W_total[i, :],
                    color='gray',
                    alpha=trajectory_alpha,
                    linewidth=0.8,
                    label='Trajectories' if i == 0 else '_nolegend_'
                )
        
        mean_total = W_total.mean(axis=0)
        ax_total.plot(
            time_axis,
            mean_total,
            color='black',
            linewidth=2.5,
            label='Mean'
        )
        
        ax_total.set_xlabel(xlabel)
        ax_total.set_ylabel("Total Wealth (CLP)")
        ax_total.set_title(r'Total Portfolio Wealth $(\sum_{m} W_t^m)$')
        ax_total.legend(loc='lower right')
        ax_total.grid(True, alpha=0.3)
        
        # Annotation with final wealth statistics
        final_mean = mean_total[-1]
        final_std = W_total[:, -1].std()
        final_median = np.median(W_total[:, -1])
        ax_total.text(
            0.02, 0.98,
            f"Final Wealth:\nMean: ${final_mean:,.0f}\nMedian: ${final_median:,.0f}\nStd: ${final_std:,.0f}".replace(",", "."),
            transform=ax_total.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Lateral histogram for final wealth distribution (only if n_sims > 1)
        if n_sims > 1:
            divider = make_axes_locatable(ax_total)
            ax_hist = divider.append_axes("right", size=1.0, pad=0.15)
            
            final_wealth = W_total[:, -1]  # (n_sims,)
            ax_hist.hist(
                final_wealth,
                bins=hist_bins,
                orientation='horizontal',
                color=hist_color,
                alpha=0.6,
                edgecolor='black',
                linewidth=0.5
            )
            
            ax_hist.set_xlabel("Count", fontsize=9)
            ax_hist.set_ylim(ax_total.get_ylim())
            ax_hist.tick_params(axis='both', labelsize=8)
            ax_hist.grid(True, alpha=0.2)
            ax_hist.set_title("Final\nDistribution", fontsize=9)
        
        # ========== Panel 3: Portfolio composition (stacked area) ==========
        mean_wealth_by_account = W.mean(axis=0)  # (T+1, M)
        
        ax_composition.stackplot(
            time_axis,
            *[mean_wealth_by_account[:, m] for m in range(M)],
            labels=[acc.name for acc in self.accounts],
            colors=account_colors,
            alpha=0.7
        )
        
        ax_composition.set_xlabel(xlabel)
        ax_composition.set_ylabel("Wealth (CLP)")
        ax_composition.set_title("Portfolio Composition")
        ax_composition.legend(loc='upper left', fontsize=8)
        ax_composition.grid(True, alpha=0.3)
        
        # ========== Panel 4: Allocation policy heatmap ==========
        im = ax_policy.imshow(
            X.T,
            aspect='auto',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )
        
        ax_policy.set_xlabel("Month")
        ax_policy.set_ylabel("Account")
        ax_policy.set_title(r'Allocation Policy $(X = (x_t^m)_{t,m})$')
        ax_policy.set_yticks(range(M))
        ax_policy.set_yticklabels([acc.name for acc in self.accounts], fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax_policy, fraction=0.046, pad=0.04, label='Fraction')
        
        # Add allocation values as text (if not too many)
        if T <= 24 and M <= 5:
            for t in range(T):
                for m in range(M):
                    text_color = 'white' if X[t, m] > 0.5 else 'black'
                    ax_policy.text(
                        t, m, f"{X[t, m]:.2f}",
                        ha='center', va='center',
                        color=text_color, fontsize=7
                    )
        
        # ========== Date formatting (if applicable) ==========
        if start is not None:
            # Rotate labels and format as dates for time series plots
            for ax in [ax_accounts, ax_total, ax_composition]:
                ax.tick_params(axis='x', rotation=45)
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=max(1, T//12)))
            
            # For policy heatmap, format ticks manually (imshow uses numeric indices)
            # Select subset of months to avoid label saturation
            n_ticks = min(12, T)  # Maximum 12 ticks on x-axis
            tick_positions = np.linspace(0, T-1, n_ticks, dtype=int)
            
            # Generate date labels for those tick positions
            tick_labels = [time_axis[pos].strftime('%Y-%m') for pos in tick_positions]
            
            ax_policy.set_xticks(tick_positions)
            ax_policy.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
        
        # Main title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Annotation with simulation info
        param_text = f"n_sims={n_sims} | T={T} | accounts={M}"
        fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=8, alpha=0.7)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if return_fig_ax:
            return fig, {
                'accounts': ax_accounts,
                'total': ax_total,
                'composition': ax_composition,
                'policy': ax_policy
            }