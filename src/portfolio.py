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
    Supports W0 override for optimization scenarios with varying initial conditions.

Design principles
-----------------
- Separation of concerns: Portfolio executes dynamics, does NOT generate returns
- Vectorized computation: Processes full Monte Carlo batches (n_sims, T, M)
- Optimization-ready: Affine representation exposes gradients ∂W/∂X analytically
- Matching income.py pattern: Same batch processing structure
- Annual parameters by default: Use .from_annual() for user-friendly API
- Flexible initialization: W0_override enables optimization without dummy accounts

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
>>> # 6. Optimization scenario: override initial wealth
>>> W0_scenario = np.array([5_000_000, 2_000_000])
>>> result_opt = portfolio.simulate(A=A_sims, R=R_sims, X=X, W0_override=W0_scenario)
>>>
>>> # 7. Visualize
>>> portfolio.plot(result=result, X=X, save_path="portfolio_analysis.png")
"""

from __future__ import annotations
from typing import List, Literal, Optional, Dict
from dataclasses import dataclass

import numpy as np
from matplotlib.ticker import FuncFormatter

from .utils import check_non_negative, annual_to_monthly, monthly_to_annual, millions_formatter

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
        Short account identifier used for goal references (e.g., "RN", "CC", "SLV").
        This is the key used when specifying goals: TerminalGoal(account="RN", ...).
    initial_wealth : float
        Starting balance W_0^m (non-negative).
    return_strategy : dict
        Expected return and volatility in **monthly arithmetic** space:
        {"mu": float, "sigma": float}

        - mu: monthly expected return (e.g., 0.0033 for ~4% annual)
        - sigma: monthly volatility (e.g., 0.0144 for ~5% annual)

        For annual parameters, use Account.from_annual() instead (recommended).
        For monthly parameters, use Account.from_monthly() (advanced/explicit).
    display_name : str, optional
        Long descriptive name for plots and reports (e.g., "Risky Norris (Fintual)").
        If not provided, `name` is used for display. This allows using short
        acronyms for goal specification while showing descriptive names in visuals.

    Properties
    ----------
    label : str
        Display name for plots (returns display_name if set, otherwise name).
    monthly_params : Dict[str, float]
        Monthly return parameters {"mu": float, "sigma": float}.
    annual_params : Dict[str, float]
        Annualized return parameters {"return": float, "volatility": float}.

    Methods
    -------
    from_annual(name, annual_return, annual_volatility, initial_wealth=0.0, display_name=None)
        Create account from annual parameters (recommended API).
    from_monthly(name, monthly_mu, monthly_sigma, initial_wealth=0.0, display_name=None)
        Create account from monthly parameters (advanced/explicit API).

    Examples
    --------
    # With display_name for cleaner goal specification
    >>> risky = Account.from_annual("RN", annual_return=0.12,
    ...                             annual_volatility=0.15,
    ...                             display_name="Risky Norris (Fintual)")
    >>> print(risky.name)       # Short name for goals
    'RN'
    >>> print(risky.label)      # Long name for plots
    'Risky Norris (Fintual)'

    # Goal uses short name
    >>> goal = TerminalGoal(account="RN", threshold=5_000_000, confidence=0.8)

    # Without display_name (backward compatible)
    >>> emergency = Account.from_annual("Emergency", annual_return=0.04,
    ...                                 annual_volatility=0.05)
    >>> print(emergency.label)  # Falls back to name
    'Emergency'

    # Low-level: Direct construction (deserialization/internal use)
    >>> acc = Account("SLV", 10_000, {"mu": 0.005, "sigma": 0.03},
    ...               display_name="iShares Silver Trust")
    """
    name: str
    initial_wealth: float
    return_strategy: dict  # {"mu": monthly, "sigma": monthly}
    display_name: Optional[str] = None

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
        display_name: Optional[str] = None,
    ) -> "Account":
        """
        Create account from annual return parameters (recommended API).

        Converts annual parameters to monthly representation internally,
        following the same pattern as FixedIncome.annual_growth and
        VariableIncome.annual_growth in income.py.

        Parameters
        ----------
        name : str
            Short account identifier for goal references (e.g., "RN", "CC").
        annual_return : float
            Expected annual return (e.g., 0.09 for 9%/year).
        annual_volatility : float
            Annual volatility (e.g., 0.15 for 15%/year).
        initial_wealth : float, default 0.0
            Starting balance W_0^m (non-negative).
        display_name : str, optional
            Long descriptive name for plots (e.g., "Risky Norris (Fintual)").
            If not provided, `name` is used for display.

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
        >>> # With display_name for cleaner goal specification
        >>> risky = Account.from_annual("RN", annual_return=0.12,
        ...                             annual_volatility=0.15,
        ...                             display_name="Risky Norris (Fintual)")
        >>> risky.name, risky.label
        ('RN', 'Risky Norris (Fintual)')

        >>> # Without display_name (backward compatible)
        >>> emergency = Account.from_annual("Emergency", annual_return=0.04,
        ...                                 annual_volatility=0.05)
        >>> emergency.label
        'Emergency'
        """
        mu_monthly = annual_to_monthly(annual_return)
        sigma_monthly = annual_volatility / np.sqrt(12)

        return cls(
            name=name,
            initial_wealth=initial_wealth,
            return_strategy={"mu": mu_monthly, "sigma": sigma_monthly},
            display_name=display_name
        )
    
    @classmethod
    def from_monthly(
        cls,
        name: str,
        monthly_mu: float,
        monthly_sigma: float,
        initial_wealth: float = 0.0,
        display_name: Optional[str] = None,
    ) -> "Account":
        """
        Create account from monthly return parameters (advanced/explicit API).

        For most use cases, prefer from_annual() which uses more intuitive
        annualized inputs. Use this method when you have explicit monthly
        parameters from external sources or require precise control.

        Parameters
        ----------
        name : str
            Short account identifier for goal references.
        monthly_mu : float
            Monthly expected return (arithmetic).
        monthly_sigma : float
            Monthly volatility (arithmetic).
        initial_wealth : float, default 0.0
            Starting balance W_0^m (non-negative).
        display_name : str, optional
            Long descriptive name for plots. If not provided, `name` is used.

        Returns
        -------
        Account
            Account instance with monthly parameters in return_strategy.

        Examples
        --------
        >>> # Explicit monthly parameters
        >>> acc = Account.from_monthly("TAC", monthly_mu=0.0058,
        ...                            monthly_sigma=0.0347,
        ...                            display_name="Tactical Fund")
        >>> acc.name, acc.label
        ('TAC', 'Tactical Fund')
        """
        return cls(
            name=name,
            initial_wealth=initial_wealth,
            display_name=display_name,
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

    @property
    def label(self) -> str:
        """
        Display name for plots and reports.

        Returns display_name if set, otherwise falls back to name.
        Use this property when displaying account names in visualizations.

        Returns
        -------
        str
            Human-readable account name for display purposes.

        Examples
        --------
        >>> acc = Account.from_annual("RN", 0.12, 0.15,
        ...                           display_name="Risky Norris (Fintual)")
        >>> acc.label
        'Risky Norris (Fintual)'

        >>> acc2 = Account.from_annual("Emergency", 0.04, 0.05)
        >>> acc2.label
        'Emergency'
        """
        return self.display_name if self.display_name else self.name

    def __repr__(self) -> str:
        """
        String representation showing annualized parameters (user-friendly).

        Examples
        --------
        >>> acc = Account.from_annual("RN", 0.12, 0.15, 10_000,
        ...                           display_name="Risky Norris")
        >>> print(acc)
        Account('RN' [Risky Norris]: 12.0%/year, σ=15.0%, W₀=$10,000)
        """
        ap = self.annual_params
        name_part = f"'{self.name}'"
        if self.display_name:
            name_part += f" [{self.display_name}]"
        return (
            f"Account({name_part}: {ap['return']:.1%}/year, "
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
    simulate(A, R, X, method="recursive", W0_override=None) -> dict
        Execute wealth dynamics for given contributions, returns, and allocations.
        Supports batch processing of Monte Carlo samples and W0 override.
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
    - W0_override enables optimization scenarios without creating dummy accounts
    
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
    >>> # Execute with default W0 (from accounts)
    >>> result = portfolio.simulate(A=A, R=R, X=X)
    >>> result["wealth"].shape
    (500, 25, 2)
    >>> 
    >>> # Execute with overridden W0 (optimization scenario)
    >>> W0_custom = np.array([3_000_000, 1_500_000])
    >>> result_opt = portfolio.simulate(A=A, R=R, X=X, W0_override=W0_custom)
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
        """
        Initial wealth vector W0 for all accounts.
        
        Returns
        -------
        np.ndarray, shape (M,)
            W0[m] = initial_wealth of account m.
        
        Examples
        --------
        >>> portfolio = Portfolio(accounts)
        >>> W0 = portfolio.initial_wealth_vector
        >>> print(W0)  # [0., 0.]
        """
        return np.array([acc.initial_wealth for acc in self.accounts])
    
    def simulate(
        self,
        A: np.ndarray,  # Contributions: (T,) or (n_sims, T)
        R: np.ndarray,  # Returns: (n_sims, T, M)
        X: np.ndarray,  # Allocations: (T, M)
        D: Optional[np.ndarray] = None,  # Withdrawals: (T, M) or (n_sims, T, M)
        method: Literal["recursive", "affine"] = "recursive",
        W0_override: Optional[np.ndarray] = None
    ) -> dict:
        """
        Execute wealth dynamics W_{t+1}^m = (W_t^m + A_t^m - D_t^m)(1 + R_t^m).
        
        Supports batch processing of Monte Carlo samples with automatic broadcasting
        and optional initial wealth override for optimization scenarios.
        
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
        D : np.ndarray, optional
            Monthly withdrawals from accounts.
            - Shape (T, M): deterministic withdrawals (broadcast across simulations)
            - Shape (n_sims, T, M): stochastic withdrawals (per scenario)
            - None (default): no withdrawals (D_t^m = 0 for all t, m)
        method : {"recursive", "affine"}, default "recursive"
            Computation method:
            - "recursive": iterative W_{t+1} = (W_t + A_t)(1+R_t) [faster]
            - "affine": closed-form W_t = W_0 F_{0,t} + Σ_s A_s x_s F_{s,t}
        W0_override : np.ndarray, shape (M,), optional
            Override initial wealth vector. If None, uses self.initial_wealth_vector.
            Useful for optimization scenarios where W0 varies without creating
            temporary Account objects.
        
        Returns
        -------
        dict with keys:
            - "wealth": np.ndarray, shape (n_sims, T+1, M)
                Wealth trajectories including W_0.
            - "total_wealth": np.ndarray, shape (n_sims, T+1)
                Sum across accounts: Σ_m W_t^m.
        
        Raises
        ------
        ValueError
            If shapes are incompatible or allocation policy violates constraints.
            If W0_override has incorrect shape.
        
        Notes
        -----
        - Initial wealth: W_0^m from self.accounts[m].initial_wealth (default)
          or W0_override[m] if provided
        - Contributions split: A_t^m = A_t * X[t, m]
        - For deterministic A (shape (T,)), broadcast across simulations
        - Vectorized: no Python-level loops over simulations
        
        Algorithm Complexity
        --------------------
        - Recursive: O(n_sims * T * M) via vectorized operations
        - Affine: O(T² * M * n_sims) for factor computation + application
        
        Examples
        --------
        # Default: use accounts' initial wealth
        >>> A = np.full(24, 100_000.0)  # (24,)
        >>> R = returns.generate(T=24, n_sims=500, seed=42)  # (500, 24, 2)
        >>> X = np.tile([0.6, 0.4], (24, 1))  # (24, 2)
        >>> result = portfolio.simulate(A, R, X)
        >>> result["wealth"].shape
        (500, 25, 2)
        
        # Override initial wealth (optimization scenario)
        >>> W0_scenario = np.array([5_000_000, 2_000_000])
        >>> result_opt = portfolio.simulate(A, R, X, W0_override=W0_scenario)
        >>> np.allclose(result_opt["wealth"][:, 0, :], W0_scenario)
        True
        
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
        
        # Validate W0_override if provided
        if W0_override is not None:
            if W0_override.shape != (self.M,):
                raise ValueError(
                    f"W0_override shape {W0_override.shape} != expected ({self.M},)"
                )
            if np.any(W0_override < 0):
                raise ValueError("W0_override must be non-negative")
        
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
        
        # Handle D broadcasting and validation
        if D is not None:
            if D.ndim == 2:
                # Deterministic: (T, M) -> broadcast to (n_sims, T, M)
                if D.shape != (T, M):
                    raise ValueError(f"D shape {D.shape} != expected ({T}, {M})")
                D_expanded = np.tile(D[np.newaxis, :, :], (n_sims, 1, 1))
            elif D.ndim == 3:
                # Stochastic: (n_sims, T, M)
                if D.shape != (n_sims, T, M):
                    raise ValueError(f"D shape {D.shape} != expected ({n_sims}, {T}, {M})")
                D_expanded = D
            else:
                raise ValueError(f"D must be 2D or 3D, got shape {D.shape}")
            
            # Validate D is non-negative
            if np.any(D_expanded < 0):
                raise ValueError("D must be non-negative (withdrawals cannot be negative)")
        else:
            D_expanded = None

        # Dispatch to computation method
        if method == "recursive":
            W = self._simulate_recursive(A_expanded, R, X, D_expanded, W0_override)
        elif method == "affine":
            W = self._simulate_affine(A_expanded, R, X, D_expanded, W0_override)
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
        X: np.ndarray,  # (T, M)
        D: Optional[np.ndarray] = None,  # (n_sims, T, M) or None
        W0_override: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Recursive wealth dynamics: W_{t+1}^m = (W_t^m + A_t^m - D_t^m)(1 + R_t^m).
        
        Vectorized over simulations (no Python loops over n_sims).
        
        Parameters
        ----------
        W0_override : np.ndarray, shape (M,), optional
            Override initial wealth. If None, uses self.initial_wealth_vector.
        
        Returns
        -------
        W : np.ndarray, shape (n_sims, T+1, M)
        """
        n_sims, T, M = R.shape
        W0 = W0_override if W0_override is not None else self.initial_wealth_vector
        
        # Initialize wealth
        W = np.zeros((n_sims, T + 1, M))
        W[:, 0, :] = W0  # Broadcast W0 to all simulations
        
        # Vectorized recursive evolution
        for t in range(T):
            # A_t^m = A_t * X[t, m]
            # Broadcasting: (n_sims, 1) * (1, M) → (n_sims, M)
            A_allocated = A[:, t, None] * X[t, :]
            
            # Apply withdrawals if provided
            if D is not None:
                # W_{t+1} = (W_t + A_t - D_t) * (1 + R_t)
                W[:, t + 1, :] = (W[:, t, :] + A_allocated - D[:, t, :]) * (1 + R[:, t, :])
            else:
                # W_{t+1} = (W_t + A_t) * (1 + R_t)
                W[:, t + 1, :] = (W[:, t, :] + A_allocated) * (1 + R[:, t, :])
        
        return W
    
    def _simulate_affine(
        self,
        A: np.ndarray,  # (n_sims, T)
        R: np.ndarray,  # (n_sims, T, M)
        X: np.ndarray,  # (T, M)
        D: Optional[np.ndarray] = None,  # (n_sims, T, M) or None
        W0_override: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Affine wealth representation: W_t^m = W_0^m F_{0,t}^m + Σ_s (A_s x_s^m - D_s^m) F_{s,t}^m.
        
        Closed-form formula, useful for optimization (wealth is linear in X).
        
        Parameters
        ----------
        W0_override : np.ndarray, shape (M,), optional
            Override initial wealth. If None, uses self.initial_wealth_vector.
        
        Returns
        -------
        W : np.ndarray, shape (n_sims, T+1, M)
        """
        n_sims, T, M = R.shape
        W0 = W0_override if W0_override is not None else self.initial_wealth_vector

        # Compute accumulation factors: (n_sims, T+1, T+1, M)
        F = self.compute_accumulation_factors(R)

        # Initialize wealth
        W = np.zeros((n_sims, T + 1, M))
        W[:, 0, :] = W0

        # Precompute contribution-weighted allocations minus withdrawals
        # net_cashflow[i, s, m] = A[i, s] * X[s, m] - D[i, s, m]
        # Shape: (n_sims, T, M)
        contrib_weighted = A[:, :, None] * X[None, :, :]
        if D is not None:
            net_cashflow = contrib_weighted - D
        else:
            net_cashflow = contrib_weighted

        # Apply affine formula for each time t (vectorized inner sum)
        # W_t^m = W_0^m * F_{0,t}^m + Σ_{s=0}^{t-1} (A_s * x_s^m - D_s^m) * F_{s,t}^m
        for t in range(1, T + 1):
            # Initial wealth term: W_0^m F_{0,t}^m
            W[:, t, :] = W0 * F[:, 0, t, :]

            # Net cashflow term: vectorized sum over s ∈ [0, t)
            # F[:, :t, t, :] has shape (n_sims, t, M)
            # net_cashflow[:, :t, :] has shape (n_sims, t, M)
            # Element-wise multiply and sum over axis=1 (time dimension)
            W[:, t, :] += (net_cashflow[:, :t, :] * F[:, :t, t, :]).sum(axis=1)

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
           Sample Average Approximation uses: (1/N)Σ_i 1{W_t^i(X) ≥ b} ≥ 1-ε
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

        if T == 0:
            return F

        gross_returns = 1.0 + R  # (n_sims, T, M)

        # Optimized computation using cumulative products
        # Key insight: F[s, t] = C[t] / C[s] where C[t] = ∏_{r=0}^{t-1} (1+R_r)
        #
        # This reduces redundant product calculations:
        # - Old: O(T²) calls to np.prod, each computing products from scratch
        # - New: O(T) for cumprod + O(T²) divisions (much faster than products)

        # C[t] = cumulative product from time 0 to t-1
        # Shape: (n_sims, T, M) where C[:, t, :] = ∏_{r=0}^{t} gross_returns[:, r, :]
        cum_prod = np.cumprod(gross_returns, axis=1)  # (n_sims, T, M)

        # Fill F using the relationship F[s, t] = C[t-1] / C[s-1]
        for t in range(1, T + 1):
            # F[0, t] = C[t-1] = ∏_{r=0}^{t-1} (1+R_r)
            F[:, 0, t, :] = cum_prod[:, t - 1, :]

            # F[s, t] = C[t-1] / C[s-1] for s > 0
            for s in range(1, t):
                F[:, s, t, :] = cum_prod[:, t - 1, :] / cum_prod[:, s - 1, :]

        return F
    
    def plot(
        self,
        result: dict,
        X: np.ndarray,
        start: Optional[date] = None,
        figsize: tuple = (16, 10),
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        return_fig_ax: bool = False,
        show_trajectories: bool = True,
        trajectory_alpha: float = 0.05,
        colors: Optional[dict] = None,
        hist_bins: int = 30,
        hist_color: str = 'mediumseagreen',
        goals: Optional[list] = None,
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
        goals : list, optional
            List of Goal objects (TerminalGoal or IntermediateGoal) to visualize.
            Terminal goals are shown as horizontal lines at t=T.
            Intermediate goals are shown with markers at their specified month.
        
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
        
        if start is not None:
            from .utils import month_index
            time_axis = month_index(start, T_plus_1)
            xlabel = "Date"
        else:
            time_axis = np.arange(T_plus_1)
            xlabel = "Month"
        
        # Setup colors (lookup by name or label for flexibility)
        if colors is None:
            colors = {}
        default_colors = plt.cm.Dark2(np.linspace(0, 1, M))
        account_colors = []
        for i, acc in enumerate(self.accounts):
            if acc.name in colors:
                account_colors.append(colors[acc.name])
            elif acc.label in colors:
                account_colors.append(colors[acc.label])
            elif i in colors:
                account_colors.append(colors[i])
            else:
                account_colors.append(default_colors[i])
        
        # Setup figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        
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
                        label='_nolegend_'
                    )
        
        # Mean trajectories (thicker)
        for m in range(M):
            mean_wealth = W[:, :, m].mean(axis=0)
            ax_accounts.plot(
                time_axis,
                mean_wealth,
                color=account_colors[m],
                linewidth=2.5,
                label=self.accounts[m].label
            )
        
        ax_accounts.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax_accounts.set_xlabel(xlabel)
        ax_accounts.set_ylabel("Wealth (CLP)")
        ax_accounts.set_title(r'Wealth by Account $(W_t^m)$')
        ax_accounts.legend(loc='best', fontsize=8)
        ax_accounts.grid(True, alpha=0.3)

        # ========== Panel 1: Goal visualization ==========
        if goals is not None:
            # Map account names to indices for color matching
            account_name_to_idx = {acc.name: idx for idx, acc in enumerate(self.accounts)}
            
            for goal in goals:
                if goal.account not in account_name_to_idx:
                    continue  # Skip if account not in portfolio
                
                acc_idx = account_name_to_idx[goal.account]
                goal_color = account_colors[acc_idx]
                
                # Check goal type using class name
                goal_type = goal.__class__.__name__
                
                if goal_type == 'TerminalGoal':
                    # Horizontal line at threshold across entire horizon
                    ax_accounts.axhline(
                        goal.threshold,
                        color=goal_color,
                        linestyle='--',
                        linewidth=2,
                        alpha=0.8,
                        zorder=10
                    )
                    
                    # Annotation at the right edge
                    ax_accounts.text(
                        time_axis[-1], goal.threshold,
                        f" ${goal.threshold:,.0f}".replace(",", "."),
                        color=goal_color,
                        fontsize=7,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor=goal_color, alpha=0.9, linewidth=1.5)
                    )
                
                elif goal_type == 'IntermediateGoal':
                    # Marker at specific month
                    goal_time = time_axis[goal.month] if goal.month < len(time_axis) else time_axis[-1]
                    
                    # Horizontal line up to goal month
                    if start is not None:
                        ax_accounts.plot(
                            [time_axis[0], goal_time],
                            [goal.threshold, goal.threshold],
                            color=goal_color,
                            linestyle=':',
                            linewidth=2,
                            alpha=0.7,
                            zorder=10
                        )
                    else:
                        ax_accounts.axhline(
                            goal.threshold,
                            xmin=0,
                            xmax=goal.month / T,
                            color=goal_color,
                            linestyle=':',
                            linewidth=2,
                            alpha=0.7,
                            zorder=10
                        )
                    
                    # Marker at goal month
                    ax_accounts.scatter(
                        goal_time, goal.threshold,
                        color=goal_color,
                        s=150,
                        marker='D',
                        edgecolor='white',
                        linewidth=2,
                        zorder=15,
                        alpha=0.9
                    )
                    
                    # Annotation
                    ax_accounts.text(
                        goal_time, goal.threshold,
                        f" t={goal.month}\n ${goal.threshold:,.0f}".replace(",", "."),
                        color=goal_color,
                        fontsize=7,
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor=goal_color, alpha=0.9, linewidth=1.5)
                    )
        
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
        
        ax_total.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
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
            f"Final Wealth:\n" +
            f"Mean: ${final_mean:,.0f}".replace(",", ".") + "\n" +
            f"Median: ${final_median:,.0f}".replace(",", ".") + "\n" +
            f"Std: ${final_std:,.0f}".replace(",", "."),
            transform=ax_total.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Lateral histogram for final wealth distribution (only if n_sims > 1)
        if n_sims > 1:
            divider = make_axes_locatable(ax_total)
            ax_hist = divider.append_axes("right", size=0.8, pad=0.15)
            
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
            labels=[acc.label for acc in self.accounts],
            colors=account_colors,
            alpha=0.7
        )
        
        ax_composition.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax_composition.set_xlabel(xlabel)
        ax_composition.set_ylabel("Wealth (CLP)")
        ax_composition.set_title(r"Portfolio Composition $(\mathbb{E} [W_t^m])$")
        ax_composition.legend(loc='upper left', fontsize=8)
        ax_composition.grid(True, alpha=0.3)
        
        # ========== Panel 4: Allocation policy (stacked bar) ==========
        time_axis_policy = time_axis[:T]

        bar_width = np.diff(time_axis_policy).min() if start is not None else 0.9

        bottom = np.zeros(T)
        for m in range(M):
            ax_policy.bar(
                time_axis_policy,
                X[:, m],
                bottom=bottom,
                width=bar_width,
                color=account_colors[m],
                label=self.accounts[m].label,
                edgecolor='white',
                linewidth=0.3,
                alpha=0.85
            )
            bottom += X[:, m]

        ax_policy.set_xlabel(xlabel)
        ax_policy.set_ylabel("Allocation Fraction")
        ax_policy.set_title(r"Allocation Policy $(X = (x_t^m))$")
        ax_policy.set_ylim(0, 1)
        ax_policy.grid(True, alpha=0.3)
        ax_policy.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)

        if start is not None:
            margin = (time_axis_policy[-1] - time_axis_policy[0]) * 0.02
            ax_policy.set_xlim(time_axis_policy[0] - margin, time_axis_policy[-1] + margin)
        else:
            ax_policy.set_xlim(-0.5, T - 0.5)
        
        # ========== Date formatting ==========
        if start is not None:
            # Rotate labels and format as dates for time series plots
            for ax in [ax_accounts, ax_total, ax_composition, ax_policy]:
                ax.tick_params(axis='x', rotation=45)
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
                tick_interval = max(3, T // 8) if T > 24 else max(1, T // 12)
                ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=tick_interval))
                
                for label in ax.get_xticklabels():
                    label.set_ha('right')
                    label.set_fontsize(8)
        # Main title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Annotation with simulation info
        param_text = f"Simulations: {n_sims} | Horizon: {T} months | Accounts: {M}"
        fig.tight_layout(rect=[0, 0.01, 1, 0.96 if title else 1])
        fig.text(0.99, 0.01, param_text, ha='right', va='bottom', fontsize=8, alpha=0.85)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if return_fig_ax:
            return fig, {
                'accounts': ax_accounts,
                'total': ax_total,
                'composition': ax_composition,
                'policy': ax_policy
            }