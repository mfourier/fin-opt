"""
Stochastic optimization module for FinOpt.

Purpose
-------
Goal-driven allocation policy synthesis under uncertainty with support for
intermediate and terminal goals. Integrates convex programming with Monte Carlo
simulation for solving chance-constrained portfolio optimization problems.

Mathematical Framework
----------------------
Bilevel optimization problem:

    min T ∈ ℕ  s.t.  ∃X* ∈ arg max f(X)
                          X ∈ 𝒳_T
                          ℙ(W_t^m(X) ≥ b_t^m) ≥ 1-ε_t^m  ∀ intermediate goals
                          ℙ(W_T^m(X) ≥ b^m) ≥ 1-ε^m      ∀ terminal goals

Outer problem: Binary/linear search over horizon T (GoalSeeker)
Inner problem: Convex program via affine wealth representation (CVaROptimizer)

Key Components
--------------
- OptimizationResult: Container for X*, T*, objective value, goal_set, and diagnostics
- AllocationOptimizer: Abstract base with parametrizable objectives f(X)
- CVaROptimizer: Convex LP via CVaR reformulation (supports multiple convex objectives)
- GoalSeeker: Bilevel solver with binary/linear search and warm start

Design Principles
-----------------
- Separation of concerns: Goals defined in goals.py, solvers here
- Scenario-driven: Receives pre-generated (A, R) from FinancialModel
- Convex optimization: Exploits affine wealth W_t^m(X) for global optimality
- Portfolio-aware: Consumes Account objects for type safety and metadata access
- Reproducible: Explicit seed management for stochastic scenarios

Convex Objectives in CVaROptimizer
-----------------------------------
All objectives exploit the affine structure: W[:,t,m] = b + Φ @ X[:t,m]

Available objectives:
    - "risky": E[Σ_m W_T^m]
    - "balanced": E[W_T] - λ·Σ_{t,m}|x_{t+1,m} - x_t^m| (default)
    - "conservative": E[W_T] - λ·Std(W_T)
    - "risky_turnover": E[W_T] - λ·Σ_{t,m}|x_{t+1,m} - x_t^m|
    - Custom callable: f(W, X, T, M) → float

All objectives maintain convexity via:
- Affine wealth: W is linear in X
- Variance: Var(W) = E[W²] - E[W]² is quadratic convex
- L1 norm: ||ΔX||₁ is convex
- Max-min: max_X min_i f_i(X) is convex when f_i are concave in X

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Union, Literal, TYPE_CHECKING
from abc import ABC, abstractmethod
from datetime import date
import time
import numpy as np

from .goals import IntermediateGoal, TerminalGoal, GoalSet, check_goals
from .portfolio import Portfolio
if TYPE_CHECKING:
    from .model import SimulationResult
    from .portfolio import Account

__all__ = [
    "OptimizationResult",
    "AllocationOptimizer",
    "CVaROptimizer",
    "GoalSeeker",
]


# Type alias for objective specifications
ObjectiveType = Union[
    Literal["risky", "risky_turnover", "balanced", "conservative"],
    Callable[[np.ndarray, np.ndarray, int, int], float]
]


# ---------------------------------------------------------------------------
# Optimization Result Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationResult:
    """
    Container for allocation optimization output.
    
    Immutable dataclass holding optimal policy X*, horizon T*, objective value,
    feasibility status, goal_set reference, and solver diagnostics.
    
    Attributes
    ----------
    X : np.ndarray, shape (T, M)
        Optimal allocation policy matrix.
        X[t, m] = fraction of contribution A_t allocated to account m at month t.
        Satisfies: Σ_m X[t, m] = 1, X[t, m] ≥ 0 for all t, m.
    T : int
        Optimization horizon (number of months).
    objective_value : float
        Final objective f(X*) at optimum.
    feasible : bool
        Whether all goals satisfied at X*.
        True → all chance constraints hold within tolerance.
        False → at least one goal violated (infeasible solution).
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        Original goal specifications from problem formulation.
    goal_set : GoalSet
        Validated goal collection with resolved accounts and metadata.
        Provides access to accounts, start_date, and resolved indices.
    solve_time : float
        Total solver execution time (seconds).
    n_iterations : int, optional
        Number of solver iterations (solver-dependent).
    diagnostics : dict, optional
        Solver-specific metadata (duality_gap, convergence_status, etc.)
    
    Examples
    --------
    >>> from finopt.src.portfolio import Account
    >>> accounts = [Account.from_annual("Emergency", 0.04, 0.05)]
    >>> result = optimizer.solve(T=24, A=A, R=R, W0=W0, goals=goals, 
    ...                          accounts=accounts, start_date=date(2025,1,1))
    >>> print(result.summary())
    OptimizationResult(
      Status: ✓ Feasible
      Horizon: T=24 months
      Objective: 11234567.89
      Goals: 3 (1 intermediate, 2 terminal)
      Solve time: 0.342s
      Iterations: 18
    )
    >>> 
    >>> # Access accounts via goal_set
    >>> result.goal_set.accounts
    [Account('Emergency': 4.0%/year, ...)]
    """
    X: np.ndarray
    T: int
    objective_value: float
    feasible: bool
    goals: List[Union[IntermediateGoal, TerminalGoal]]
    goal_set: GoalSet
    solve_time: float
    n_iterations: Optional[int] = None
    diagnostics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result structure at construction."""
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {self.X.shape}")
        T_X, M = self.X.shape
        if T_X != self.T:
            raise ValueError(f"X.shape[0]={T_X} != T={self.T}")
        
        if not isinstance(self.T, int) or self.T < 1:
            raise ValueError(f"T must be positive int, got {self.T}")
        if not isinstance(self.feasible, bool):
            raise TypeError(f"feasible must be bool, got {type(self.feasible)}")
        if not isinstance(self.solve_time, (int, float)) or self.solve_time < 0:
            raise ValueError(f"solve_time must be non-negative, got {self.solve_time}")
        
        # Validate goal_set consistency
        if not isinstance(self.goal_set, GoalSet):
            raise TypeError(f"goal_set must be GoalSet, got {type(self.goal_set)}")
        if self.goal_set.M != M:
            raise ValueError(
                f"goal_set.M={self.goal_set.M} != X.shape[1]={M}"
            )
    
    def summary(self) -> str:
        """Human-readable optimization summary."""
        status = "✓ Feasible" if self.feasible else "✗ Infeasible"
        
        n_intermediate = sum(1 for g in self.goals if isinstance(g, IntermediateGoal))
        n_terminal = sum(1 for g in self.goals if isinstance(g, TerminalGoal))
        
        lines = [
            "OptimizationResult(",
            f"  Status: {status}",
            f"  Horizon: T={self.T} months",
            f"  Objective: {self.objective_value:.2f}",
            f"  Goals: {len(self.goals)} ({n_intermediate} intermediate, "
            f"{n_terminal} terminal)",
            f"  Solve time: {self.solve_time:.3f}s",
            f"  Iterations: {self.n_iterations if self.n_iterations is not None else 'N/A'}",
        ]
        
        if self.diagnostics:
            if 'duality_gap' in self.diagnostics:
                lines.append(f"  Duality gap: {self.diagnostics['duality_gap']:.2e}")
            if 'convergence_status' in self.diagnostics:
                lines.append(f"  Convergence: {self.diagnostics['convergence_status']}")
        
        lines.append(")")
        return "\n".join(lines)
    
    def validate_goals(
        self,
        result: SimulationResult
    ) -> Dict[Union[IntermediateGoal, TerminalGoal], Dict[str, float]]:
        """
        Validate goal satisfaction in simulation result using X*.
        
        Parameters
        ----------
        result : SimulationResult
            Simulation output to validate against goals
        
        Returns
        -------
        dict
            Goal satisfaction metrics (from check_goals)
        
        Notes
        -----
        Uses self.goal_set.accounts and self.goal_set.start_date internally,
        eliminating the need to pass these parameters explicitly.
        """
        if result.T != self.T:
            raise ValueError(
                f"Result horizon mismatch: result.T={result.T} != self.T={self.T}"
            )
        
        return check_goals(
            result, 
            self.goals, 
            self.goal_set.accounts,
            self.goal_set.start_date
        )
    
    def is_valid_allocation(self, tol: float = 1e-6) -> bool:
        """Check if X satisfies allocation constraints."""
        if np.any(self.X < -tol):
            return False
        
        row_sums = self.X.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=tol):
            return False
        
        return True
    
    @property
    def M(self) -> int:
        """Number of accounts (portfolio size)."""
        return self.X.shape[1]


# ---------------------------------------------------------------------------
# Abstract Optimizer Interface
# ---------------------------------------------------------------------------

class AllocationOptimizer(ABC):
    """
    Abstract base class for allocation policy optimizers.
    
    Defines interface for inner problem solvers in bilevel optimization.
    Subclasses implement specific formulations (CVaR, SAA, etc.) by
    providing concrete solve() method.
    
    Supports parametrizable objectives f(X):
    - "risky": E[Σ_m W_T^m]
    - "balanced": E[W_T] - λ·Σ_{t,m}|x_{t+1,m} - x_t^m| (default)
    - "conservative": E[W_T] - λ·Std(W_T)
    - "risky_turnover": E[W_T] - λ·Σ_{t,m}|x_{t+1,m} - x_t^m|
    - Custom callable: f(W, X, T, M) → float
    
    Parameters
    ----------
    n_accounts : int
        Number of portfolio accounts M
    objective : ObjectiveType, default "terminal_wealth"
        Objective function specification
    objective_params : dict, optional
        Parameters for objective function (e.g., {"lambda": 0.1})
    account_names : List[str], optional
        Account name labels for goal resolution
    
    Notes
    -----
    Subclasses must implement:
    - solve(T, A, R, W0, goal_set, X_init, **kwargs) → OptimizationResult
    
    Provided utilities:
    - _validate_inputs(...) → GoalSet
    - _check_feasibility(...) → bool
    - _compute_objective(W, X, T, M) → float (dispatches to objective functions)
    """
    
    def __init__(
        self,
        n_accounts: int,
        objective: ObjectiveType = "balanced",
        objective_params: Optional[Dict[str, Any]] = None,
        account_names: Optional[List[str]] = None
    ):
        if n_accounts < 1:
            raise ValueError(f"n_accounts must be ≥ 1, got {n_accounts}")
        
        self.M = n_accounts
        self.objective = objective
        self.objective_params = objective_params or {}
        self.account_names = account_names or [f"Account_{i}" for i in range(n_accounts)]
        
        if len(self.account_names) != n_accounts:
            raise ValueError(
                f"account_names length {len(self.account_names)} != "
                f"n_accounts {n_accounts}"
            )
    
    @abstractmethod
    def solve(
        self,
        T: int,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        goal_set: GoalSet,
        X_init: Optional[np.ndarray] = None,
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Solve allocation optimization problem for fixed horizon T.
        
        Caller (typically GoalSeeker) is responsible for creating GoalSet once.
        
        Parameters
        ----------
        T : int
            Optimization horizon (months)
        A : np.ndarray, shape (n_sims, T)
            Contribution scenarios
        R : np.ndarray, shape (n_sims, T, M)
            Return scenarios
        W0 : np.ndarray, shape (M,)
            Initial wealth vector
        goal_set : GoalSet
            Pre-validated goal collection with resolved accounts and metadata.
            MUST be created by caller (not None).
        X_init : np.ndarray, optional
            Warm start allocation (T, M)
        **solver_kwargs
            Solver-specific parameters
        
        Returns
        -------
        OptimizationResult
            Optimal X*, objective value, feasibility, goal_set, diagnostics
        """
        pass
    
    def _check_feasibility(
        self,
        X: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        portfolio: Portfolio,
        goal_set: GoalSet
    ) -> bool:
        """
        Check if allocation X satisfies all goals using exact SAA.
        
        Uses non-smoothed indicator function for final validation.
        
        Parameters
        ----------
        X : np.ndarray, shape (T, M)
            Allocation policy to validate
        A : np.ndarray, shape (n_sims, T)
            Contribution scenarios
        R : np.ndarray, shape (n_sims, T, M)
            Return scenarios
        W0 : np.ndarray, shape (M,)
        portfolio : Portfolio
            Portfolio instance 
        goal_set : GoalSet
            Validated goal collection
        
        Returns
        -------
        bool
            True if all goals satisfied, False otherwise
        """
        
        X_normalized = X.copy()
        row_sums = X_normalized.sum(axis=1, keepdims=True)
        
        max_deviation = np.abs(row_sums - 1.0).max()
        if max_deviation > 0.01:
            return False
        
        X_normalized = X_normalized / row_sums
        
        result = portfolio.simulate(
            A=A, R=R, X=X_normalized, 
            method="affine", 
            W0_override=W0
        )
        W = result["wealth"]  # (n_sims, T+1, M)
        
        # Check intermediate goals
        for goal in goal_set.intermediate_goals:
            m = goal_set.get_account_index(goal)
            t = goal_set.get_resolved_month(goal)
            
            W_t_m = W[:, t, m]
            violations = W_t_m < goal.threshold
            violation_rate = violations.mean()
            
            if violation_rate > goal.epsilon:
                return False
        
        # Check terminal goals
        T = X.shape[0]
        for goal in goal_set.terminal_goals:
            m = goal_set.get_account_index(goal)
            
            W_T_m = W[:, T, m]
            violations = W_T_m < goal.threshold
            violation_rate = violations.mean()
            
            if violation_rate > goal.epsilon:
                return False
        
        return True
    
    def _compute_objective(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """
        Compute objective function value (to be MAXIMIZED).
        
        Parameters
        ----------
        W : np.ndarray, shape (n_sims, T+1, M)
            Wealth trajectories
        X : np.ndarray, shape (T, M)
            Allocation policy
        T : int
            Horizon
        M : int
            Number of accounts
        
        Returns
        -------
        float
            Objective value (higher is better)
        """
        if callable(self.objective):
            return self.objective(W, X, T, M)
        
        # Dispatch to predefined objectives
        if self.objective == "risky":
            return self._objective_risky(W, X, T, M)
        elif self.objective == "balanced":
            return self._objective_balanced(W, X, T, M)
        elif self.objective == "risky_turnover":
            return self._objective_risky_turnover(W, X, T, M)
        elif self.objective == "conservative":
            return self._objective_conservative(W, X, T, M)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def _objective_risky(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """Expected total terminal wealth: E[Σ_m W_T^m]."""
        return W[:, T, :].sum(axis=1).mean()
    
    def _objective_balanced(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """
        Turnover penalty.
        
        f(X) = - Σ_{t,m} (x_{t+1,m} - x_t^m)^2
        """
        if T > 1:
            turnover = ((X[1:, :] - X[:-1, :]) ** 2).sum()
        else:
            turnover = 0.0
        
        return -turnover

    def _objective_risky_turnover(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """
        Terminal wealth with turnover penalty.
        
        f(X) = E[W_T] - λ·Σ_{t,m} (x_{t+1,m} - x_t^m)^2
        """
        lambda_ = self.objective_params.get("lambda", 0.5)
        
        W_T_total = W[:, T, :].sum(axis=1)
        mean_wealth = W_T_total.mean()

        if T > 1:
            turnover = ((X[1:, :] - X[:-1, :]) ** 2).sum()
        else:
            turnover = 0.0
        return mean_wealth - lambda_ * turnover
    
    def _objective_conservative(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """
        Mean-variance objective.
        
        f(X) = E[W_T] - λ·Std(W_T)
        """
        lambda_ = self.objective_params.get("lambda", 0.5)
        
        W_T_total = W[:, T, :].sum(axis=1)
        mean_wealth = W_T_total.mean()
        std_wealth = W_T_total.std()
        
        return mean_wealth - lambda_ * std_wealth


# ---------------------------------------------------------------------------
# CVaR Optimizer (Convex Programming)
# ---------------------------------------------------------------------------

class CVaROptimizer(AllocationOptimizer):
    """
    CVaR-based portfolio allocation optimizer using convex programming.
    
    Reformulates chance constraints P(W >= threshold) >= 1-ε as CVaR constraints:
        CVaR_ε(threshold - W) <= 0
    
    Uses the epigraphic formulation of Rockafellar & Uryasev (2000):
        CVaR_α(L) = min_{γ,z} { γ + (1/αN) Σ_i z^i }
        subject to: z^i >= L^i - γ, z^i >= 0, ∀i
    
    where L^i = threshold - W^i is the shortfall in scenario i.
    
    Mathematical Formulation
    ------------------------
    Decision variables:
        X[t,m] : allocation fraction to account m at time t
                 satisfying Σ_m X[t,m] = 1 (simplex) and X[t,m] >= 0
    
    Wealth dynamics (affine in X):
        W[:,t,m] = W0[m] * F[:,0,t,m] + Σ_{s<t} A[:,s] * X[s,m] * F[:,s,t,m]
        
        where:
        - F[i,s,t,m] = ∏_{τ=s+1}^t (1 + R[i,τ,m]) is the accumulation factor
        - A[i,s] is the contribution at time s in scenario i
        - W0[m] is the initial wealth in account m
    
    CVaR constraints (per goal):
        γ_g + (1/(ε_g * N)) * Σ_i z^i_g <= 0
        z^i_g >= (threshold_g - W[i,T,m_g]) - γ_g
        z^i_g >= 0
    
    Supported Convex Objectives
    ----------------------------
    All objectives exploit affine wealth structure for convexity:
    
    1. risky: max E[Σ_m W_T^m]
       - Linear program (fastest)
       - Maximizes expected total final wealth
       - Use case: Wealth accumulation without risk considerations
    
    2. balanced: -||ΔX||_2
       - Reduces transaction costs via L2 penalty on allocation changes
    
    3. conservative: max E[W_T] - λ·Var(W_T)
       - Mean-variance tradeoff (uses variance, not std, for convexity)
       - λ controls risk aversion (typical: 0.1-1.0)
       - Use case: Markowitz-style optimization
    
    Parameters
    ----------
    n_accounts : int
        Number of investment accounts in the portfolio
    objective : str, default='balanced'
        Optimization objective. Options:
        'risky', 'balanced', 'conservative'.
    objective_params : dict, optional
        Objective-specific parameters:
        - risky:
        - balanced: {'lambda': 0.5}
        - conservative: {'lambda': 0.3}
    account_names : list of str, optional
        Names of accounts for improved diagnostics
    
    Attributes
    ----------
    cp : module
        CVXPY module (imported at initialization)
    
    Notes
    -----
    Complexity:
        Variables: T*M + G*(1 + N) + (objective-dependent)
        Constraints: T + G*(N + 1) + (objective-dependent)
        Solve time: O(n^{3.5}) where n = number of variables
        
        For typical parameters (T=24, M=3, G=3, N=300):
        - Variables: ~900
        - Constraints: ~900
        - Solve time: 30-80ms depending on objective (ECOS solver)
    
    The formulation is a pure convex program (LP or SOCP), solved efficiently
    using interior-point methods (ECOS, SCS, or CLARABEL). All objectives
    maintain convexity and global optimality guarantees.
    """
    
    def __init__(
        self,
        n_accounts: int,
        objective: str = 'balanced',
        objective_params: Optional[Dict[str, Any]] = None,
        account_names: Optional[List[str]] = None
    ):
        super().__init__(n_accounts, objective, objective_params, account_names)
        
        # Validate objective
        valid_objectives = [
            'risky', 'balanced', 'risky_turnover', 'conservative'
        ]
        if objective not in valid_objectives:
            raise ValueError(
                f"Invalid objective '{objective}'. "
                f"Valid options: {', '.join(valid_objectives)}"
            )
        
        # Validate and import CVXPY
        try:
            import cvxpy as cp
            self.cp = cp
        except ImportError:
            raise ImportError(
                "CVaROptimizer requires cvxpy. "
                "Install with: pip install cvxpy"
            )
    
    def solve(
        self,
        T: int,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        goal_set: GoalSet,
        X_init: Optional[np.ndarray] = None,
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Solve the portfolio allocation problem using convex optimization.
        
        Parameters
        ----------
        T : int
            Optimization horizon (number of months)
        A : ndarray, shape (n_sims, T)
            Monthly contributions per scenario
            A[i,t] = contribution at month t in scenario i
        R : ndarray, shape (n_sims, T, M)
            Monthly returns per scenario and account
            R[i,t,m] = return of account m at month t in scenario i
        W0 : ndarray, shape (M,)
            Initial wealth per account
        goal_set : GoalSet
            Pre-validated goal collection (REQUIRED).
            Must be created by caller before calling solve().
        X_init : ndarray, optional
            Initial guess for allocations (ignored by CVXPY)
        **solver_kwargs
            Additional solver options:
            - solver : {'ECOS', 'SCS', 'CLARABEL'}, default='ECOS'
            - verbose : bool, default=False
            - max_iters : int, default=10000
            - abstol : float, default=1e-7 (absolute tolerance)
            - reltol : float, default=1e-6 (relative tolerance)
        
        Returns
        -------
        OptimizationResult
            Result object containing:
            - X : optimal allocation policy (T, M)
            - T : horizon length
            - objective_value : optimal objective value
            - feasible : whether all goals are satisfied
            - solve_time : solver time in seconds
            - diagnostics : dict with solver information
        
        Raises
        ------
        ValueError
            If intermediate goal exceeds horizon T
        RuntimeError
            If solver succeeds but returns None for X.value
        
        Notes
        -----
        The solver automatically handles numerical tolerances. If the simplex
        constraint is violated within numerical precision (< 1e-6), the solution
        is projected back to the simplex via normalization.
        
        Solver selection:
        - ECOS: Fast, recommended for LP/SOCP (default)
        - SCS: More robust, handles ill-conditioned problems
        - CLARABEL: Modern, balanced performance
        """
        import time
        cp = self.cp
        
        start_time = time.time()
        
        # Validation of inputs
        if not isinstance(goal_set, GoalSet):
            raise TypeError(
                f"goal_set must be GoalSet instance, got {type(goal_set)}"
            )
        
        if T < goal_set.T_min:
            raise ValueError(
                f"T={T} < goal_set.T_min={goal_set.T_min} "
                f"(required by intermediate goals)"
            )
        
        n_sims = A.shape[0]
        M = self.M
        
        if A.shape != (n_sims, T):
            raise ValueError(f"A must have shape ({n_sims}, {T}), got {A.shape}")
        if R.shape != (n_sims, T, M):
            raise ValueError(f"R must have shape ({n_sims}, {T}, {M}), got {R.shape}")
        if W0.shape != (M,):
            raise ValueError(f"W0 must have shape ({M},), got {W0.shape}")
        
        if goal_set.M != M:
            raise ValueError(
                f"goal_set.M={goal_set.M} != n_accounts={M}"
            )
        
        # Extract solver options
        solver_name = solver_kwargs.get('solver', 'ECOS')
        verbose = solver_kwargs.get('verbose', False)
        max_iters = solver_kwargs.get('max_iters', 10000)
        abstol = solver_kwargs.get('abstol', 1e-7)
        reltol = solver_kwargs.get('reltol', 1e-6)
        
        # Precompute accumulation factors: (n_sims, T+1, T+1, M)
        portfolio = Portfolio(goal_set.accounts)
        F = portfolio.compute_accumulation_factors(R)
        
        # ============= DECISION VARIABLES =============
        
        # X[t,m]: allocation fraction to account m at time t
        # Constraints: X[t,m] >= 0 and Σ_m X[t,m] = 1 (defined below)
        X = cp.Variable((T, M), nonneg=True, name="allocations")
        
        # CVaR auxiliary variables (one set per goal)
        gamma = {}  # γ: VaR level (scalar per goal)
        z = {}      # z^i: excess over VaR in scenario i (vector per goal)
        
        # ============= HELPER FUNCTION: AFFINE WEALTH =============
        
        def build_wealth_affine(t: int, m: int):
            """
            Build affine expression for wealth: W[:,t,m] = b + Φ @ X[:t,m]
            
            Mathematical formulation:
                W[i,t,m] = W0[m] * F[i,0,t,m] + Σ_{s=0}^{t-1} A[i,s] * X[s,m] * F[i,s,t,m]
            
            Matrix form:
                W[:,t,m] = b_{t,m} + Φ_{t,m} @ X[:t,m]
                
                where:
                - b[i] = W0[m] * F[i,0,t,m]  (constant term)
                - Φ[i,s] = A[i,s] * F[i,s,t,m]  (coefficient matrix)
            
            Parameters
            ----------
            t : int
                Time index (1 <= t <= T)
            m : int
                Account index (0 <= m < M)
            
            Returns
            -------
            W_t_m : cp.Expression, shape (n_sims,)
                CVXPY expression representing wealth in account m at time t
                across all scenarios (affine function of X)
            """
            # Constant term: initial wealth compounded to time t
            b = W0[m] * F[:, 0, t, m]  # shape: (n_sims,)
            
            if t == 0:
                return b  # No contributions before t=0
            
            # Coefficient matrix: Φ[i,s] = A[i,s] * F[i,s,t,m]
            # Element-wise multiplication creates the affine coefficients
            Phi = A[:, :t] * F[:, :t, t, m]  # shape: (n_sims, t)
            
            # Wealth = constant + matrix @ decision_vars
            # W[:,t,m] = b + Φ @ X[:t,m]
            return b + Phi @ X[:t, m]
        
        # ============= CONSTRAINTS =============
        
        constraints = []
        
        # 1. Simplex constraint: Σ_m X[t,m] = 1 for all t
        #    Ensures allocations sum to 100% at each time period
        constraints.append(cp.sum(X, axis=1) == 1)
        
        # 2. CVaR constraints for terminal goals
        for goal in goal_set.terminal_goals:
            m = goal_set.get_account_index(goal)
            
            # Compute terminal wealth W[:,T,m] as affine expression
            W_T_m = build_wealth_affine(T, m)
            
            # Shortfall: L[i] = threshold - W[i,T,m]
            # Positive values indicate goal violation in scenario i
            shortfall = goal.threshold - W_T_m
            
            # Create CVaR auxiliary variables
            # γ: VaR at level ε (scalar)
            # z: excess shortfall over VaR (vector, one per scenario)
            gamma[goal] = cp.Variable(name=f"gamma_terminal_{m}")
            z[goal] = cp.Variable(n_sims, nonneg=True, name=f"z_terminal_{m}")
            
            # CVaR epigraphic constraints:
            # (1) z[i] >= L[i] - γ  for all i (defines excess over VaR)
            # (2) z[i] >= 0         for all i (non-negativity)
            # (3) γ + (1/εN) Σ_i z[i] <= 0  (CVaR <= 0 ensures goal satisfaction)
            constraints.append(z[goal] >= shortfall - gamma[goal])
            constraints.append(
                gamma[goal] + cp.sum(z[goal]) / (goal.epsilon * n_sims) <= 0
            )
        
        # 3. CVaR constraints for intermediate goals
        for goal in goal_set.intermediate_goals:
            m = goal_set.get_account_index(goal)
            t = goal_set.get_resolved_month(goal)
            
            # Validate temporal consistency
            if t > T:
                raise ValueError(
                    f"Intermediate goal at month {t} exceeds horizon T={T}"
                )
            
            # Compute wealth at intermediate time t
            W_t_m = build_wealth_affine(t, m)
            
            # Shortfall at time t
            shortfall = goal.threshold - W_t_m
            
            # CVaR auxiliary variables for intermediate goal
            gamma[goal] = cp.Variable(name=f"gamma_intermediate_{m}_{t}")
            z[goal] = cp.Variable(n_sims, nonneg=True, name=f"z_intermediate_{m}_{t}")
            
            # Same epigraphic constraints as terminal goals
            constraints.append(z[goal] >= shortfall - gamma[goal])
            constraints.append(
                gamma[goal] + cp.sum(z[goal]) / (goal.epsilon * n_sims) <= 0
            )
        
        # ============= OBJECTIVE FUNCTION =============
        
        # Pre-compute total terminal wealth (used by multiple objectives)
        W_T_per_account = [build_wealth_affine(T, m) for m in range(M)]
        W_T_total_per_scenario = sum(W_T_per_account)  # (n_sims,) - element-wise sum
        mean_wealth = cp.sum(W_T_total_per_scenario) / n_sims
        
        # Build objective based on self.objective
        if self.objective == "risky":
            # Maximize expected total terminal wealth: E[Σ_m W[T,m]]
            # Linear program - fastest option
            objective = cp.Maximize(mean_wealth)
        
        elif self.objective == "conservative":
            # Maximize E[W_T] - λ·Var(W_T)
            # Variance formulated as sum of squared deviations for DCP compliance
            lambda_ = self.objective_params.get("lambda", 0.5)
            
            # Var(W) = (1/N) Σ_i (W_i - mean_W)²
            # DCP-compliant: sum_squares of affine expressions is convex
            variance = cp.sum_squares(W_T_total_per_scenario - mean_wealth) / n_sims
            
            objective = cp.Maximize(mean_wealth - lambda_ * variance)
        
        elif self.objective == "risky_turnover":
            # Maximize E[W_T] - λ·||ΔX||_2
            lambda_ = self.objective_params.get("lambda", 15000)
            turnover = cp.sum_squares(X[1:, :] - X[:-1, :]) if T > 1 else 0.0

            objective = cp.Maximize(mean_wealth - lambda_ * turnover)
        
        elif self.objective == "balanced":
            # Penalty on allocation changes ||ΔX||_2
            # Turnover component
            turnover = cp.sum_squares(X[1:, :] - X[:-1, :]) if T > 1 else 0.0
            
            objective = cp.Maximize(-turnover)
        
        else:
            raise ValueError(
                f"Unknown objective '{self.objective}'. "
                f"Valid options: 'risky', 'balanced', 'conservative'"
            )
        
        # ============= SOLVE CONVEX PROGRAM =============
        
        prob = cp.Problem(objective, constraints)
        
        # Configure solver
        solver_options = {
            'max_iters': max_iters,
            'abstol': abstol,
            'reltol': reltol,
        }
        
        # Select solver and solve
        if solver_name.upper() == 'ECOS':
            prob.solve(solver=cp.ECOS, verbose=verbose, **solver_options)
        elif solver_name.upper() == 'SCS':
            prob.solve(solver=cp.SCS, verbose=verbose, **solver_options)
        elif solver_name.upper() == 'CLARABEL':
            prob.solve(solver=cp.CLARABEL, verbose=verbose, **solver_options)
        else:
            # Let CVXPY auto-select solver
            prob.solve(verbose=verbose, **solver_options)
        
        solve_time = time.time() - start_time
        
        # ============= EXTRACT AND VALIDATE SOLUTION =============
        
        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Problem is infeasible or solver failed
            if verbose:
                print(f"\n[CVXPY Status] {prob.status}")
                print(f"  Problem is infeasible or solver failed.")
            
            # Return infeasible result with uniform allocation fallback
            X_star = np.ones((T, M)) / M
            
            diagnostics = {
                'solver_status': prob.status,
                'solver_time': solve_time,
                'optimal_value': None,
                'solver_name': solver_name,
            }
            
            return OptimizationResult(
                X=X_star,
                T=T,
                objective_value=0.0,
                feasible=False,
                goals=list(goal_set.intermediate_goals) + list(goal_set.terminal_goals),
                goal_set=goal_set,
                solve_time=solve_time,
                n_iterations=0,
                diagnostics=diagnostics
            )
        
        # Extract optimal allocation
        X_star = X.value
        
        if X_star is None:
            raise RuntimeError(
                f"Solver returned status '{prob.status}' but X.value is None. "
                f"This should not happen. Check CVXPY installation."
            )
        
        # Project to simplex if numerical violations exist
        # (Due to finite solver tolerance, Σ_m X[t,m] may be ≈ 1.0 ± 1e-8)
        for t in range(T):
            row_sum = X_star[t, :].sum()
            if abs(row_sum - 1.0) > 1e-6:
                # Renormalize: ensures exact simplex satisfaction
                X_star[t, :] = np.maximum(X_star[t, :], 0)  # Clip negatives
                X_star[t, :] /= X_star[t, :].sum()
        
        obj_value = prob.value
        
        # ============= DIAGNOSTICS =============
        
        if verbose:
            print(f"\n[CVXPY Solution]")
            print(f"  Status: {prob.status}")
            print(f"  Objective: {obj_value:,.2f}")
            print(f"  Solve time: {solve_time:.3f}s")
            
            # Validate simplex constraint
            X_sums = X_star.sum(axis=1)
            simplex_violations = np.abs(X_sums - 1.0)
            max_simplex_error = simplex_violations.max()
            
            print(f"\n[Simplex Validation]")
            print(f"  Max |Σx_t - 1|: {max_simplex_error:.2e}")
            print(f"  X bounds: [{X_star.min():.4f}, {X_star.max():.4f}]")
            
            if max_simplex_error > 1e-4:
                print(f"Minor simplex violations detected (auto-corrected)")
            
            # Compute wealth using NumPy for validation
            def compute_wealth_numpy(X, t, m):
                """Compute W[:,t,m] using NumPy (for diagnostic validation)"""
                W = W0[m] * F[:, 0, t, m]
                for s in range(t):
                    W += A[:, s] * X[s, m] * F[:, s, t, m]
                return W
            
            # Validate CVaR constraints and goal satisfaction
            print(f"\n[Goal Satisfaction Diagnostics]")
            
            for goal in goal_set.terminal_goals:
                m = goal_set.get_account_index(goal)
                W_T_m = compute_wealth_numpy(X_star, T, m)
                
                violations = W_T_m < goal.threshold
                violation_rate = violations.mean()
                
                # Compute actual CVaR value
                shortfall = goal.threshold - W_T_m
                gamma_val = gamma[goal].value
                z_val = z[goal].value
                cvar_val = gamma_val + z_val.sum() / (goal.epsilon * n_sims)
                
                print(f"  Account {m} ({goal.account}):")
                print(f"    Threshold:      {goal.threshold:>12,.0f}")
                print(f"    Mean wealth:    {W_T_m.mean():>12,.0f}")
                print(f"    Violation rate: {violation_rate:.2%} (max: {goal.epsilon:.2%})")
                print(f"    CVaR value:     {cvar_val:>12,.2f} (target: ≤ 0)")
                
                if cvar_val > 1e-4:
                    print(f"    ⚠️  CVaR constraint not satisfied within tolerance!")
            
            for goal in goal_set.intermediate_goals:
                m = goal_set.get_account_index(goal)
                t = goal_set.get_resolved_month(goal)
                W_t_m = compute_wealth_numpy(X_star, t, m)
                
                violations = W_t_m < goal.threshold
                violation_rate = violations.mean()
                
                gamma_val = gamma[goal].value
                z_val = z[goal].value
                cvar_val = gamma_val + z_val.sum() / (goal.epsilon * n_sims)
                
                print(f"  Account {m} at month {t} ({goal.account}):")
                print(f"    Threshold:      {goal.threshold:>12,.0f}")
                print(f"    Mean wealth:    {W_t_m.mean():>12,.0f}")
                print(f"    Violation rate: {violation_rate:.2%} (max: {goal.epsilon:.2%})")
                print(f"    CVaR value:     {cvar_val:>12,.2f} (target: ≤ 0)")
        
        # Exact feasibility check using base class validation
        feasible = self._check_feasibility(
            X_star, A, R, W0, 
            portfolio,
            goal_set
        )
        
        # Package diagnostics
        diagnostics = {
            'solver_status': prob.status,
            'solver_time': solve_time,
            'optimal_value': obj_value,
            'solver_name': solver_name,
            'n_constraints': len(constraints),
            'n_variables': T * M + len(gamma) * (1 + n_sims),
            'objective_type': self.objective,
        }
        
        return OptimizationResult(
            X=X_star,
            T=T,
            objective_value=obj_value,
            feasible=feasible,
            goals=list(goal_set.intermediate_goals) + list(goal_set.terminal_goals),
            goal_set=goal_set,
            solve_time=solve_time,
            n_iterations=0,  # CVXPY doesn't expose iteration count
            diagnostics=diagnostics
        )


# ---------------------------------------------------------------------------
# Goal Seeker (Bilevel Solver)
# ---------------------------------------------------------------------------

class GoalSeeker:
    """
    Bilevel optimizer: find minimum horizon T* for goal feasibility.
    
    Outer problem: min T ∈ ℕ
    Inner problem: AllocationOptimizer.solve(T) → feasible X*
    
    Supports linear and binary search strategies with warm start.
    
    Parameters
    ----------
    optimizer : AllocationOptimizer
        Inner problem solver (e.g., CVaROptimizer)
    T_max : int, default 240
        Maximum search horizon (20 years)
    verbose : bool, default True
        Print iteration progress
    
    Examples
    --------
    >>> from finopt.src.portfolio import Account
    >>> accounts = [
    ...     Account.from_annual("Emergency", 0.04, 0.05),
    ...     Account.from_annual("Housing", 0.07, 0.12)
    ... ]
    >>> 
    >>> optimizer = CVaROptimizer(n_accounts=model.M, objective='terminal_wealth')
    >>> seeker = GoalSeeker(optimizer, T_max=120, verbose=True)
    >>> 
    >>> def A_gen(T, n, s):
    ...     return model.income.contributions(T, n_sims=n, seed=s, output="array")
    >>> def R_gen(T, n, s):
    ...     return model.returns.generate(T, n_sims=n, seed=s)
    >>> 
    >>> result = seeker.seek(goals, A_gen, R_gen, W0, accounts,
    ...                     start_date=date(2025, 1, 1),
    ...                     n_sims=500, seed=42, search_method="binary")
    >>> print(f"Optimal horizon: T*={result.T} months")
    """
    
    def __init__(
        self,
        optimizer: AllocationOptimizer,
        T_max: int = 240,
        verbose: bool = True
    ):
        if not isinstance(optimizer, AllocationOptimizer):
            raise TypeError(
                f"optimizer must be AllocationOptimizer subclass, "
                f"got {type(optimizer)}"
            )
        if T_max < 1:
            raise ValueError(f"T_max must be ≥ 1, got {T_max}")
        
        self.optimizer = optimizer
        self.T_max = T_max
        self.verbose = verbose

    def seek(
        self,
        goals: List[Union[IntermediateGoal, TerminalGoal]],
        A_generator: Callable[[int, int, Optional[int]], np.ndarray],
        R_generator: Callable[[int, int, Optional[int]], np.ndarray],
        W0: np.ndarray,
        accounts: List[Account],
        start_date: date,
        n_sims: int = 500,
        seed: Optional[int] = None,
        search_method: str = "binary",
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Find minimum horizon T* for goal feasibility via intelligent search.
        
        Parameters
        ----------
        goals : List[Union[IntermediateGoal, TerminalGoal]]
            Goal specifications
        A_generator : callable
            Function (T, n_sims, seed) → A array of shape (n_sims, T)
        R_generator : callable
            Function (T, n_sims, seed) → R array of shape (n_sims, T, M)
        W0 : np.ndarray, shape (M,)
            Initial wealth vector
        accounts : List[Account]
            Portfolio accounts
        start_date : datetime.date
            Simulation start date
        n_sims : int, default 500
            Number of scenarios
        seed : int, optional
            Random seed for reproducibility
        search_method : str, default "binary"
            Search strategy:
            - "linear": Sequential T = T_start, T_start+1, ... (safer, slower)
            - "binary": Binary search (faster, requires monotonicity assumption)
        **solver_kwargs
            Parameters passed to optimizer.solve()
        
        Returns
        -------
        OptimizationResult
            Result at minimum feasible horizon T*
        
        Notes
        -----
        Binary search assumes monotonicity: if T is feasible, then T+1 is feasible.
        This holds for most practical goal structures but may fail with:
        - Time-dependent contribution schedules with sudden drops
        - Pathological goal configurations
        
        For safety-critical applications, use search_method="linear".
        For typical financial planning, "binary" saves ~50% iterations.
        """
        if not goals:
            raise ValueError("goals list cannot be empty")
        
        if not accounts:
            raise ValueError("accounts list cannot be empty")
        
        goal_set = GoalSet(goals, accounts, start_date)
        
        # Determine intelligent starting horizon
        if goal_set.terminal_goals and not goal_set.intermediate_goals:
            if self.verbose:
                print("\n=== Estimating minimum horizon (terminal goals only) ===")
            
            # Sample contributions
            sample_months = 12
            sample_sims = min(100, n_sims)
            
            A_sample = A_generator(sample_months, sample_sims, seed)
            avg_contrib = float(np.mean(A_sample))
            
            if self.verbose:
                print(f"  Sampled contributions: {sample_sims} scenarios × {sample_months} months")
                print(f"  Average monthly contribution: ${avg_contrib:,.2f}")
            
            # Use improved heuristic with account information
            T_start = goal_set.estimate_minimum_horizon(
                monthly_contribution=avg_contrib,
                accounts=accounts,
                expected_return=None,
                safety_margin=0.75,
                T_max=self.T_max
            )
            
            if self.verbose:
                print(f"  Estimated minimum horizon: T={T_start} months")
                print(f"  (using account-specific returns and safety_margin=0.75)")
        else:
            T_start = goal_set.T_min
        
        # Validate starting horizon
        if T_start > self.T_max:
            raise ValueError(
                f"Estimated T_start={T_start} > T_max={self.T_max}. "
                f"Increase T_max or reduce goal thresholds."
            )
        
        # Display search range
        if self.verbose:
            print(f"\n=== GoalSeeker: {search_method.upper()} search T ∈ [{T_start}, {self.T_max}] ===")
            if T_start > goal_set.T_min:
                print(f"    (T_start={T_start} from heuristic, T_min={goal_set.T_min} from constraints)")
        
        # Dispatch to search method
        if search_method == "linear":
            return self._linear_search(
                T_start, goal_set, A_generator, R_generator, 
                W0, n_sims, seed, **solver_kwargs
            )
        elif search_method == "binary":
            return self._binary_search(
                T_start, goal_set, A_generator, R_generator,
                W0, n_sims, seed, **solver_kwargs
            )
        else:
            raise ValueError(f"Unknown search_method: {search_method}")

    def _linear_search(
        self,
        T_start: int,
        goal_set: GoalSet,
        A_generator: Callable,
        R_generator: Callable,
        W0: np.ndarray,
        n_sims: int,
        seed: Optional[int],
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Linear search T = T_start, T_start+1, ... (original algorithm).

        """
        X_prev = None
        
        for T in range(T_start, self.T_max + 1):
            if self.verbose:
                print(f"[Iter {T - T_start + 1}] Testing T={T}...")
            
            A = A_generator(T, n_sims, seed)
            R = R_generator(T, n_sims, None if seed is None else seed + 1)
            
            result = self.optimizer.solve(
                T=T, A=A, R=R, W0=W0,
                goal_set=goal_set,
                X_init=X_prev,
                **solver_kwargs
            )
            
            if self.verbose:
                status = "✓ Feasible" if result.feasible else "✗ Infeasible"
                print(f"    {status}, obj={result.objective_value:.2f}, "
                    f"time={result.solve_time:.3f}s\n")
            
            if result.feasible:
                if self.verbose:
                    print(f"=== Optimal: T*={T} ===\n")
                return result
            
            if result.X is not None:
                X_prev = np.vstack([result.X, result.X[-1:, :]])
        
        raise ValueError(
            f"No feasible solution found in T ∈ [{T_start}, {self.T_max}]. "
            f"Try increasing T_max or relaxing goal constraints."
        )

    def _binary_search(
        self,
        T_start: int,
        goal_set: GoalSet,
        A_generator: Callable,
        R_generator: Callable,
        W0: np.ndarray,
        n_sims: int,
        seed: Optional[int],
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Binary search for minimum feasible T.
        
        Assumes monotonicity: if T is feasible, then T' > T is also feasible.
        Reduces iterations from O(T_max - T_start) to O(log(T_max - T_start)).
        
        Algorithm
        ---------
        1. Initialize: left = T_start, right = T_max
        2. While left < right:
           a. mid = (left + right) // 2
           b. Test feasibility at mid
           c. If feasible: right = mid (search lower half)
           d. If infeasible: left = mid + 1 (search upper half)
        3. Return solution at left
        """
        left = T_start
        right = self.T_max
        
        best_result = None
        iteration = 0
        
        while left < right:
            iteration += 1
            mid = (left + right) // 2
            
            if self.verbose:
                print(f"[Iter {iteration}] Binary search: testing T={mid} "
                    f"(range=[{left}, {right}])...")
            
            # Generate scenarios for mid
            A = A_generator(mid, n_sims, seed)
            R = R_generator(mid, n_sims, None if seed is None else seed + 1)
            
            # Warm start: if we have a previous result, extend it
            X_init = None
            if best_result is not None and best_result.T < mid:
                # Extend previous X by repeating last row
                n_extend = mid - best_result.T
                X_init = np.vstack([
                    best_result.X,
                    np.tile(best_result.X[-1:, :], (n_extend, 1))
                ])
            
            result = self.optimizer.solve(
                T=mid, A=A, R=R, W0=W0,
                goal_set=goal_set,
                X_init=X_init,
                **solver_kwargs
            )
            
            if self.verbose:
                status = "✓ Feasible" if result.feasible else "✗ Infeasible"
                print(f"    {status}, obj={result.objective_value:.2f}, "
                    f"time={result.solve_time:.3f}s\n")
            
            if result.feasible:
                # Found feasible solution: try lower half
                best_result = result
                right = mid
            else:
                # Infeasible: try upper half
                left = mid + 1
        
        # At convergence: left == right
        if best_result is not None and best_result.T == left:
            if self.verbose:
                print(f"=== Optimal: T*={left} (binary search converged) ===\n")
            return best_result
        
        # Need to verify left (may not have been tested)
        if self.verbose:
            print(f"[Iter {iteration + 1}] Verifying T={left} (final check)...")
        
        A = A_generator(left, n_sims, seed)
        R = R_generator(left, n_sims, None if seed is None else seed + 1)
        
        result = self.optimizer.solve(
            T=left, A=A, R=R, W0=W0,
            goal_set=goal_set,
            X_init=None,
            **solver_kwargs
        )
        
        if result.feasible:
            if self.verbose:
                print(f"=== Optimal: T*={left} ===\n")
            return result
        
        raise ValueError(
            f"Binary search failed: T={left} infeasible. "
            f"Monotonicity assumption may be violated."
        )