# finopt/src/optimization.py
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

Outer problem: Linear search over horizon T (GoalSeeker)
Inner problem: Convex program via affine wealth representation (AllocationOptimizer)

Key Components
--------------
- OptimizationResult: Container for X*, T*, objective value, goal_set, and diagnostics
- AllocationOptimizer: Abstract base with parametrizable objectives f(X)
- CVaROptimizer: Risk-adjusted optimization (stub for future CVXPY implementation)
- SAAOptimizer: Sample Average Approximation with sigmoid smoothing
- GoalSeeker: Bilevel solver with linear search and warm start

Design Principles
-----------------
- Separation of concerns: Goals defined in goals.py, solvers here
- Scenario-driven: Receives pre-generated (A, R) from FinancialModel
- Solver-agnostic: Abstract base allows scipy, CVXPY, or custom solvers
- Optimization-ready: Exploits affine wealth W_t^m(X) for analytical gradients
- Portfolio-aware: Consumes Account objects for type safety and metadata access
- Reproducible: Explicit seed management for stochastic scenarios
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any, Union, Literal, TYPE_CHECKING
from abc import ABC, abstractmethod
import time
import numpy as np

from .goals import IntermediateGoal, TerminalGoal, GoalSet, check_goals

if TYPE_CHECKING:
    from .model import SimulationResult
    from .portfolio import Account

__all__ = [
    "OptimizationResult",
    "AllocationOptimizer",
    "CVaROptimizer",
    "SAAOptimizer",
    "GoalSeeker",
]


# Type alias for objective specifications
ObjectiveType = Union[
    Literal["terminal_wealth", "low_turnover", "risk_adjusted", "balanced"],
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
    - "terminal_wealth": E[Σ_m W_T^m] (default)
    - "low_turnover": E[W_T] - λ·Σ_{t,m}|x_{t+1,m} - x_t^m|
    - "risk_adjusted": E[W_T] - λ·Std(W_T)
    - "balanced": Combination of above
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
    - solve(T, A, R, W0, goals, accounts, start_date, goal_set, X_init, **kwargs) 
      → OptimizationResult
    
    Provided utilities:
    - _validate_inputs(...) → GoalSet
    - _check_feasibility(...) → bool
    - _compute_objective(W, X, T, M) → float (dispatches to objective functions)
    """
    
    def __init__(
        self,
        n_accounts: int,
        objective: ObjectiveType = "terminal_wealth",
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
        goals: List[Union[IntermediateGoal, TerminalGoal]],
        accounts: List[Account],
        start_date,
        goal_set: Optional[GoalSet] = None,
        X_init: Optional[np.ndarray] = None,
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Solve allocation optimization problem for fixed horizon T.
        
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
        goals : List[Union[IntermediateGoal, TerminalGoal]]
            Goal specifications
        accounts : List[Account]
            Portfolio accounts (source of truth for metadata)
        start_date : datetime.date
            Simulation start date for goal resolution
        goal_set : GoalSet, optional
            Pre-validated goal collection. If None, constructed internally.
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
    
    def _validate_inputs(
        self,
        T: int,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        goals: List[Union[IntermediateGoal, TerminalGoal]],
        accounts: List[Account],
        start_date
    ) -> GoalSet:
        """
        Validate inputs and return GoalSet.
        
        Parameters
        ----------
        accounts : List[Account]
            Portfolio accounts for GoalSet construction
        start_date : datetime.date
            Simulation start date for goal resolution
        
        Returns
        -------
        GoalSet
            Validated goal collection with resolved accounts
        """
        if not isinstance(T, int) or T < 1:
            raise ValueError(f"T must be positive int, got {T}")
        
        if A.ndim != 2 or A.shape[1] != T:
            raise ValueError(f"A must have shape (n_sims, T), got {A.shape}")
        
        if R.ndim != 3 or R.shape[1:] != (T, self.M):
            raise ValueError(f"R must have shape (n_sims, T, M), got {R.shape}")
        
        if A.shape[0] != R.shape[0]:
            raise ValueError(
                f"A and R must have same n_sims: {A.shape[0]} != {R.shape[0]}"
            )
        
        if W0.shape != (self.M,):
            raise ValueError(f"W0 must have shape (M,), got {W0.shape}")
        
        if not goals:
            raise ValueError("goals list cannot be empty")
        
        if not accounts:
            raise ValueError("accounts list cannot be empty")
        
        if len(accounts) != self.M:
            raise ValueError(
                f"accounts length {len(accounts)} != n_accounts {self.M}"
            )
        
        # Construct GoalSet with Account objects
        goal_set = GoalSet(goals, accounts, start_date)
        
        if goal_set.T_min > T:
            raise ValueError(
                f"Goals require T ≥ {goal_set.T_min}, but got T={T}. "
                f"Latest intermediate goal at month {goal_set.T_min}."
            )
        
        return goal_set
    
    def _check_feasibility(
        self,
        X: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        accounts: List[Account],
        goal_set: GoalSet
    ) -> bool:
        """
        Check if allocation X satisfies all goals using exact SAA.
        
        Uses non-smoothed indicator function for final validation.
        Uses Portfolio.simulate with W0_override to avoid creating dummy accounts.
        
        Parameters
        ----------
        accounts : List[Account]
            Portfolio accounts (metadata only, W0 overridden)
        """
        # Import here to avoid circular dependency
        from .portfolio import Portfolio
        
        # Create portfolio with accounts (W0 will be overridden)
        portfolio = Portfolio(accounts)
        
        # Simulate wealth with W0 override
        result = portfolio.simulate(A=A, R=R, X=X, method="affine", W0_override=W0)
        W = result["wealth"]  # (n_sims, T+1, M)
        
        n_sims = W.shape[0]
        
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
        if self.objective == "terminal_wealth":
            return self._objective_terminal_wealth(W, X, T, M)
        elif self.objective == "low_turnover":
            return self._objective_low_turnover(W, X, T, M)
        elif self.objective == "risk_adjusted":
            return self._objective_risk_adjusted(W, X, T, M)
        elif self.objective == "balanced":
            return self._objective_balanced(W, X, T, M)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
    
    def _objective_terminal_wealth(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """Expected total terminal wealth: E[Σ_m W_T^m]."""
        return W[:, T, :].sum(axis=1).mean()
    
    def _objective_low_turnover(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """
        Terminal wealth with turnover penalty.
        
        f(X) = E[W_T] - λ·Σ_{t,m} |x_{t+1,m} - x_t^m|
        """
        lambda_ = self.objective_params.get("lambda", 0.1)
        
        terminal_wealth = W[:, T, :].sum(axis=1).mean()
        
        if T > 1:
            turnover = np.abs(X[1:, :] - X[:-1, :]).sum()
        else:
            turnover = 0.0
        
        return terminal_wealth - lambda_ * turnover
    
    def _objective_risk_adjusted(
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
    
    def _objective_balanced(
        self,
        W: np.ndarray,
        X: np.ndarray,
        T: int,
        M: int
    ) -> float:
        """
        Balanced objective combining wealth, risk, and turnover.
        
        f(X) = E[W_T] - λ_risk·Std(W_T) - λ_turnover·Turnover(X)
        """
        lambda_risk = self.objective_params.get("lambda_risk", 0.3)
        lambda_turnover = self.objective_params.get("lambda_turnover", 0.05)
        
        W_T_total = W[:, T, :].sum(axis=1)
        mean_wealth = W_T_total.mean()
        std_wealth = W_T_total.std()
        
        if T > 1:
            turnover = np.abs(X[1:, :] - X[:-1, :]).sum()
        else:
            turnover = 0.0
        
        return mean_wealth - lambda_risk * std_wealth - lambda_turnover * turnover


# ---------------------------------------------------------------------------
# CVaR Optimizer (Stub)
# ---------------------------------------------------------------------------

class CVaROptimizer(AllocationOptimizer):
    """
    CVaR-based allocation optimizer with risk-adjusted objective.
    
    **STUB IMPLEMENTATION**: Requires CVXPY for full functionality.
    
    Mathematical Formulation
    ------------------------
    Objective:
        maximize  E[W_T] - λ·CVaR_α(-W_T)
    
    where CVaR_α (Conditional Value-at-Risk) is computed via auxiliary variables:
        CVaR_α(loss) = ξ + (1/(α·n))Σ_i u_i
        u_i ≥ loss_i - ξ
        u_i ≥ 0
    
    Parameters
    ----------
    n_accounts : int
        Number of portfolio accounts
    risk_aversion : float, default 0.5
        Risk-return tradeoff parameter λ ≥ 0
    alpha : float, default 0.95
        CVaR confidence level α ∈ (0.5, 1.0)
    objective : ObjectiveType, default "terminal_wealth"
        Base objective (applied before CVaR adjustment)
    objective_params : dict, optional
        Additional objective parameters
    account_names : List[str], optional
        Account name labels
    
    Notes
    -----
    Implementation requires CVXPY. See NotImplementedError message for
    detailed implementation checklist.
    
    References
    ----------
    Rockafellar & Uryasev (2000), "Optimization of conditional value-at-risk"
    """
    
    def __init__(
        self,
        n_accounts: int,
        risk_aversion: float = 0.5,
        alpha: float = 0.95,
        objective: ObjectiveType = "terminal_wealth",
        objective_params: Optional[Dict[str, Any]] = None,
        account_names: Optional[List[str]] = None
    ):
        super().__init__(n_accounts, objective, objective_params, account_names)
        
        if risk_aversion < 0:
            raise ValueError(f"risk_aversion must be ≥ 0, got {risk_aversion}")
        if not (0.5 < alpha < 1.0):
            raise ValueError(f"alpha must be ∈ (0.5, 1.0), got {alpha}")
        
        self.lambda_ = risk_aversion
        self.alpha = alpha
    
    def solve(
        self,
        T: int,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        goals: List[Union[IntermediateGoal, TerminalGoal]],
        accounts: List[Account],
        start_date,
        goal_set: Optional[GoalSet] = None,
        X_init: Optional[np.ndarray] = None,
        **solver_kwargs
    ) -> OptimizationResult:
        """Solve CVaR-constrained allocation problem."""
        raise NotImplementedError(
            "CVaROptimizer.solve() requires CVXPY implementation.\n\n"
            "Implementation checklist:\n"
            "1. Install CVXPY: pip install cvxpy\n"
            "2. Import: import cvxpy as cp\n"
            "3. Validate inputs:\n"
            "   goal_set = self._validate_inputs(T, A, R, W0, goals, accounts, start_date)\n"
            "4. Compute F: from .portfolio import Portfolio\n"
            "   portfolio = Portfolio(accounts)\n"
            "   F = portfolio.compute_accumulation_factors(R)\n"
            "5. Variables:\n"
            "   - X = cp.Variable((T, M), nonneg=True)\n"
            "   - ξ_obj = cp.Variable()\n"
            "   - u = cp.Variable(n_sims, nonneg=True)\n"
            "   - For each goal: ξ_g = cp.Variable(), "
            "v_g = cp.Variable(n_sims, nonneg=True)\n"
            "6. Affine wealth: W_t_m[i] = W0[m]*F[i,0,t,m] + "
            "sum(A[i,s]*X[s,m]*F[i,s,t,m] for s in range(t))\n"
            "7. Objective:\n"
            "   mean_wealth = cp.sum([cp.sum(W_T[i,:]) for i in range(n_sims)]) / n_sims\n"
            "   cvar_obj = ξ_obj + cp.sum(u) / (self.alpha * n_sims)\n"
            "   objective = cp.Maximize(mean_wealth - self.lambda_ * cvar_obj)\n"
            "8. Constraints:\n"
            "   - u[i] >= -W_T[i] - ξ_obj for all i\n"
            "   - For each goal: ξ_g + cp.sum(v_g) / (goal.epsilon * n_sims) <= -threshold\n"
            "   - v_g[i] >= -W_t_m[i] - ξ_g for all i\n"
            "   - cp.sum(X[t,:]) == 1 for all t\n"
            "9. Solve: prob.solve(solver=solver_kwargs.get('solver', 'ECOS'), ...)\n"
            "10. Extract: X_star = X.value, obj = prob.value\n"
            "11. Validate:\n"
            "    feasible = self._check_feasibility(X_star, A, R, W0, accounts, goal_set)\n"
            "12. Return OptimizationResult(X=X_star, T=T, ..., goal_set=goal_set)\n\n"
            "Reference: Rockafellar & Uryasev (2000), "
            "'Optimization of conditional value-at-risk'"
        )


# ---------------------------------------------------------------------------
# SAA Optimizer (Smoothed Sigmoid Approximation)
# ---------------------------------------------------------------------------

class SAAOptimizer(AllocationOptimizer):
    """
    Sample Average Approximation optimizer with smoothed sigmoid constraints.
    
    Mathematical Formulation
    ------------------------
    Objective:
        maximize  f(X) (parametrizable via objective argument)
    
    Subject to:
        Intermediate goals: ℙ(W_t^m ≥ b) ≥ 1 - ε  (fixed t)
        Terminal goals: ℙ(W_T^m ≥ b) ≥ 1 - ε  (variable T)
        Σ_m x_t^m = 1
        x_t^m ≥ 0
    
    Smoothed Approximation
    ----------------------
    The discontinuous indicator 𝟙{W ≥ b} is replaced by sigmoid:
    
        𝟙{W_i ≥ b}  ≈  σ((W_i - b)/τ)
        
        where σ(x) = 1/(1 + exp(-x))
    
    Chance constraint becomes:
        (1/n) Σ_i σ((W_t^{m,i}(X) - b)/τ) ≥ 1 - ε
    
    Key Properties
    --------------
    1. Differentiability: σ is C^∞ → analytical gradients
       σ'(x) = σ(x)(1 - σ(x))
    
    2. Approximation quality: Controlled by temperature τ
       - Small τ (0.01): σ ≈ 𝟙 (better approximation, steeper gradient)
       - Large τ (1.0): σ ≈ 0.5 (smoother, worse approximation)
       - Balanced τ (0.1): Trade-off for practical optimization
    
    3. Convexity: Smoothed constraint is convex (albeit non-linear)
       Enables gradient-based solvers like SLSQP
    
    Parameters
    ----------
    n_accounts : int
        Number of portfolio accounts
    tau : float, default 0.1
        Sigmoid temperature parameter (smaller = sharper approximation)
    objective : ObjectiveType, default "terminal_wealth"
        Objective function specification
    objective_params : dict, optional
        Parameters for objective function
    account_names : List[str], optional
        Account name labels
    
    Examples
    --------
    >>> # Sharp approximation (may be harder to optimize)
    >>> opt_sharp = SAAOptimizer(n_accounts=2, tau=0.01)
    >>> 
    >>> # Smooth approximation (easier convergence)
    >>> opt_smooth = SAAOptimizer(n_accounts=2, tau=1.0)
    >>> 
    >>> # Balanced (recommended)
    >>> opt = SAAOptimizer(n_accounts=2, tau=0.1, 
    ...                   objective="low_turnover",
    ...                   objective_params={"lambda": 0.1})
    >>> result = opt.solve(T=24, A=A, R=R, W0=W0, goals=goals, 
    ...                   accounts=accounts, start_date=date(2025,1,1))
    
    References
    ----------
    Luedtke & Ahmed (2008), "A sample approximation approach for 
    optimization with probabilistic constraints"
    """
    
    def __init__(
        self,
        n_accounts: int,
        tau: float = 0.1,
        objective: ObjectiveType = "terminal_wealth",
        objective_params: Optional[Dict[str, Any]] = None,
        account_names: Optional[List[str]] = None
    ):
        super().__init__(n_accounts, objective, objective_params, account_names)
        
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        
        self.tau = tau
    
    def solve(
        self,
        T: int,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        goals: List[Union[IntermediateGoal, TerminalGoal]],
        accounts: List[Account],
        start_date,
        goal_set: Optional[GoalSet] = None,
        X_init: Optional[np.ndarray] = None,
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Solve SAA problem with smoothed sigmoid approximation.
        
        Uses scipy.optimize.minimize with SLSQP method for gradient-based
        optimization with equality and inequality constraints.
        
        Parameters
        ----------
        T : int
            Optimization horizon
        A : np.ndarray, shape (n_sims, T)
            Contribution scenarios
        R : np.ndarray, shape (n_sims, T, M)
            Return scenarios
        W0 : np.ndarray, shape (M,)
            Initial wealth vector
        goals : List[Union[IntermediateGoal, TerminalGoal]]
            Goal specifications
        accounts : List[Account]
            Portfolio accounts (metadata and initial wealth reference)
        start_date : datetime.date
            Simulation start date for goal resolution
        goal_set : GoalSet, optional
            Pre-validated goal collection. If None, constructed internally.
        X_init : np.ndarray, optional
            Warm start allocation (T, M). If None, uses uniform (1/M).
        **solver_kwargs
            Solver parameters:
            - maxiter : int, default 1000
            - gtol : float, default 1e-6
            - ftol : float, default 1e-9
            - verbose : bool, default False
        
        Returns
        -------
        OptimizationResult
            Optimal X*, objective value, feasibility, goal_set, diagnostics
        
        Raises
        ------
        ImportError
            If scipy not installed
        RuntimeError
            If solver fails to converge
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "scipy is required for SAAOptimizer. Install with: pip install scipy"
            )
        
        start_time = time.time()
        
        # Validate inputs and construct GoalSet if not provided
        if goal_set is None:
            goal_set = self._validate_inputs(T, A, R, W0, goals, accounts, start_date)
        
        # Extract parameters
        n_sims = A.shape[0]
        M = self.M
        maxiter = solver_kwargs.get('maxiter', 1000)
        gtol = solver_kwargs.get('gtol', 1e-6)
        ftol = solver_kwargs.get('ftol', 1e-9)
        verbose = solver_kwargs.get('verbose', False)
        
        # Compute accumulation factors F: (n_sims, T+1, T+1, M)
        from .portfolio import Portfolio
        
        portfolio = Portfolio(accounts)
        F = portfolio.compute_accumulation_factors(R)
        
        # Sigmoid functions
        def sigmoid(z):
            """Numerically stable sigmoid."""
            return np.where(
                z >= 0,
                1 / (1 + np.exp(-z)),
                np.exp(z) / (1 + np.exp(z))
            )
        
        def sigmoid_prime(z):
            """Derivative of sigmoid."""
            s = sigmoid(z)
            return s * (1 - s)
        
        # Compute wealth for all scenarios given X
        def compute_wealth_all(X_flat):
            """
            Compute W: (n_sims, T+1, M) using affine formula.
            
            W_t^m = W0^m * F_{0,t}^m + Σ_{s<t} A_s * x_s^m * F_{s,t}^m
            """
            X = X_flat.reshape(T, M)
            W = np.zeros((n_sims, T + 1, M))
            
            # Initial wealth
            W[:, 0, :] = W0
            
            # Affine formula for each time step
            for t in range(1, T + 1):
                # Initial wealth term
                W[:, t, :] = W0 * F[:, 0, t, :]
                
                # Contribution term
                for s in range(t):
                    # A_s * x_s^m: (n_sims, 1) * (1, M) → (n_sims, M)
                    contrib = A[:, s, None] * X[s, :]
                    # contrib * F_{s,t}^m
                    W[:, t, :] += contrib * F[:, s, t, :]
            
            return W
        
        # Objective function (negative for minimization)
        def objective(X_flat):
            X = X_flat.reshape(T, M)
            W = compute_wealth_all(X_flat)
            obj_value = self._compute_objective(W, X, T, M)
            return -obj_value  # Negative for minimization
        
        # Gradient of objective (analytical)
        def objective_grad(X_flat):
            """
            Gradient of objective w.r.t. X (flattened).
            
            For terminal_wealth: ∂E[W_T]/∂x_s^m = E[A_s * F_{s,T}^m]
            """
            X = X_flat.reshape(T, M)
            grad = np.zeros((T, M))
            
            if self.objective == "terminal_wealth":
                # ∂E[Σ_m W_T^m]/∂x_s^m = E[A_s * F_{s,T}^m]
                for s in range(T):
                    for m in range(M):
                        grad[s, m] = (A[:, s] * F[:, s, T, m]).mean()
            
            elif self.objective == "low_turnover":
                # Terminal wealth gradient + turnover penalty gradient
                lambda_ = self.objective_params.get("lambda", 0.1)
                
                # Terminal wealth part
                for s in range(T):
                    for m in range(M):
                        grad[s, m] = (A[:, s] * F[:, s, T, m]).mean()
                
                # Turnover penalty: -λ * sign(x_{t+1} - x_t)
                if T > 1:
                    for t in range(T - 1):
                        grad[t, :] -= lambda_ * np.sign(X[t + 1, :] - X[t, :])
                        grad[t + 1, :] += lambda_ * np.sign(X[t + 1, :] - X[t, :])
            
            else:
                # For complex objectives, use numerical approximation
                eps = 1e-6
                for i in range(T * M):
                    X_plus = X_flat.copy()
                    X_plus[i] += eps
                    grad.flat[i] = (objective(X_plus) - objective(X_flat)) / eps
            
            return -grad.flatten()  # Negative because we minimize -obj
        
        # Constraint: smoothed intermediate goal
        def constraint_intermediate_goal(X_flat, goal: IntermediateGoal, m: int, t: int):
            """
            Smoothed intermediate goal constraint (≥ 0 for feasibility).
            
            c(X) = (1/n)Σ_i σ((W_t^{m,i} - b)/τ) - (1 - ε) ≥ 0
            """
            W = compute_wealth_all(X_flat)
            W_t_m = W[:, t, m]  # (n_sims,)
            
            z = (W_t_m - goal.threshold) / self.tau
            satisfaction = sigmoid(z).mean()
            
            return satisfaction - (1 - goal.epsilon)
        
        # Gradient of intermediate goal constraint
        def constraint_intermediate_goal_grad(X_flat, goal: IntermediateGoal, m: int, t: int):
            """
            Gradient of smoothed intermediate constraint.
            
            ∂c/∂x_s^m = (1/(n·τ)) Σ_i σ'((W_t^i - b)/τ) * A_s * F_{s,t}^m
            """
            W = compute_wealth_all(X_flat)
            W_t_m = W[:, t, m]
            
            z = (W_t_m - goal.threshold) / self.tau
            sigmoid_deriv = sigmoid_prime(z) / self.tau  # (n_sims,)
            
            grad = np.zeros(T * M)
            for s in range(t):  # Only s < t contribute
                for m_idx in range(M):
                    idx = s * M + m_idx
                    if m_idx == m:
                        # Gradient component
                        grad[idx] = (sigmoid_deriv * A[:, s] * F[:, s, t, m]).mean()
            
            return grad
        
        # Constraint: smoothed terminal goal
        def constraint_terminal_goal(X_flat, goal: TerminalGoal, m: int):
            """
            Smoothed terminal goal constraint (≥ 0 for feasibility).
            
            c(X) = (1/n)Σ_i σ((W_T^{m,i} - b)/τ) - (1 - ε) ≥ 0
            """
            W = compute_wealth_all(X_flat)
            W_T_m = W[:, T, m]  # (n_sims,)
            
            z = (W_T_m - goal.threshold) / self.tau
            satisfaction = sigmoid(z).mean()
            
            return satisfaction - (1 - goal.epsilon)
        
        # Gradient of terminal goal constraint
        def constraint_terminal_goal_grad(X_flat, goal: TerminalGoal, m: int):
            """
            Gradient of smoothed terminal constraint.
            
            ∂c/∂x_s^m = (1/(n·τ)) Σ_i σ'((W_T^i - b)/τ) * A_s * F_{s,T}^m
            """
            W = compute_wealth_all(X_flat)
            W_T_m = W[:, T, m]
            
            z = (W_T_m - goal.threshold) / self.tau
            sigmoid_deriv = sigmoid_prime(z) / self.tau  # (n_sims,)
            
            grad = np.zeros(T * M)
            for s in range(T):
                for m_idx in range(M):
                    idx = s * M + m_idx
                    if m_idx == m:
                        # Gradient component
                        grad[idx] = (sigmoid_deriv * A[:, s] * F[:, s, T, m]).mean()
            
            return grad
        
        # Constraint: simplex (Σ_m x_t^m = 1)
        def constraint_simplex(X_flat):
            """Equality constraint: row sums = 1."""
            X = X_flat.reshape(T, M)
            return X.sum(axis=1) - 1.0  # Shape: (T,)
        
        def constraint_simplex_jac(X_flat):
            """Jacobian of simplex constraint: (T, T*M)."""
            jac = np.zeros((T, T * M))
            for t in range(T):
                jac[t, t * M:(t + 1) * M] = 1.0
            return jac
        
        # Build constraints list
        constraints = []
        
        # Simplex constraints
        constraints.append({
            'type': 'eq',
            'fun': constraint_simplex,
            'jac': constraint_simplex_jac
        })
        
        # Intermediate goal constraints
        for goal in goal_set.intermediate_goals:
            m = goal_set.get_account_index(goal)
            t = goal_set.get_resolved_month(goal)
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda X, g=goal, m_=m, t_=t: 
                    constraint_intermediate_goal(X, g, m_, t_),
                'jac': lambda X, g=goal, m_=m, t_=t:
                    constraint_intermediate_goal_grad(X, g, m_, t_)
            })
        
        # Terminal goal constraints
        for goal in goal_set.terminal_goals:
            m = goal_set.get_account_index(goal)
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda X, g=goal, m_=m: constraint_terminal_goal(X, g, m_),
                'jac': lambda X, g=goal, m_=m: constraint_terminal_goal_grad(X, g, m_)
            })
        
        # Initial guess
        if X_init is None:
            X0 = np.ones((T, M)) / M  # Uniform allocation
        else:
            if X_init.shape != (T, M):
                raise ValueError(f"X_init shape {X_init.shape} != (T={T}, M={M})")
            X0 = X_init.copy()
        
        X0_flat = X0.flatten()
        
        # Bounds: x_t^m ∈ [0, 1]
        bounds = [(0.0, 1.0) for _ in range(T * M)]
        
        # Solve
        result = minimize(
            objective,
            X0_flat,
            method='SLSQP',
            jac=objective_grad,
            constraints=constraints,
            bounds=bounds,
            options={
                'maxiter': maxiter,
                'ftol': ftol,
                'disp': verbose
            }
        )
        
        solve_time = time.time() - start_time
        
        # Extract solution
        X_star = result.x.reshape(T, M)
        obj_value = -result.fun  # Convert back to maximization
        
        # Check feasibility with exact SAA
        feasible = self._check_feasibility(X_star, A, R, W0, accounts, goal_set)
        
        # Diagnostics
        diagnostics = {
            'convergence_status': result.message,
            'success': result.success,
            'final_gradient_norm': np.linalg.norm(result.jac) if result.jac is not None else None,
        }
        
        return OptimizationResult(
            X=X_star,
            T=T,
            objective_value=obj_value,
            feasible=feasible,
            goals=goals,
            goal_set=goal_set,
            solve_time=solve_time,
            n_iterations=result.nit,
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
    
    Uses linear search with warm start for efficiency.
    
    Parameters
    ----------
    optimizer : AllocationOptimizer
        Inner problem solver (e.g., SAAOptimizer, CVaROptimizer)
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
    >>> optimizer = SAAOptimizer(n_accounts=2, tau=0.1)
    >>> seeker = GoalSeeker(optimizer, T_max=120, verbose=True)
    >>> 
    >>> def A_gen(T, n, s):
    ...     return model.income.contributions(T, n_sims=n, seed=s, output="array")
    >>> def R_gen(T, n, s):
    ...     return model.returns.generate(T, n_sims=n, seed=s)
    >>> 
    >>> result = seeker.seek(goals, A_gen, R_gen, W0, accounts,
    ...                     start_date=date(2025, 1, 1),
    ...                     n_sims=500, seed=42)
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
        start_date,
        n_sims: int = 500,
        seed: Optional[int] = None,
        **solver_kwargs
    ) -> OptimizationResult:
        """
        Find minimum horizon T* for goal feasibility via linear search.
        
        Uses intelligent starting point estimation for terminal-only goals:
        - If only terminal goals exist: estimates T_start via heuristic
        - If intermediate goals exist: uses T_min from intermediate constraints
        
        Parameters
        ----------
        goals : List[Union[IntermediateGoal, TerminalGoal]]
            Goal specifications
        A_generator : Callable[[int, int, Optional[int]], np.ndarray]
            Function: (T, n_sims, seed) → A array (n_sims, T)
        R_generator : Callable[[int, int, Optional[int]], np.ndarray]
            Function: (T, n_sims, seed) → R array (n_sims, T, M)
        W0 : np.ndarray, shape (M,)
            Initial wealth vector
        accounts : List[Account]
            Portfolio accounts (metadata for goal resolution and simulation)
        start_date : datetime.date
            Simulation start date for goal resolution
        n_sims : int, default 500
            Number of Monte Carlo scenarios
        seed : int, optional
            Random seed for reproducibility
        **solver_kwargs
            Additional parameters for optimizer.solve()
        
        Returns
        -------
        OptimizationResult
            Optimal solution at minimum feasible T*
        
        Raises
        ------
        ValueError
            If T_min > T_max or no feasible solution found
        
        Notes
        -----
        For terminal-only goals, the method samples contributions to estimate
        average monthly cash flow, then uses worst-case accumulation formula
        to compute a conservative starting horizon. This avoids testing
        infeasible horizons (e.g., T=1, 2, 3...) when goals require T >> 1.
        """
        if not goals:
            raise ValueError("goals list cannot be empty")
        
        if not accounts:
            raise ValueError("accounts list cannot be empty")
        
        # Construct GoalSet with accounts
        goal_set = GoalSet(goals, accounts, start_date)
        
        # Determine intelligent starting horizon
        if goal_set.terminal_goals and not goal_set.intermediate_goals:
            # Only terminal goals: use heuristic estimation
            if self.verbose:
                print("\n=== Estimating minimum horizon (terminal goals only) ===")
            
            # Sample contributions to estimate average monthly cash flow
            # Use small sample for efficiency (12 months, min(100, n_sims) scenarios)
            sample_months = 12
            sample_sims = min(100, n_sims)
            
            A_sample = A_generator(sample_months, sample_sims, seed)
            avg_contrib = float(np.mean(A_sample))
            
            if self.verbose:
                print(f"  Sampled contributions: {sample_sims} scenarios × {sample_months} months")
                print(f"  Average monthly contribution: ${avg_contrib:,.2f}")
            
            # Estimate minimum horizon via worst-case analysis
            T_start = goal_set.estimate_minimum_horizon(
                monthly_contribution=avg_contrib,
                expected_return=0.0,      # Conservative: assume no growth
                safety_margin=0.8,         # Start 20% earlier to avoid missing T*
                T_max=self.T_max
            )
            
            if self.verbose:
                print(f"  Estimated minimum horizon: T={T_start} months")
                print(f"  (using safety_margin=0.8 for conservative start)")
        else:
            # Has intermediate goals: use their constraint
            T_start = goal_set.T_min
        
        # Validate starting horizon
        if T_start > self.T_max:
            raise ValueError(
                f"Estimated T_start={T_start} > T_max={self.T_max}. "
                f"Increase T_max or reduce goal thresholds."
            )
        
        # Display search range
        if self.verbose:
            print(f"\n=== GoalSeeker: Linear search T ∈ [{T_start}, {self.T_max}] ===")
            if T_start > goal_set.T_min:
                print(f"    (T_start={T_start} from heuristic, T_min={goal_set.T_min} from constraints)")
            print()
        
        # Linear search with warm start
        X_prev = None
        
        for T in range(T_start, self.T_max + 1):
            if self.verbose:
                print(f"[Iter {T - T_start + 1}] Testing T={T}...")
            
            # Generate scenarios for current T
            A = A_generator(T, n_sims, seed)
            R = R_generator(T, n_sims, None if seed is None else seed + 1)
            
            # Solve inner problem
            result = self.optimizer.solve(
                T=T,
                A=A,
                R=R,
                W0=W0,
                goals=goals,
                accounts=accounts,
                start_date=start_date,
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
            
            # Warm start: extend previous X for next iteration
            if result.X is not None:
                X_prev = np.vstack([result.X, result.X[-1:, :]])  # Repeat last row
        
        raise ValueError(
            f"No feasible solution found in T ∈ [{T_start}, {self.T_max}]. "
            f"Try increasing T_max or relaxing goal constraints."
        )