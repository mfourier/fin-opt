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
- GoalSeeker: Bilevel solver with linear/binary search and warm start

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
            Portfolio accounts (metadata only, W0 will be overridden)
        
        Notes
        -----
        FIX: No renormalization - trust SLSQP constraint satisfaction.
        Validates simplex constraint and returns False if violated.
        """
        # Import here to avoid circular dependency
        from .portfolio import Portfolio
        
        # Validate simplex constraint (tolerance matches SLSQP ftol)
        row_sums = X.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            max_deviation = np.abs(row_sums - 1.0).max()
            if max_deviation > 1e-6:
                return False
        
        # Create portfolio and simulate
        portfolio = Portfolio(accounts)
        result = portfolio.simulate(A=A, R=R, X=X, method="affine", W0_override=W0)
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
# SAA Optimizer (Smoothed Sigmoid Approximation) - FIXED VERSION
# ---------------------------------------------------------------------------

class SAAOptimizer(AllocationOptimizer):
    """
    Sample Average Approximation optimizer with smoothed sigmoid constraints.
    
    Mathematical Formulation
    ------------------------
    Objective:
        maximize  E[Σ_m W_T^m]  (expected total terminal wealth)
    
    Subject to:
        Intermediate goals: ℙ(W_t^m ≥ b) ≥ 1 - ε  (fixed t)
        Terminal goals: ℙ(W_T^m ≥ b) ≥ 1 - ε  (variable T)
        Σ_m x_t^m = 1
        x_t^m ≥ 0
    
    Smoothed Approximation
    ----------------------
    The discontinuous indicator 𝟙{W ≥ b} is replaced by sigmoid:
    
        𝟙{W_i ≥ b}  ≈  σ((W_i - b)/(τ·b))
        
        where σ(x) = 1/(1 + exp(-x))
    
    The approximation error is controlled via threshold buffer:
    
        b_opt = b · (1 + β)
        
    where β is computed to ensure target satisfaction rate at real threshold.
    
    Key Improvements
    ----------------
    1. Theoretically-grounded buffer: β = -τ·z*/(1 + τ·z*) where z* = logit(p_target)
    2. Wealth-optimal initialization: Grid search over simplex policies
    3. Non-trivial objective: Maximizes E[W_T] for better convergence
    4. No renormalization: Trust SLSQP constraint satisfaction
    
    Parameters
    ----------
    n_accounts : int
        Number of portfolio accounts
    tau : float, default 0.1
        Sigmoid temperature parameter (transition zone = ±τ·b)
    target_satisfaction : float, default 0.90
        Target satisfaction rate at real threshold (0 < p < 1)
    objective : ObjectiveType, default "terminal_wealth"
        Objective function specification
    objective_params : dict, optional
        Parameters for objective function
    account_names : List[str], optional
        Account name labels
    
    Examples
    --------
    >>> # Standard configuration (90% satisfaction)
    >>> opt = SAAOptimizer(n_accounts=2, tau=0.1, target_satisfaction=0.90)
    >>> 
    >>> # Conservative (95% satisfaction, tighter buffer)
    >>> opt = SAAOptimizer(n_accounts=2, tau=0.1, target_satisfaction=0.95)
    >>> 
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
        tau: float = 0.02,
        target_satisfaction: float = 0.90,
        objective: ObjectiveType = "terminal_wealth",
        objective_params: Optional[Dict[str, Any]] = None,
        account_names: Optional[List[str]] = None
    ):
        super().__init__(n_accounts, objective, objective_params, account_names)
        
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if not (0.5 < target_satisfaction < 1.0):
            raise ValueError(
                f"target_satisfaction must be ∈ (0.5, 1.0), got {target_satisfaction}"
            )
        
        self.tau = tau
        self.target_satisfaction = target_satisfaction
        
        # FIX 1: Compute theoretically-correct buffer
        # We want: σ((b - b_opt)/(τ·b_opt)) ≥ target_satisfaction
        # Using σ^(-1)(p) = ln(p/(1-p)) (logit function)
        z_target = np.log(target_satisfaction / (1 - target_satisfaction))
        
        # From z = (b - b_opt)/(τ·b_opt) ≥ z_target
        # → b_opt ≤ b / (1 + τ·z_target)
        # → buffer = (b_opt - b)/b = -τ·z_target / (1 + τ·z_target)
        self.threshold_buffer_factor = -tau * z_target / (1 + tau * z_target)
    
    def _initialize_smart(
        self,
        T: int,
        M: int,
        A: np.ndarray,
        R: np.ndarray,
        W0: np.ndarray,
        goal_set: GoalSet,
        accounts: List[Account]
    ) -> np.ndarray:
        """
        FIX 2: Smart initialization via grid search over simplex policies.
        
        Strategy
        --------
        1. Generate candidate constant policies (pure + mixed)
        2. Simulate wealth for each policy
        3. Select policy maximizing minimum safety margin across goals
        4. Return as time-constant policy X[t,:] = x* ∀t
        
        Returns
        -------
        X0 : np.ndarray, shape (T, M)
            Initial allocation policy
        
        Complexity
        ----------
        O(K · T · n_test) where K = M + M(M-1)/2 + 1 is number of test policies
        """
        from .portfolio import Portfolio
        
        # Subsample scenarios for speed
        n_test = min(A.shape[0], 100)
        A_test = A[:n_test, :]
        R_test = R[:n_test, :, :]
        
        # Generate test policies
        test_policies = []
        
        # Pure policies: 100% in each account
        for m in range(M):
            x = np.zeros(M)
            x[m] = 1.0
            test_policies.append(("pure", m, x))
        
        # Mixed policies: 50-50 splits
        if M >= 2:
            for m1 in range(M):
                for m2 in range(m1 + 1, M):
                    x = np.zeros(M)
                    x[m1] = 0.5
                    x[m2] = 0.5
                    test_policies.append(("mixed", (m1, m2), x))
        
        # Uniform policy
        test_policies.append(("uniform", None, np.ones(M) / M))
        
        # Evaluate each policy
        portfolio = Portfolio(accounts)
        best_policy = None
        best_score = -np.inf
        best_name = None
        
        for name, idx, x_const in test_policies:
            X_test = np.tile(x_const, (T, 1))
            
            result = portfolio.simulate(
                A=A_test, R=R_test, X=X_test, 
                method="affine", W0_override=W0
            )
            W = result["wealth"]  # (n_test, T+1, M)
            
            # Score: minimum safety margin across all goals
            # Safety margin = (median_wealth - threshold) / threshold
            min_margin = np.inf
            
            for goal in goal_set.terminal_goals:
                m = goal_set.get_account_index(goal)
                W_T_m = W[:, T, m]
                margin = (np.median(W_T_m) - goal.threshold) / goal.threshold
                min_margin = min(min_margin, margin)
            
            for goal in goal_set.intermediate_goals:
                m = goal_set.get_account_index(goal)
                t = goal_set.get_resolved_month(goal)
                W_t_m = W[:, t, m]
                margin = (np.median(W_t_m) - goal.threshold) / goal.threshold
                min_margin = min(min_margin, margin)
            
            if min_margin > best_score:
                best_score = min_margin
                best_policy = x_const
                best_name = f"{name}_{idx}" if idx is not None else name
        
        return np.tile(best_policy, (T, 1))
    
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
        Solve SAA problem with all fixes integrated.
        
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
            Warm start allocation (T, M). If None, uses smart initialization.
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
        
        # Sigmoid functions with numerical stability
        def sigmoid(z):
            """Numerically stable sigmoid with clipping."""
            z_safe = np.clip(z, -50, 50)
            return np.where(
                z_safe >= 0,
                1 / (1 + np.exp(-z_safe)),
                np.exp(z_safe) / (1 + np.exp(z_safe))
            )

        def sigmoid_prime(z):
            """Derivative of sigmoid with clipping."""
            z_safe = np.clip(z, -50, 50)
            s = sigmoid(z_safe)
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
                    contrib = A[:, s, None] * X[s, :]
                    W[:, t, :] += contrib * F[:, s, t, :]
            
            return W
        
        # FIX 5: Non-trivial objective with analytical gradient
        def objective(X_flat):
            """
            Maximize expected terminal wealth: E[Σ_m W_T^m]
            
            Returns NEGATIVE for scipy.minimize (converts max to min).
            """
            W = compute_wealth_all(X_flat)
            return -W[:, T, :].sum(axis=1).mean()

        def objective_grad(X_flat):
            """
            Gradient: ∂f/∂x_{s,m} = -E[A_s · F_{s,T}^m]
            """
            grad = np.zeros(T * M)
            
            for s in range(T):
                for m in range(M):
                    idx = s * M + m
                    grad[idx] = -(A[:, s] * F[:, s, T, m]).mean()
            
            return grad
        
        # Constraint functions (unchanged, mathematically correct)
        def constraint_intermediate_goal(X_flat, goal: IntermediateGoal, m: int, t: int, threshold: float):
            W = compute_wealth_all(X_flat)
            W_t_m = W[:, t, m]
            tau_scaled = self.tau * threshold
            z = (W_t_m - threshold) / tau_scaled
            satisfaction = sigmoid(z).mean()
            return satisfaction - (1 - goal.epsilon)

        def constraint_intermediate_goal_grad(X_flat, goal: IntermediateGoal, m: int, t: int, threshold: float):
            W = compute_wealth_all(X_flat)
            W_t_m = W[:, t, m]
            tau_scaled = self.tau * threshold
            z = (W_t_m - threshold) / tau_scaled
            sigmoid_deriv = sigmoid_prime(z) / tau_scaled
            
            grad = np.zeros(T * M)
            for s in range(t):
                for m_idx in range(M):
                    idx = s * M + m_idx
                    if m_idx == m:
                        grad[idx] = (sigmoid_deriv * A[:, s] * F[:, s, t, m]).mean()
            return grad

        def constraint_terminal_goal(X_flat, goal: TerminalGoal, m: int, threshold: float):
            W = compute_wealth_all(X_flat)
            W_T_m = W[:, T, m]
            tau_scaled = self.tau * threshold
            z = (W_T_m - threshold) / tau_scaled
            satisfaction = sigmoid(z).mean()
            return satisfaction - (1 - goal.epsilon)

        def constraint_terminal_goal_grad(X_flat, goal: TerminalGoal, m: int, threshold: float):
            W = compute_wealth_all(X_flat)
            W_T_m = W[:, T, m]
            tau_scaled = self.tau * threshold
            z = (W_T_m - threshold) / tau_scaled
            sigmoid_deriv = sigmoid_prime(z) / tau_scaled
            
            grad = np.zeros(T * M)
            for s in range(T):
                for m_idx in range(M):
                    idx = s * M + m_idx
                    if m_idx == m:
                        grad[idx] = (sigmoid_deriv * A[:, s] * F[:, s, T, m]).mean()
            return grad
        
        def constraint_simplex(X_flat):
            X = X_flat.reshape(T, M)
            return X.sum(axis=1) - 1.0
        
        def constraint_simplex_jac(X_flat):
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

        # FIX 1: Theoretically-correct buffer (computed in __init__)
        buffer = self.threshold_buffer_factor
        
        # Intermediate goal constraints
        for goal in goal_set.intermediate_goals:
            m = goal_set.get_account_index(goal)
            t = goal_set.get_resolved_month(goal)
            threshold_opt = goal.threshold * (1 + buffer)
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda X, g=goal, m_=m, t_=t, thresh_=threshold_opt: 
                    constraint_intermediate_goal(X, g, m_, t_, thresh_),
                'jac': lambda X, g=goal, m_=m, t_=t, thresh_=threshold_opt:
                    constraint_intermediate_goal_grad(X, g, m_, t_, thresh_)
            })

        # Terminal goal constraints
        for goal in goal_set.terminal_goals:
            m = goal_set.get_account_index(goal)
            threshold_opt = goal.threshold * (1 + buffer)
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda X, g=goal, m_=m, thresh_=threshold_opt: 
                    constraint_terminal_goal(X, g, m_, thresh_),
                'jac': lambda X, g=goal, m_=m, thresh_=threshold_opt:
                    constraint_terminal_goal_grad(X, g, m_, thresh_)
            })
        
        # FIX 2: Smart initialization
        if X_init is None:
            X0 = self._initialize_smart(T, M, A, R, W0, goal_set, accounts)
        else:
            X0 = X_init.copy()
        
        X0_flat = X0.flatten()
        
        # Bounds: x_t^m ∈ [0, 1]
        bounds = [(0.0, 1.0) for _ in range(T * M)]
        
        # Solve with non-trivial objective
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

        # Diagnostics
        if verbose:
            W_final = compute_wealth_all(X_star.flatten())
            print("\n[Diagnostics] Terminal wealth per account:")
            for goal in goal_set.terminal_goals:
                m = goal_set.get_account_index(goal)
                W_T_m = W_final[:, T, m]
                violations = W_T_m < goal.threshold
                violation_rate = violations.mean()
                
                tau_scaled = self.tau * goal.threshold
                smoothed_sat = sigmoid((W_T_m - goal.threshold) / tau_scaled).mean()
                
                print(f"  Account {m} ({goal.account}):")
                print(f"    Threshold:      {goal.threshold:>12,.0f}")
                print(f"    Mean wealth:    {W_T_m.mean():>12,.0f}")
                print(f"    P10/P50/P90:    {np.percentile(W_T_m, [10,50,90])}")
                print(f"    Violation rate: {violation_rate:.2%} (max: {goal.epsilon:.2%})")
                print(f"    Smoothed sat.:  {smoothed_sat:.4f} (target: {1-goal.epsilon:.4f})")

        # FIX 4: No renormalization - check feasibility with exact SAA
        feasible = self._check_feasibility(X_star, A, R, W0, accounts, goal_set)
        
        # Diagnostics
        diagnostics = {
            'convergence_status': result.message,
            'success': result.success,
            'final_gradient_norm': np.linalg.norm(result.jac) if result.jac is not None else None,
            'buffer_factor': buffer,
            'target_satisfaction': self.target_satisfaction,
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
    
    Supports linear and binary search strategies with warm start.
    
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
    >>> optimizer = SAAOptimizer(n_accounts=2, tau=0.1, target_satisfaction=0.90)
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
        start_date,
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
        
        # Construct GoalSet with accounts
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
            print()
        
        # Dispatch to search method
        if search_method == "linear":
            return self._linear_search(
                T_start, goal_set, goals, A_generator, R_generator, 
                W0, accounts, start_date, n_sims, seed, **solver_kwargs
            )
        elif search_method == "binary":
            return self._binary_search(
                T_start, goal_set, goals, A_generator, R_generator,
                W0, accounts, start_date, n_sims, seed, **solver_kwargs
            )
        else:
            raise ValueError(f"Unknown search_method: {search_method}")

    def _linear_search(
        self,
        T_start: int,
        goal_set: GoalSet,
        goals: List,
        A_generator: Callable,
        R_generator: Callable,
        W0: np.ndarray,
        accounts: List[Account],
        start_date,
        n_sims: int,
        seed: Optional[int],
        **solver_kwargs
    ) -> OptimizationResult:
        """Linear search T = T_start, T_start+1, ... (original algorithm)."""
        X_prev = None
        
        for T in range(T_start, self.T_max + 1):
            if self.verbose:
                print(f"[Iter {T - T_start + 1}] Testing T={T}...")
            
            A = A_generator(T, n_sims, seed)
            R = R_generator(T, n_sims, None if seed is None else seed + 1)
            
            result = self.optimizer.solve(
                T=T, A=A, R=R, W0=W0, goals=goals,
                accounts=accounts, start_date=start_date,
                goal_set=goal_set, X_init=X_prev, **solver_kwargs
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
        goals: List,
        A_generator: Callable,
        R_generator: Callable,
        W0: np.ndarray,
        accounts: List[Account],
        start_date,
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
            
            # Solve at mid
            result = self.optimizer.solve(
                T=mid, A=A, R=R, W0=W0, goals=goals,
                accounts=accounts, start_date=start_date,
                goal_set=goal_set, X_init=X_init, **solver_kwargs
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
            T=left, A=A, R=R, W0=W0, goals=goals,
            accounts=accounts, start_date=start_date,
            goal_set=goal_set, X_init=None, **solver_kwargs
        )
        
        if result.feasible:
            if self.verbose:
                print(f"=== Optimal: T*={left} ===\n")
            return result
        
        raise ValueError(
            f"Binary search failed: T={left} infeasible. "
            f"Monotonicity assumption may be violated."
        )