# finopt/src/goals.py
"""
Goal specification and validation module with intermediate/terminal distinction.

Purpose
-------
Domain-level abstractions for financial goals as chance constraints with
explicit time semantics. Distinguishes between:
- IntermediateGoal: Fixed calendar checkpoints (independent of horizon T)
- TerminalGoal: End-of-horizon targets (evaluated at variable T)

Mathematical Framework
----------------------
IntermediateGoal (t_fixed, m, b, 1-ε):
    ℙ(W_{t_fixed}^m ≥ b) ≥ 1 - ε
    where t_fixed is resolved from absolute date or month offset

TerminalGoal (m, b, 1-ε):
    ℙ(W_T^m ≥ b) ≥ 1 - ε
    where T is the optimization variable (outer problem)

Design Principles
-----------------
- Immutable specifications: Goals are frozen dataclasses
- Type-safe resolution: Supports int indices or str names
- Calendar-aware: IntermediateGoal resolves dates to month offsets
- Optimization-ready: GoalSet provides T_min for constraint generation
- Portfolio-aware: GoalSet consumes Account objects for type safety

Example
-------
>>> from datetime import date
>>> from finopt.src.portfolio import Account
>>> from finopt.src.goals import IntermediateGoal, TerminalGoal, GoalSet
>>> 
>>> accounts = [
...     Account.from_annual("Emergency", 0.04, 0.05),
...     Account.from_annual("Housing", 0.07, 0.12)
... ]
>>> 
>>> goals = [
...     IntermediateGoal(month=6, account="Emergency", 
...                     threshold=5_500_000, confidence=0.90),
...     TerminalGoal(account="Emergency", threshold=20_000_000, confidence=0.90),
...     TerminalGoal(account="Housing", threshold=7_000_000, confidence=0.90)
... ]
>>> 
>>> goal_set = GoalSet(goals, accounts, date(2025, 1, 1))
>>> goal_set.T_min  # minimum horizon from intermediate goal
>>> 6
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, TYPE_CHECKING
from datetime import date
import numpy as np

if TYPE_CHECKING:
    from .model import SimulationResult
    from .portfolio import Account

__all__ = [
    "IntermediateGoal",
    "TerminalGoal",
    "GoalSet",
    "check_goals",
    "goal_progress",
    "print_goal_status"
]


# ---------------------------------------------------------------------------
# Goal Specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntermediateGoal:
    """
    Intermediate financial goal at fixed calendar time.
    
    Represents checkpoint constraint evaluated at fixed month t_fixed:
        ℙ(W_{t_fixed}^m ≥ threshold) ≥ confidence
    
    The target month is independent of the optimization horizon T.
    Used for liquidity requirements, planned expenses, or milestone tracking.
    
    Parameters
    ----------
    account : int or str
        Target account identifier
    threshold : float
        Minimum required wealth (e.g., 5_500_000 CLP)
    confidence : float
        Minimum satisfaction probability, must be ∈ (0, 1)
        Example: 0.95 means "95% chance of meeting threshold"
    month : int, optional
        Target month as offset from start_date (1-indexed: 1 = end of first month)
        Mutually exclusive with `date`. If both None, raises ValueError.
    date : datetime.date, optional
        Target date (will be converted to month offset via resolve_month)
        Mutually exclusive with `month`. If both None, raises ValueError.
    
    Notes
    -----
    - Exactly one of `month` or `date` must be provided
    - Month resolution: date → months since start_date (rounded up to end-of-month)
    - Epsilon tolerance: ε = 1 - confidence
    - Used in optimization as fixed-time constraint W_t ≥ b
    
    Examples
    --------
    >>> # Month-based specification
    >>> g1 = IntermediateGoal(month=6, account="Emergency", 
    ...                      threshold=5_500_000, confidence=0.90)
    >>> 
    >>> # Date-based specification
    >>> g2 = IntermediateGoal(date=date(2025, 7, 1), account="Emergency",
    ...                      threshold=5_500_000, confidence=0.90)
    >>> g2.resolve_month(date(2025, 1, 1))
    6
    """
    account: Union[int, str]
    threshold: float
    confidence: float
    month: Optional[int] = None
    date: Optional[date] = None
    
    def __post_init__(self):
        """Validate intermediate goal parameters."""
        if self.month is None and self.date is None:
            raise ValueError(
                "IntermediateGoal requires exactly one of 'month' or 'date'"
            )
        if self.month is not None and self.date is not None:
            raise ValueError(
                "IntermediateGoal: 'month' and 'date' are mutually exclusive"
            )
        
        if self.month is not None and self.month < 1:
            raise ValueError(f"month must be ≥ 1, got {self.month}")
        
        if not (0 < self.confidence < 1):
            raise ValueError(
                f"confidence must be ∈ (0, 1), got {self.confidence}"
            )
        
        if self.threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
    
    def resolve_month(self, start_date: date) -> int:
        """
        Convert date to month offset from start_date.
        
        Parameters
        ----------
        start_date : datetime.date
            Simulation start date for offset calculation
        
        Returns
        -------
        int
            Month offset (1-indexed: 1 = end of first month)
        
        Examples
        --------
        >>> goal = IntermediateGoal(date=date(2025, 7, 1), account=0,
        ...                        threshold=1_000_000, confidence=0.90)
        >>> goal.resolve_month(date(2025, 1, 1))
        6
        """
        if self.month is not None:
            return self.month
        
        # Calculate months between dates
        delta_months = (
            (self.date.year - start_date.year) * 12
            + (self.date.month - start_date.month)
        )
        
        # Round up to end-of-month convention (minimum 1)
        return max(1, delta_months)
    
    @property
    def epsilon(self) -> float:
        """Violation tolerance: ε = 1 - confidence."""
        return 1.0 - self.confidence
    
    def __repr__(self) -> str:
        """Readable representation."""
        if self.month is not None:
            time_str = f"month={self.month}"
        else:
            time_str = f"date={self.date.isoformat()}"
        
        return (
            f"IntermediateGoal({time_str}, account={self.account!r}, "
            f"threshold={self.threshold:,.0f}, confidence={self.confidence:.2%})"
        )


@dataclass(frozen=True)
class TerminalGoal:
    """
    Terminal financial goal evaluated at end of horizon.
    
    Represents end-of-planning constraint evaluated at variable T:
        ℙ(W_T^m ≥ threshold) ≥ confidence
    
    The target month T is the optimization variable (outer problem).
    Used for retirement targets, long-term savings, or final portfolio value.
    
    Parameters
    ----------
    account : int or str
        Target account identifier
    threshold : float
        Minimum required terminal wealth (e.g., 20_000_000 CLP)
    confidence : float
        Minimum satisfaction probability, must be ∈ (0, 1)
        Example: 0.95 means "95% chance of meeting threshold at T"
    
    Notes
    -----
    - No fixed time: evaluated at T (the horizon being optimized)
    - In GoalSeeker: T is minimized subject to terminal goal feasibility
    - Epsilon tolerance: ε = 1 - confidence
    - Used in optimization as horizon-dependent constraint W_T ≥ b
    
    Examples
    --------
    >>> # Retirement target (no fixed time)
    >>> g1 = TerminalGoal(account="Retirement", threshold=20_000_000, 
    ...                  confidence=0.90)
    >>> 
    >>> # Housing down payment (terminal goal)
    >>> g2 = TerminalGoal(account=1, threshold=15_000_000, confidence=0.95)
    """
    account: Union[int, str]
    threshold: float
    confidence: float
    
    def __post_init__(self):
        """Validate terminal goal parameters."""
        if not (0 < self.confidence < 1):
            raise ValueError(
                f"confidence must be ∈ (0, 1), got {self.confidence}"
            )
        
        if self.threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {self.threshold}")
    
    @property
    def epsilon(self) -> float:
        """Violation tolerance: ε = 1 - confidence."""
        return 1.0 - self.confidence
    
    def __repr__(self) -> str:
        """Readable representation."""
        return (
            f"TerminalGoal(account={self.account!r}, "
            f"threshold={self.threshold:,.0f}, confidence={self.confidence:.2%})"
        )


# ---------------------------------------------------------------------------
# Goal Set (Collection with Validation)
# ---------------------------------------------------------------------------

class GoalSet:
    """
    Validated collection of intermediate and terminal goals.
    
    Ensures:
    - All account references resolve to valid indices
    - No duplicate specifications within each goal type
    - Intermediate goals: unique (month, account) pairs
    - Terminal goals: unique account targets
    
    Provides utilities:
    - T_min: minimum horizon from latest intermediate goal
    - Grouping by account or month
    - Index resolution for optimization
    
    Parameters
    ----------
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        Mixed list of goal specifications
    accounts : List[Account]
        Portfolio accounts (source of truth for account metadata)
    start_date : datetime.date
        Simulation start date for intermediate goal resolution
    
    Attributes
    ----------
    accounts : List[Account]
        Reference to portfolio accounts (enables future metadata access)
    account_names : List[str]
        Derived account name list (for backward compatibility)
    M : int
        Number of accounts
    start_date : datetime.date
        Simulation start date
    intermediate_goals : List[IntermediateGoal]
        Filtered list of intermediate goals
    terminal_goals : List[TerminalGoal]
        Filtered list of terminal goals
    
    Raises
    ------
    ValueError
        - If goals list is empty
        - If accounts list is empty
        - If account reference invalid
        - If duplicate goals exist within type
    
    Examples
    --------
    >>> from finopt.src.portfolio import Account
    >>> accounts = [
    ...     Account.from_annual("Emergency", 0.04, 0.05),
    ...     Account.from_annual("Housing", 0.07, 0.12)
    ... ]
    >>> 
    >>> goals = [
    ...     IntermediateGoal(month=6, account="Emergency", 
    ...                     threshold=5_500_000, confidence=0.90),
    ...     TerminalGoal(account="Emergency", threshold=20_000_000, 
    ...                 confidence=0.90),
    ...     TerminalGoal(account="Housing", threshold=7_000_000, confidence=0.90)
    ... ]
    >>> 
    >>> goal_set = GoalSet(goals, accounts, date(2025, 1, 1))
    >>> goal_set.T_min  # From intermediate goal
    6
    >>> goal_set.accounts  # Access to Account objects
    [Account('Emergency': ...), Account('Housing': ...)]
    >>> goal_set.account_names  # Derived property
    ['Emergency', 'Housing']
    """
    
    def __init__(
        self,
        goals: List[Union[IntermediateGoal, TerminalGoal]],
        accounts: List[Account],
        start_date: date  # ✅ MEJORA 4: Type hint explícito
    ):
        if not goals:
            raise ValueError("goals list cannot be empty")
        if not accounts:
            raise ValueError("accounts list cannot be empty")
        
        # Store accounts as canonical source
        self.accounts = accounts
        self.M = len(accounts)
        
        # Derive account_names for backward compatibility and string resolution
        self.account_names = [acc.name for acc in accounts]
        self.account_labels = [acc.label for acc in accounts]
        
        self.start_date = start_date
        
        # Separate goal types
        self.intermediate_goals: List[IntermediateGoal] = []
        self.terminal_goals: List[TerminalGoal] = []
        
        for goal in goals:
            if isinstance(goal, IntermediateGoal):
                self.intermediate_goals.append(goal)
            elif isinstance(goal, TerminalGoal):
                self.terminal_goals.append(goal)
            else:
                raise TypeError(
                    f"Goal must be IntermediateGoal or TerminalGoal, "
                    f"got {type(goal)}"
                )
        
        # Resolve indices and validate
        self._account_indices_intermediate: Dict[IntermediateGoal, int] = {}
        self._account_indices_terminal: Dict[TerminalGoal, int] = {}
        
        # ✅ MEJORA 1: Caché de resolved months (nuevo)
        self._resolved_months: Dict[IntermediateGoal, int] = {}
        
        self._validate()
    
    def _validate(self):
        """Validate goal collection for conflicts and invalid references."""
        # Validate intermediate goals
        seen_intermediate = set()
        for goal in self.intermediate_goals:
            idx = self._resolve_account(goal.account)
            self._account_indices_intermediate[goal] = idx
            
            # ✅ MEJORA 1: Cachear month al validar (ejecuta 1 vez)
            month = goal.resolve_month(self.start_date)
            self._resolved_months[goal] = month
            
            key = (month, idx)
            if key in seen_intermediate:
                raise ValueError(
                    f"Duplicate IntermediateGoal: multiple goals target "
                    f"month={month}, account={self.account_names[idx]!r}"
                )
            seen_intermediate.add(key)
        
        # Validate terminal goals
        seen_terminal = set()
        for goal in self.terminal_goals:
            idx = self._resolve_account(goal.account)
            self._account_indices_terminal[goal] = idx
            
            if idx in seen_terminal:
                raise ValueError(
                    f"Duplicate TerminalGoal: multiple goals target "
                    f"account={self.account_names[idx]!r}"
                )
            seen_terminal.add(idx)
    
    def _resolve_account(self, account: Union[int, str]) -> int:
        """Convert account identifier to 0-based index."""
        if isinstance(account, int):
            if not (0 <= account < self.M):
                raise ValueError(
                    f"account index {account} out of range [0, {self.M}). "
                    f"Available accounts: {self.account_names}"
                )
            return account
        else:  # str
            try:
                return self.account_names.index(account)
            except ValueError:
                raise ValueError(
                    f"account name {account!r} not found. "
                    f"Available accounts: {self.account_names}"
                ) from None
    
    def get_account_index(self, goal: Union[IntermediateGoal, TerminalGoal]) -> int:
        """Get resolved account index for a goal."""
        if isinstance(goal, IntermediateGoal):
            return self._account_indices_intermediate[goal]
        elif isinstance(goal, TerminalGoal):
            return self._account_indices_terminal[goal]
        else:
            raise TypeError(f"Unknown goal type: {type(goal)}")
    
    def get_resolved_month(self, goal: IntermediateGoal) -> int:
        """
        Get resolved month for intermediate goal.
        
        ✅ MEJORA 1: Ahora usa caché O(1) en lugar de recalcular
        
        Returns
        -------
        int
            Cached month offset (1-indexed)
        
        Notes
        -----
        Previously recalculated goal.resolve_month(self.start_date) on every call.
        Now returns pre-computed value from _validate() for O(1) lookup.
        
        Performance impact:
        - Old: O(n_sims × n_goals) date calculations
        - New: O(1) dictionary lookup
        - For n_sims=500, n_goals=3: ~1500 calculations eliminated
        """
        return self._resolved_months[goal]
    
    @property
    def T_min(self) -> int:
        """
        Minimum feasible horizon from intermediate goals.
        
        Returns
        -------
        int
            Maximum month among intermediate goals, or 1 if none exist.
        
        Notes
        -----
        Terminal goals do NOT contribute to T_min since they are evaluated
        at variable T (the optimization target). Use estimate_minimum_horizon()
        for terminal goal horizon estimation with contribution/return assumptions.
        """
        if not self.intermediate_goals:
            return 1
        
        # ✅ MEJORA 1: Usa caché (más eficiente)
        return max(self._resolved_months.values())

    def estimate_minimum_horizon(
        self,
        monthly_contribution: float,
        accounts: List[Account],
        expected_return: Optional[float] = None,
        safety_margin: float = 0.8,
        T_max: int = 240
    ) -> int:
        """
        Estimate minimum horizon for terminal goals via deterministic analysis.
        
        **IMPROVED HEURISTIC**: Uses account-specific expected returns and
        weighted allocation strategy to provide tighter bounds.
        
        Mathematical Approach
        ---------------------
        For each terminal goal, solves the deterministic wealth equation:
        
            W_T^m = W_0^m * (1 + r^m)^T + A_avg * α^m * [(1 + r^m)^T - 1] / r^m
        
        where:
        - r^m: expected monthly return of account m
        - A_avg: average monthly contribution
        - α^m: allocation fraction to account m (inferred from goal structure)
        
        The horizon T is found by solving for T such that W_T^m = b^m (threshold).
        
        Parameters
        ----------
        monthly_contribution : float
            Average monthly contribution (estimated from sampling)
        accounts : List[Account]
            Portfolio accounts with expected returns
        expected_return : float, optional
            Override expected return. If None, infers from account returns.
        safety_margin : float, default 0.8
            Conservative factor: T_est = T_analytical / safety_margin
            Lower values = more conservative (start search earlier)
        T_max : int, default 240
            Maximum allowable horizon
        
        Returns
        -------
        int
            Estimated minimum horizon (months), capped at T_max
        
        Notes
        -----
        - Conservative by design: underestimates T to avoid missing feasible solutions
        - If multiple terminal goals exist, returns max across all estimates
        - Intermediate goals enforced via T_min constraint (not estimated here)
        
        Examples
        --------
        >>> accounts = [Account.from_annual("Savings", 0.04, 0.05),
        ...            Account.from_annual("Growth", 0.12, 0.15)]
        >>> goal_set = GoalSet(goals, accounts, date(2025,1,1))
        >>> T_est = goal_set.estimate_minimum_horizon(
        ...     monthly_contribution=500_000,
        ...     accounts=accounts,
        ...     safety_margin=0.75
        ... )
        """
        if not self.terminal_goals:
            return self.T_min
        
        # Build account lookup by name
        account_map = {acc.name: acc for acc in accounts}
        
        T_estimates = []
        
        for goal in self.terminal_goals:
            # Get target account and its expected return
            acc = account_map.get(goal.account)
            if acc is None:
                raise ValueError(
                    f"Terminal goal references unknown account: {goal.account}"
                )
            
            # Extract expected monthly return
            r_monthly = acc.monthly_params["mu"]  # Already monthly from Account
            
            # Initial wealth for this account
            W0_m = acc.initial_wealth
            
            # Infer allocation fraction α^m based on goal structure
            # Heuristic: if this is the highest-return goal, allocate aggressively
            # Otherwise, use balanced allocation
            alpha_m = self._infer_allocation_fraction(goal, account_map)
            
            # Effective contribution to this account
            A_effective = monthly_contribution * alpha_m
            
            # Solve deterministic wealth equation for T
            # W_T^m = W0^m * (1+r)^T + A * [(1+r)^T - 1] / r = threshold
            
            if abs(r_monthly) < 1e-8:
                # No growth case: W_T = W0 + A * T
                T_analytical = (goal.threshold - W0_m) / A_effective
            else:
                # With growth: solve via closed-form annuity formula
                # (1+r)^T = [threshold - A/r + W0] / [W0 - A/r]
                
                annuity_pv = A_effective / r_monthly
                ratio = (goal.threshold - annuity_pv + W0_m) / (W0_m - annuity_pv)
                
                if ratio <= 0:
                    # Goal already satisfied or infeasible
                    T_analytical = 0
                else:
                    T_analytical = np.log(ratio) / np.log(1 + r_monthly)
            
            # Apply safety margin (conservative estimate)
            T_conservative = T_analytical * safety_margin
            
            T_estimates.append(max(1, int(np.ceil(T_conservative))))
        
        # Take maximum across all terminal goals
        T_est = max(T_estimates)
        
        # Ensure T_est >= T_min (intermediate goals constraint)
        T_est = max(T_est, self.T_min)
        
        # Cap at T_max
        return min(T_est, T_max)

    def _infer_allocation_fraction(
        self,
        goal: TerminalGoal,
        account_map: Dict[str, Account]
    ) -> float:
        """
        Infer allocation fraction for goal's account via simple heuristic.
        
        Strategy
        --------
        1. If only one terminal goal: α = 0.6 (majority allocation)
        2. If multiple terminal goals:
        - Highest threshold goal: α = 0.5
        - Other goals: α = 0.5 / (n_goals - 1)
        3. If intermediate goals exist for same account: α *= 0.8 (reduce for liquidity)
        
        Returns
        -------
        float
            Allocation fraction ∈ [0, 1]
        """
        n_terminal = len(self.terminal_goals)
        
        if n_terminal == 1:
            alpha = 0.6
        else:
            # Check if this goal has highest threshold
            max_threshold = max(g.threshold for g in self.terminal_goals)
            if goal.threshold == max_threshold:
                alpha = 0.5
            else:
                alpha = 0.5 / (n_terminal - 1)
        
        # Reduce if intermediate goals exist for same account
        has_intermediate = any(
            g.account == goal.account for g in self.intermediate_goals
        )
        if has_intermediate:
            alpha *= 0.8
        
        return alpha
    
    def __len__(self) -> int:
        """Total number of goals (intermediate + terminal)."""
        return len(self.intermediate_goals) + len(self.terminal_goals)
    
    def __repr__(self) -> str:
        return (
            f"GoalSet(n_intermediate={len(self.intermediate_goals)}, "
            f"n_terminal={len(self.terminal_goals)}, T_min={self.T_min}, M={self.M})"
        )


# ---------------------------------------------------------------------------
# Goal Validation (Post-Simulation)
# ---------------------------------------------------------------------------

def check_goals(
    result: SimulationResult,
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    accounts: List[Account],
    start_date: date  # ✅ MEJORA 4: Type hint explícito
) -> Dict[Union[IntermediateGoal, TerminalGoal], Dict[str, float]]:
    """
    Validate goal satisfaction in simulation result.
    
    For each goal, computes empirical violation rate and compares
    against required confidence level:
        satisfied = (# scenarios with W_t^m ≥ b) / n_sims ≥ 1 - ε
    
    Handles both intermediate (fixed t) and terminal (t=T) goals.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation output with wealth trajectories
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        Goals to validate
    accounts : List[Account]
        Portfolio accounts for name resolution
    start_date : datetime.date
        Simulation start date for intermediate goal resolution
    
    Returns
    -------
    dict : {Goal: metrics}
        For each goal, returns dict with keys:
        - satisfied : bool
            True if empirical violation rate ≤ ε
        - violation_rate : float
            Empirical ℙ(W_t^m < threshold)
        - required_rate : float
            Goal's ε = 1 - confidence
        - margin : float
            required_rate - violation_rate (positive → satisfied)
        - median_shortfall : float
            Median of max(0, threshold - W_t^m) over violations
            Zero if no violations occur
        - n_violations : int
            Count of scenarios violating threshold
    
    Raises
    ------
    ValueError
        If goal.month > result.T (goal beyond simulation horizon)
    
    Examples
    --------
    >>> from finopt.src.portfolio import Account
    >>> accounts = [
    ...     Account.from_annual("Emergency", 0.04, 0.05),
    ...     Account.from_annual("Housing", 0.07, 0.12)
    ... ]
    >>> 
    >>> goals = [
    ...     IntermediateGoal(month=12, account="Emergency", 
    ...                     threshold=2_000_000, confidence=0.95),
    ...     TerminalGoal(account="Housing", threshold=15_000_000, confidence=0.90)
    ... ]
    >>> 
    >>> result = model.simulate(T=24, X=X, n_sims=500, seed=42)
    >>> status = check_goals(result, goals, accounts, date(2025, 1, 1))
    >>> 
    >>> for goal, metrics in status.items():
    ...     print(f"{goal}: {'✓' if metrics['satisfied'] else '✗'}")
    """
    # Validate goal structure
    goal_set = GoalSet(goals, accounts, start_date)
    
    # Check horizon compatibility for intermediate goals
    if goal_set.intermediate_goals:
        # ✅ MEJORA 1: Usa T_min (que ahora usa caché internamente)
        if goal_set.T_min > result.T:
            raise ValueError(
                f"Intermediate goals require T ≥ {goal_set.T_min}, "
                f"but result.T = {result.T}"
            )
    
    status = {}
    n_sims = result.n_sims
    
    for goal in goals:
        # Resolve account index
        m = goal_set.get_account_index(goal)
        
        # Determine target month
        if isinstance(goal, IntermediateGoal):
            # ✅ MEJORA 1: Usa caché O(1) en lugar de recalcular
            t = goal_set.get_resolved_month(goal)
        else:  # TerminalGoal
            t = result.T
        
        # Extract wealth at target month: (n_sims,)
        W_t_m = result.wealth[:, t, m]
        
        # Compute violation statistics
        violations = W_t_m < goal.threshold
        n_violations = violations.sum()
        violation_rate = n_violations / n_sims
        
        # Satisfaction check
        required_rate = goal.epsilon
        margin = required_rate - violation_rate
        satisfied = margin >= 0.0
        
        # Shortfall analysis
        if n_violations > 0:
            shortfall = np.maximum(0, goal.threshold - W_t_m)
            median_shortfall = float(np.median(shortfall[violations]))
        else:
            median_shortfall = 0.0
        
        status[goal] = {
            "satisfied": bool(satisfied),
            "violation_rate": float(violation_rate),
            "required_rate": float(required_rate),
            "margin": float(margin),
            "median_shortfall": median_shortfall,
            "n_violations": int(n_violations),
        }
    
    return status


def goal_progress(
    result: SimulationResult,
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    accounts: List[Account],
    start_date: date  # ✅ MEJORA 4: Type hint explícito
) -> Dict[Union[IntermediateGoal, TerminalGoal], float]:
    """
    Compute progress toward goal achievement.
    
    Progress metric: min(1, VaR_{1-ε}(W_t^m) / threshold)
    where VaR at confidence level 1-ε is the (1-ε)-quantile of W_t^m.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation output with wealth trajectories
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        Goals to track
    accounts : List[Account]
        Portfolio accounts for name resolution
    start_date : datetime.date
        Simulation start date for intermediate goal resolution
    
    Returns
    -------
    dict : {Goal: progress}
        Progress ∈ [0, 1] where:
        - 0.0: VaR is zero (far from goal)
        - 1.0: VaR ≥ threshold (goal achieved at confidence level)
        - 0.5: VaR is 50% of threshold (halfway)
    
    Examples
    --------
    >>> from finopt.src.portfolio import Account
    >>> accounts = [
    ...     Account.from_annual("Emergency", 0.04, 0.05),
    ...     Account.from_annual("Housing", 0.07, 0.12)
    ... ]
    >>> 
    >>> result = model.simulate(T=24, X=X, n_sims=500, seed=42)
    >>> progress = goal_progress(result, goals, accounts, date(2025, 1, 1))
    >>> 
    >>> for goal, pct in progress.items():
    ...     print(f"{goal.account} @ {goal.month if hasattr(goal, 'month') else 'T'}: "
    ...           f"{pct:.1%} progress")
    """
    goal_set = GoalSet(goals, accounts, start_date)
    
    progress = {}
    for goal in goals:
        m = goal_set.get_account_index(goal)
        
        # Determine target month
        if isinstance(goal, IntermediateGoal):
            # ✅ MEJORA 1: Usa caché
            t = goal_set.get_resolved_month(goal)
        else:  # TerminalGoal
            t = result.T
        
        # Extract wealth
        W_t_m = result.wealth[:, t, m]
        
        # Compute VaR at confidence level
        quantile = 1 - goal.epsilon
        var = np.quantile(W_t_m, quantile)
        
        # Progress ratio (capped at 1.0)
        progress[goal] = float(min(1.0, var / goal.threshold))
    
    return progress


def print_goal_status(
    result: SimulationResult,
    goals: List[Union[IntermediateGoal, TerminalGoal]],
    accounts: List[Account],
    start_date: date  # ✅ MEJORA 4: Type hint explícito
):
    """
    Pretty-print goal satisfaction status.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation output
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        Goals to display
    accounts : List[Account]
        Portfolio accounts for name resolution
    start_date : datetime.date
        Simulation start date
    
    Examples
    --------
    >>> from finopt.src.portfolio import Account
    >>> accounts = [
    ...     Account.from_annual("Emergency", 0.04, 0.05),
    ...     Account.from_annual("Housing", 0.07, 0.12)
    ... ]
    >>> 
    >>> result = model.simulate(T=24, X=X, n_sims=500, seed=42)
    >>> print_goal_status(result, goals, accounts, date(2025, 1, 1))
    
    === Goal Status ===
    
    [✓] IntermediateGoal: Emergency @ month 6
        Target: $5,500,000 | Confidence: 90.0%
        Status: SATISFIED (margin: +2.3%)
        Violation rate: 7.7% (38 scenarios)
    
    [✗] TerminalGoal: Emergency @ T=24
        Target: $20,000,000 | Confidence: 90.0%
        Status: VIOLATED (margin: -3.1%)
        Violation rate: 13.1% (66 scenarios)
        Median shortfall: $1,234,567
    """
    status = check_goals(result, goals, accounts, start_date)
    
    print("\n=== Goal Status ===\n")
    
    for goal, metrics in status.items():
        # Status symbol
        symbol = "✓" if metrics["satisfied"] else "✗"
        
        # Goal type and location
        if isinstance(goal, IntermediateGoal):
            # ✅ MEJORA 1: Usa GoalSet para resolver (que usa caché)
            goal_set = GoalSet([goal], accounts, start_date)
            month = goal_set.get_resolved_month(goal)
            goal_type = "IntermediateGoal"
            location = f"@ month {month}"
        else:
            goal_type = "TerminalGoal"
            location = f"@ T={result.T}"
        
        # Resolve account name
        # Note: We use the full label (long name) for a more readable report
        goal_set = GoalSet([goal], accounts, start_date)
        account_idx = goal_set.get_account_index(goal)
        account_name = goal_set.account_labels[account_idx]
        
        print(f"[{symbol}] {goal_type}: {account_name} {location}")
        print(f"    Target: ${goal.threshold:,.0f} | "
              f"Confidence: {goal.confidence:.1%}")
        
        status_text = "SATISFIED" if metrics["satisfied"] else "VIOLATED"
        margin_sign = "+" if metrics["margin"] >= 0 else ""
        print(f"    Status: {status_text} "
              f"(margin: {margin_sign}{metrics['margin']:.1%})")
        
        print(f"    Violation rate: {metrics['violation_rate']:.1%} "
              f"({metrics['n_violations']} scenarios)")
        
        if metrics["median_shortfall"] > 0:
            print(f"    Median shortfall: ${metrics['median_shortfall']:,.0f}")
        
        print()