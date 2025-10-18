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

Example
-------
>>> from datetime import date
>>> from finopt.src.goals import IntermediateGoal, TerminalGoal, GoalSet
>>> 
>>> goals = [
...     IntermediateGoal(month=6, account="Emergency", 
...                     threshold=5_500_000, confidence=0.90),
...     TerminalGoal(account="Emergency", threshold=20_000_000, confidence=0.90),
...     TerminalGoal(account="Housing", threshold=7_000_000, confidence=0.90)
... ]
>>> 
>>> goal_set = GoalSet(goals, ["Emergency", "Housing"], date(2025, 1, 1))
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
    account_names : List[str]
        Account name mapping (e.g., ["Emergency", "Housing"])
    start_date : datetime.date
        Simulation start date for intermediate goal resolution
    
    Raises
    ------
    ValueError
        - If goals list is empty
        - If account reference invalid
        - If duplicate goals exist within type
    
    Examples
    --------
    >>> goals = [
    ...     IntermediateGoal(month=6, account="Emergency", 
    ...                     threshold=5_500_000, confidence=0.90),
    ...     TerminalGoal(account="Emergency", threshold=20_000_000, 
    ...                 confidence=0.90),
    ...     TerminalGoal(account="Housing", threshold=7_000_000, confidence=0.90)
    ... ]
    >>> 
    >>> goal_set = GoalSet(goals, ["Emergency", "Housing"], date(2025, 1, 1))
    >>> goal_set.T_min  # From intermediate goal
    6
    >>> goal_set.intermediate_goals  # Filtered list
    [IntermediateGoal(month=6, ...)]
    >>> goal_set.terminal_goals  # Filtered list
    [TerminalGoal(account='Emergency', ...), TerminalGoal(account='Housing', ...)]
    """
    
    def __init__(
        self,
        goals: List[Union[IntermediateGoal, TerminalGoal]],
        account_names: List[str],
        start_date: date
    ):
        if not goals:
            raise ValueError("goals list cannot be empty")
        if not account_names:
            raise ValueError("account_names list cannot be empty")
        
        self.account_names = account_names
        self.M = len(account_names)
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
        self._validate()
    
    def _validate(self):
        """Validate goal collection for conflicts and invalid references."""
        # Validate intermediate goals
        seen_intermediate = set()
        for goal in self.intermediate_goals:
            idx = self._resolve_account(goal.account)
            self._account_indices_intermediate[goal] = idx
            
            month = goal.resolve_month(self.start_date)
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
        """Get resolved month for intermediate goal."""
        return goal.resolve_month(self.start_date)
    
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
        
        return max(
            g.resolve_month(self.start_date) for g in self.intermediate_goals
        )

    def estimate_minimum_horizon(
        self,
        monthly_contribution: float,
        expected_return: float = 0.0,
        safety_margin: float = 1.2,
        T_max: int = 999999
    ) -> int:
        """
        Estimate minimum horizon for terminal goals via worst-case analysis.
        
        Uses closed-form wealth accumulation formula under constant contributions
        to solve for T given the maximum terminal goal threshold.
        
        Parameters
        ----------
        monthly_contribution : float
            Expected average monthly contribution across all accounts.
            Must be > 0.
        expected_return : float, default 0.0
            Expected monthly arithmetic return. Use 0 for worst-case (no growth).
        safety_margin : float, default 1.2
            Multiplicative time buffer (1.2 = +20% cushion for uncertainty).
            Must be ≥ 1.0.
        T_max : int, default 999999
            Maximum allowable horizon to cap estimate and prevent overflow.
            Used as fallback for infeasible cases.
        
        Returns
        -------
        int
            Estimated minimum horizon in months, respecting intermediate goal
            constraints. Returns T_max if problem is infeasible.
        
        Notes
        -----
        Accumulation formulas (assuming W_0 = 0):
            r = 0:  W_T = A·T  →  T = b/A
            r > 0:  W_T = A·[(1+r)^T - 1]/r  →  T = log(1 + b·r/A) / log(1+r)
        
        Returns T_max if ratio = b·r/A ≤ -1 (infeasible).
        
        Examples
        --------
        >>> goal_set = GoalSet([TerminalGoal(...)], accounts, start_date)
        >>> T_est = goal_set.estimate_minimum_horizon(
        ...     monthly_contribution=500_000,
        ...     expected_return=0.005,  # ~6% annual
        ...     safety_margin=1.5,
        ...     T_max=240
        ... )
        """
        # Input validation
        if monthly_contribution <= 0:
            raise ValueError(
                f"monthly_contribution must be > 0, got {monthly_contribution}"
            )
        if safety_margin < 1.0:
            raise ValueError(
                f"safety_margin must be ≥ 1.0, got {safety_margin}"
            )
        if not isinstance(T_max, int) or T_max < 1:
            raise ValueError(
                f"T_max must be positive int, got {T_max}"
            )
        
        # Early return: no terminal goals means no estimation needed
        if not self.terminal_goals:
            return self.T_min
        
        # Find the most stringent terminal goal
        max_threshold = max(g.threshold for g in self.terminal_goals)
        
        # Solve for T based on expected return model
        if expected_return == 0:
            # Linear accumulation: W_T = A·T
            T_est = max_threshold / monthly_contribution
        else:
            # Geometric accumulation: W_T = A·[(1+r)^T - 1]/r
            ratio = max_threshold * expected_return / monthly_contribution
            
            if ratio <= -1:
                # Infeasible: logarithm argument would be non-positive
                return T_max
            
            T_est = np.log1p(ratio) / np.log1p(expected_return)
        
        # Apply safety margin and enforce upper bound
        T_est_safe = int(np.ceil(T_est * safety_margin))
        T_est_safe = min(T_est_safe, T_max)
        
        # Respect intermediate goal constraints (T_min) and ensure positive
        return max(self.T_min, T_est_safe, 1)
    
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
    account_names: List[str],
    start_date: date
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
    account_names : List[str]
        Account name mapping for resolution
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
    >>> goals = [
    ...     IntermediateGoal(month=12, account="Emergency", 
    ...                     threshold=2_000_000, confidence=0.95),
    ...     TerminalGoal(account="Housing", threshold=15_000_000, confidence=0.90)
    ... ]
    >>> result = model.simulate(T=24, X=X, n_sims=500, seed=42)
    >>> status = check_goals(result, goals, model.account_names, date(2025, 1, 1))
    >>> 
    >>> for goal, metrics in status.items():
    ...     print(f"{goal}: {'✓' if metrics['satisfied'] else '✗'}")
    """
    # Validate goal structure
    goal_set = GoalSet(goals, account_names, start_date)
    
    # Check horizon compatibility for intermediate goals
    if goal_set.intermediate_goals:
        max_intermediate_month = max(
            g.resolve_month(start_date) for g in goal_set.intermediate_goals
        )
        if max_intermediate_month > result.T:
            raise ValueError(
                f"Intermediate goals require T ≥ {max_intermediate_month}, "
                f"but result.T = {result.T}"
            )
    
    status = {}
    n_sims = result.n_sims
    
    for goal in goals:
        # Resolve account index
        m = goal_set.get_account_index(goal)
        
        # Determine target month
        if isinstance(goal, IntermediateGoal):
            t = goal.resolve_month(start_date)
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
    account_names: List[str],
    start_date: date
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
    account_names : List[str]
        Account name mapping
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
    >>> result = model.simulate(T=24, X=X, n_sims=500, seed=42)
    >>> progress = goal_progress(result, goals, model.account_names, date(2025, 1, 1))
    >>> 
    >>> for goal, pct in progress.items():
    ...     print(f"{goal.account} @ {goal.month if hasattr(goal, 'month') else 'T'}: "
    ...           f"{pct:.1%} progress")
    """
    goal_set = GoalSet(goals, account_names, start_date)
    
    progress = {}
    for goal in goals:
        m = goal_set.get_account_index(goal)
        
        # Determine target month
        if isinstance(goal, IntermediateGoal):
            t = goal.resolve_month(start_date)
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
    account_names: List[str],
    start_date: date
):
    """
    Pretty-print goal satisfaction status.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation output
    goals : List[Union[IntermediateGoal, TerminalGoal]]
        Goals to display
    account_names : List[str]
        Account name mapping
    start_date : datetime.date
        Simulation start date
    
    Examples
    --------
    >>> result = model.simulate(T=24, X=X, n_sims=500, seed=42)
    >>> print_goal_status(result, goals, model.account_names, date(2025, 1, 1))
    
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
    status = check_goals(result, goals, account_names, start_date)
    
    print("\n=== Goal Status ===\n")
    
    for goal, metrics in status.items():
        # Status symbol
        symbol = "✓" if metrics["satisfied"] else "✗"
        
        # Goal type and location
        if isinstance(goal, IntermediateGoal):
            month = goal.resolve_month(start_date)
            goal_type = "IntermediateGoal"
            location = f"@ month {month}"
        else:
            goal_type = "TerminalGoal"
            location = f"@ T={result.T}"
        
        # Resolve account name
        goal_set = GoalSet([goal], account_names, start_date)
        account_idx = goal_set.get_account_index(goal)
        account_name = account_names[account_idx]
        
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