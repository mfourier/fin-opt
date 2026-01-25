"""
Type definitions for FinOpt.

Purpose
-------
Provides TypedDict definitions for structured dictionary types used
throughout the FinOpt codebase. Using TypedDicts improves type safety,
enables IDE autocompletion, and documents expected dictionary structures.

Usage
-----
>>> from finopt.types import ReturnStrategyDict, MonthlyContributionDict
>>>
>>> strategy: ReturnStrategyDict = {"mu": 0.005, "sigma": 0.02}
>>> contributions: MonthlyContributionDict = {
...     "fixed": [0.3] * 12,
...     "variable": [1.0] * 12
... }

Type Definitions
----------------
ReturnStrategyDict
    Monthly return parameters for Account: {"mu", "sigma"}

AnnualParamsDict
    Annualized return parameters: {"return", "volatility"}

MonthlyContributionDict
    Income contribution fractions: {"fixed", "variable"} with 12-element arrays

SimulationResultDict
    Portfolio simulation output: {"wealth", "total_wealth", "contributions", ...}

GoalMetricsDict
    Goal satisfaction metrics: {"satisfied", "achieved_pct", "var", ...}
"""

from typing import List, Union, Any, TYPE_CHECKING
from typing_extensions import TypedDict, NotRequired
import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ReturnStrategyDict",
    "AnnualParamsDict",
    "MonthlyContributionDict",
    "SimulationResultDict",
    "GoalMetricsDict",
    "DiagnosticsDict",
    "PlotColorsDict",
]


class ReturnStrategyDict(TypedDict):
    """
    Monthly return parameters for Account.

    Used internally by Account to store return characteristics in monthly space.
    For user-facing APIs, prefer Account.from_annual() which handles conversion.

    Attributes
    ----------
    mu : float
        Monthly expected return (arithmetic). E.g., 0.0033 for ~4% annual.
    sigma : float
        Monthly volatility (standard deviation). E.g., 0.0144 for ~5% annual.

    Examples
    --------
    >>> strategy: ReturnStrategyDict = {"mu": 0.00327, "sigma": 0.01443}
    """

    mu: float
    sigma: float


class AnnualParamsDict(TypedDict):
    """
    Annualized return parameters for Account.

    Returned by Account.annual_params property, provides user-friendly
    view of account return characteristics.

    Attributes
    ----------
    return : float
        Annual expected return. E.g., 0.04 for 4% annual return.
    volatility : float
        Annual volatility (standard deviation). E.g., 0.05 for 5% annual vol.

    Notes
    -----
    Key names use "return" and "volatility" (not "mu"/"sigma") to match
    financial convention for annual parameters.

    Examples
    --------
    >>> account = Account.from_annual("RN", annual_return=0.08, annual_volatility=0.15)
    >>> params: AnnualParamsDict = account.annual_params
    >>> params["return"]  # 0.08
    >>> params["volatility"]  # 0.15
    """

    # Note: "return" is a reserved word in Python, but valid as TypedDict key
    return_: float  # Will be serialized as "return"
    volatility: float


# Workaround: TypedDict doesn't allow "return" as key name directly
# We use a functional form to define this properly
AnnualParamsDict = TypedDict(
    "AnnualParamsDict",
    {"return": float, "volatility": float},
)


class MonthlyContributionDict(TypedDict, total=False):
    """
    Income contribution fractions by stream.

    Specifies what fraction of each income stream to contribute monthly.
    Each array has 12 elements corresponding to January-December.

    Attributes
    ----------
    fixed : ArrayLike
        12-element array of contribution fractions for fixed income.
        Default: [0.3] * 12 (30% of fixed income contributed).
    variable : ArrayLike
        12-element array of contribution fractions for variable income.
        Default: [1.0] * 12 (100% of variable income contributed).

    Notes
    -----
    - Fractions are rotated based on simulation start month
    - Both keys are optional; missing keys use defaults
    - Values can be list, tuple, or np.ndarray

    Examples
    --------
    >>> # Contribute 35% of fixed, 100% of variable year-round
    >>> contrib: MonthlyContributionDict = {
    ...     "fixed": [0.35] * 12,
    ...     "variable": [1.0] * 12
    ... }
    >>>
    >>> # Higher contribution during bonus months (Dec, Jun)
    >>> seasonal: MonthlyContributionDict = {
    ...     "fixed": [0.3]*11 + [0.5],  # 50% in December
    ...     "variable": [1.0] * 12
    ... }
    """

    fixed: ArrayLike
    variable: ArrayLike


class SimulationResultDict(TypedDict, total=False):
    """
    Portfolio simulation output from Portfolio.simulate().

    Contains wealth trajectories, contributions, and metadata from
    a portfolio simulation run.

    Attributes
    ----------
    wealth : NDArray
        Per-account wealth trajectories, shape (n_sims, T+1, M).
        wealth[i, t, m] = wealth in account m at start of period t in scenario i.
    total_wealth : NDArray
        Aggregate wealth across all accounts, shape (n_sims, T+1).
        total_wealth[i, t] = sum over m of wealth[i, t, m].
    contributions : NDArray
        Per-account contributions, shape (n_sims, T, M).
        contributions[i, t, m] = A_t * x_t^m in scenario i.
    allocation : NDArray
        Allocation policy used, shape (T, M).
    method : str
        Computation method used: "recursive" or "affine".
    withdrawals : NDArray, optional
        Per-account withdrawals if provided, shape (n_sims, T, M).

    Examples
    --------
    >>> result = portfolio.simulate(A=A_sims, R=R_sims, X=X)
    >>> W = result["wealth"]  # (500, 25, 2)
    >>> W_final = W[:, -1, :].sum(axis=1)  # Terminal wealth per scenario
    """

    wealth: NDArray[np.floating[Any]]
    total_wealth: NDArray[np.floating[Any]]
    contributions: NDArray[np.floating[Any]]
    allocation: NDArray[np.floating[Any]]
    method: str
    withdrawals: NDArray[np.floating[Any]]


class GoalMetricsDict(TypedDict):
    """
    Goal satisfaction metrics from check_goals() or goal_progress().

    Provides detailed statistics about goal achievement across scenarios.

    Attributes
    ----------
    satisfied : bool
        Whether goal is satisfied at required confidence level.
    achieved_pct : float
        Percentage of scenarios where goal threshold was met.
        Range: 0.0 to 1.0.
    var : float
        Value-at-Risk at goal's confidence level.
        Interpretation: (1-ε) of scenarios have wealth >= VaR.
    threshold : float
        Goal's target wealth threshold.
    shortfall : float
        VaR - threshold. Positive if goal exceeded, negative if short.

    Examples
    --------
    >>> metrics: GoalMetricsDict = check_goals(result, goals, accounts, start)
    >>> for goal, m in metrics.items():
    ...     status = "✓" if m["satisfied"] else "✗"
    ...     print(f"{status} {goal}: {m['achieved_pct']:.1%} achieved")
    """

    satisfied: bool
    achieved_pct: float
    var: float
    threshold: float
    shortfall: float


class DiagnosticsDict(TypedDict, total=False):
    """
    Optimization solver diagnostics from CVaROptimizer.

    Contains debugging information about the optimization solve.

    Attributes
    ----------
    solver_status : str
        Solver termination status (e.g., "optimal", "infeasible").
    solve_time : float
        Wall-clock time for solve in seconds.
    iterations : int
        Number of solver iterations.
    primal_objective : float
        Primal objective value at solution.
    dual_objective : float
        Dual objective value at solution.
    gap : float
        Duality gap (primal - dual).

    Examples
    --------
    >>> result = optimizer.solve(T=36, A=A, R=R, W0=W0, goal_set=goals)
    >>> diag: DiagnosticsDict = result.diagnostics
    >>> print(f"Solved in {diag['solve_time']:.3f}s with status {diag['solver_status']}")
    """

    solver_status: str
    solve_time: float
    iterations: int
    primal_objective: float
    dual_objective: float
    gap: float


class PlotColorsDict(TypedDict, total=False):
    """
    Color specifications for plotting functions.

    Used by various plot methods to customize visualization colors.

    Attributes
    ----------
    fixed : str
        Color for fixed income components. Default: varies by plot.
    variable : str
        Color for variable income components. Default: varies by plot.
    total : str
        Color for total/aggregate lines. Default: varies by plot.
    goal : str
        Color for goal threshold lines. Default: "red".
    band : str
        Color for confidence bands. Default: varies by plot.

    Examples
    --------
    >>> colors: PlotColorsDict = {
    ...     "fixed": "#2196F3",
    ...     "variable": "#4CAF50",
    ...     "total": "#FF9800"
    ... }
    >>> income.plot(colors=colors)
    """

    fixed: str
    variable: str
    total: str
    goal: str
    band: str
