"""
Global constants for FinOpt.

Purpose
-------
Centralizes default values and magic numbers used throughout the FinOpt
codebase. Using constants instead of hardcoded values improves maintainability,
ensures consistency, and makes configuration intentions explicit.

Usage
-----
>>> from finopt.constants import DEFAULT_N_SIMS, DEFAULT_FIGSIZE
>>>
>>> result = model.simulate(T=24, n_sims=DEFAULT_N_SIMS)
>>> fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

Categories
----------
- Simulation: Monte Carlo scenario counts, random seeds
- Optimization: Solver defaults, search bounds, tolerances
- Plotting: Figure sizes, colors, transparency values
- Income: Contribution rates, growth rates
- Time: Standard horizon bounds
"""

from typing import Tuple

__all__ = [
    # Simulation
    "DEFAULT_N_SIMS",
    "DEFAULT_N_SIMS_OPTIMIZATION",
    "DEFAULT_N_SIMS_PLOTTING",
    "DEFAULT_SEED",
    # Optimization
    "DEFAULT_T_MAX",
    "DEFAULT_T_MIN",
    "DEFAULT_SOLVER",
    "DEFAULT_OBJECTIVE",
    "DEFAULT_TOLERANCE",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_WITHDRAWAL_EPSILON",
    # Plotting
    "DEFAULT_FIGSIZE",
    "DEFAULT_FIGSIZE_WIDE",
    "DEFAULT_FIGSIZE_TALL",
    "DEFAULT_ALPHA_TRAJECTORIES",
    "DEFAULT_ALPHA_BANDS",
    "DEFAULT_LINEWIDTH",
    "DEFAULT_LINEWIDTH_THICK",
    # Income
    "DEFAULT_FIXED_CONTRIBUTION_RATE",
    "DEFAULT_VARIABLE_CONTRIBUTION_RATE",
    "MONTHS_PER_YEAR",
    # Confidence levels
    "DEFAULT_CONFIDENCE_LEVELS",
    "DEFAULT_GOAL_CONFIDENCE",
]


# =============================================================================
# Simulation Defaults
# =============================================================================

DEFAULT_N_SIMS: int = 500
"""Default number of Monte Carlo scenarios for general simulations."""

DEFAULT_N_SIMS_OPTIMIZATION: int = 500
"""Default number of scenarios for optimization (CVaROptimizer, GoalSeeker)."""

DEFAULT_N_SIMS_PLOTTING: int = 300
"""Default number of scenarios for plotting functions."""

DEFAULT_SEED: int = 42
"""Default random seed for reproducibility."""


# =============================================================================
# Optimization Defaults
# =============================================================================

DEFAULT_T_MAX: int = 240
"""Default maximum horizon for optimization search (20 years)."""

DEFAULT_T_MIN: int = 12
"""Default minimum horizon for optimization search (1 year)."""

DEFAULT_SOLVER: str = "ECOS"
"""Default CVXPY solver for convex optimization.

Alternatives: "SCS", "CLARABEL", "MOSEK" (commercial, requires license).
"""

DEFAULT_OBJECTIVE: str = "balanced"
"""Default optimization objective.

Options: "risky", "balanced", "conservative", "risky_turnover".
"""

DEFAULT_TOLERANCE: float = 1e-4
"""Default convergence tolerance for optimization."""

DEFAULT_MAX_ITERS: int = 10000
"""Default maximum iterations for CVXPY solvers."""

DEFAULT_WITHDRAWAL_EPSILON: float = 0.05
"""Default confidence level for withdrawal feasibility constraints.

P(W_t >= D_t) >= 1 - epsilon, so 0.05 means 95% confidence.
"""


# =============================================================================
# Plotting Defaults
# =============================================================================

DEFAULT_FIGSIZE: Tuple[int, int] = (14, 8)
"""Default figure size (width, height) in inches for standard plots."""

DEFAULT_FIGSIZE_WIDE: Tuple[int, int] = (14, 6)
"""Figure size for wide aspect ratio plots (timeseries, allocations)."""

DEFAULT_FIGSIZE_TALL: Tuple[int, int] = (10, 12)
"""Figure size for tall aspect ratio plots (multi-panel dashboards)."""

DEFAULT_ALPHA_TRAJECTORIES: float = 0.1
"""Default alpha (transparency) for Monte Carlo trajectory lines."""

DEFAULT_ALPHA_BANDS: float = 0.2
"""Default alpha for confidence band fills."""

DEFAULT_LINEWIDTH: float = 1.0
"""Default line width for standard plot lines."""

DEFAULT_LINEWIDTH_THICK: float = 2.0
"""Line width for emphasized lines (means, thresholds)."""


# =============================================================================
# Income Defaults
# =============================================================================

DEFAULT_FIXED_CONTRIBUTION_RATE: float = 0.3
"""Default fraction of fixed income to contribute (30%)."""

DEFAULT_VARIABLE_CONTRIBUTION_RATE: float = 1.0
"""Default fraction of variable income to contribute (100%)."""

MONTHS_PER_YEAR: int = 12
"""Number of months in a year (used for array sizing and conversions)."""


# =============================================================================
# Confidence Levels
# =============================================================================

DEFAULT_CONFIDENCE_LEVELS: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
"""Default quantile levels for confidence band plots."""

DEFAULT_GOAL_CONFIDENCE: float = 0.80
"""Default confidence level for financial goals (80% probability of success)."""
