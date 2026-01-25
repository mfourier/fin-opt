"""
FinOpt - Goal-based Portfolio Optimization Framework.

A modular framework for simulating, planning, and optimizing personal
investment strategies to achieve financial goals with probabilistic guarantees.

Core Modules
------------
- income: Fixed and variable income modeling with contribution scheduling
- portfolio: Account definitions and wealth dynamics execution
- returns: Correlated lognormal return generation
- goals: Probabilistic goal specification (intermediate and terminal)
- optimization: CVaR-based convex optimization solvers
- model: Unified orchestration facade with caching and plotting

Quick Start
-----------
>>> from finopt import FinancialModel, Account, IncomeModel, FixedIncome
>>> from finopt.goals import TerminalGoal
>>>
>>> # Define income
>>> income = IncomeModel(fixed=FixedIncome(base=1_500_000, annual_growth=0.03))
>>>
>>> # Configure accounts
>>> accounts = [
...     Account.from_annual("Conservative", annual_return=0.08, annual_volatility=0.09),
...     Account.from_annual("Aggressive", annual_return=0.14, annual_volatility=0.15),
... ]
>>>
>>> # Create and simulate
>>> model = FinancialModel(income, accounts)
>>> result = model.simulate(T=24, n_sims=1000, seed=42)

See Also
--------
- CLAUDE.md: Comprehensive usage guide and architecture documentation
- notebooks/FinOpt-Workflow.ipynb: Interactive examples
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "FinOpt Contributors"

# Core income modeling
from .income import FixedIncome, VariableIncome, IncomeModel

# Portfolio and account management
from .portfolio import Account, Portfolio

# Return generation
from .returns import ReturnModel

# Withdrawal modeling
from .withdrawal import (
    WithdrawalEvent,
    WithdrawalSchedule,
    StochasticWithdrawal,
    WithdrawalModel,
)

# Orchestration facade
from .model import FinancialModel, SimulationResult

# Goals
from .goals import IntermediateGoal, TerminalGoal, GoalSet

# Configuration and serialization
from .config import (
    AccountConfig,
    IncomeConfig,
    FixedIncomeConfig,
    VariableIncomeConfig,
    WithdrawalEventConfig,
    StochasticWithdrawalConfig,
    WithdrawalConfig,
)
from .serialization import (
    save_model,
    load_model,
    save_optimization_result,
    load_optimization_result,
    account_to_dict,
    account_from_dict,
    income_to_dict,
    income_from_dict,
    withdrawal_to_dict,
    withdrawal_from_dict,
    SCHEMA_VERSION,
)

# Utilities
from . import utils

# Type definitions
from .types import (
    ReturnStrategyDict,
    AnnualParamsDict,
    MonthlyContributionDict,
    SimulationResultDict,
    GoalMetricsDict,
    DiagnosticsDict,
    PlotColorsDict,
)

# Constants
from .constants import (
    DEFAULT_N_SIMS,
    DEFAULT_N_SIMS_OPTIMIZATION,
    DEFAULT_N_SIMS_PLOTTING,
    DEFAULT_SEED,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    DEFAULT_SOLVER,
    DEFAULT_OBJECTIVE,
    DEFAULT_TOLERANCE,
    DEFAULT_MAX_ITERS,
    DEFAULT_WITHDRAWAL_EPSILON,
    DEFAULT_FIGSIZE,
    DEFAULT_FIGSIZE_WIDE,
    DEFAULT_FIGSIZE_TALL,
    DEFAULT_ALPHA_TRAJECTORIES,
    DEFAULT_ALPHA_BANDS,
    DEFAULT_LINEWIDTH,
    DEFAULT_LINEWIDTH_THICK,
    DEFAULT_FIXED_CONTRIBUTION_RATE,
    DEFAULT_VARIABLE_CONTRIBUTION_RATE,
    MONTHS_PER_YEAR,
    DEFAULT_CONFIDENCE_LEVELS,
    DEFAULT_GOAL_CONFIDENCE,
)

# Exceptions
from .exceptions import (
    FinOptError,
    ConfigurationError,
    ValidationError,
    TimeIndexError,
    AllocationConstraintError,
    OptimizationError,
    InfeasibleError,
    MemoryLimitError,
)

# Lazy imports for optional dependencies
def __getattr__(name: str):
    """Lazy import for optimization module (requires cvxpy)."""
    if name in ("CVaROptimizer", "SAAOptimizer", "GoalSeeker", "OptimizationResult"):
        from . import optimization
        return getattr(optimization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Income
    "FixedIncome",
    "VariableIncome",
    "IncomeModel",
    # Portfolio
    "Account",
    "Portfolio",
    # Returns
    "ReturnModel",
    # Withdrawals
    "WithdrawalEvent",
    "WithdrawalSchedule",
    "StochasticWithdrawal",
    "WithdrawalModel",
    # Model
    "FinancialModel",
    "SimulationResult",
    # Goals
    "IntermediateGoal",
    "TerminalGoal",
    "GoalSet",
    # Config
    "AccountConfig",
    "IncomeConfig",
    "FixedIncomeConfig",
    "VariableIncomeConfig",
    "WithdrawalEventConfig",
    "StochasticWithdrawalConfig",
    "WithdrawalConfig",
    # Serialization
    "save_model",
    "load_model",
    "save_optimization_result",
    "load_optimization_result",
    "account_to_dict",
    "account_from_dict",
    "income_to_dict",
    "income_from_dict",
    "withdrawal_to_dict",
    "withdrawal_from_dict",
    "SCHEMA_VERSION",
    # Optimization (lazy loaded)
    "CVaROptimizer",
    "SAAOptimizer",
    "GoalSeeker",
    "OptimizationResult",
    # Utilities
    "utils",
    # Types
    "ReturnStrategyDict",
    "AnnualParamsDict",
    "MonthlyContributionDict",
    "SimulationResultDict",
    "GoalMetricsDict",
    "DiagnosticsDict",
    "PlotColorsDict",
    # Constants
    "DEFAULT_N_SIMS",
    "DEFAULT_N_SIMS_OPTIMIZATION",
    "DEFAULT_N_SIMS_PLOTTING",
    "DEFAULT_SEED",
    "DEFAULT_T_MAX",
    "DEFAULT_T_MIN",
    "DEFAULT_SOLVER",
    "DEFAULT_OBJECTIVE",
    "DEFAULT_TOLERANCE",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_WITHDRAWAL_EPSILON",
    "DEFAULT_FIGSIZE",
    "DEFAULT_FIGSIZE_WIDE",
    "DEFAULT_FIGSIZE_TALL",
    "DEFAULT_ALPHA_TRAJECTORIES",
    "DEFAULT_ALPHA_BANDS",
    "DEFAULT_LINEWIDTH",
    "DEFAULT_LINEWIDTH_THICK",
    "DEFAULT_FIXED_CONTRIBUTION_RATE",
    "DEFAULT_VARIABLE_CONTRIBUTION_RATE",
    "MONTHS_PER_YEAR",
    "DEFAULT_CONFIDENCE_LEVELS",
    "DEFAULT_GOAL_CONFIDENCE",
    # Exceptions
    "FinOptError",
    "ConfigurationError",
    "ValidationError",
    "TimeIndexError",
    "AllocationConstraintError",
    "OptimizationError",
    "InfeasibleError",
    "MemoryLimitError",
]
