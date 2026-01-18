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
)

# Utilities
from . import utils

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
    # Serialization
    "save_model",
    "load_model",
    "save_optimization_result",
    "load_optimization_result",
    "account_to_dict",
    "account_from_dict",
    "income_to_dict",
    "income_from_dict",
    # Optimization (lazy loaded)
    "CVaROptimizer",
    "SAAOptimizer",
    "GoalSeeker",
    "OptimizationResult",
    # Utilities
    "utils",
]
