"""
FinOpt — Personal Finance Optimizer

A modular tool for simulating, planning, and optimizing personal
investments to reach financial goals.

Modules
-------
- income       : Fixed and variable income modeling
- investment   : Capital accumulation, portfolio simulation, metrics
- simulation   : Scenario orchestration (base/optimistic/pessimistic, Monte Carlo)
- utils        : Shared utilities (validation, rates, helpers, reporting)

Usage example
-------------
>>> from finopt import IncomeModel, FixedIncome, VariableIncome, SimulationEngine, ScenarioConfig
>>> from datetime import date
>>> income = IncomeModel(
...     fixed=FixedIncome(base=1_400_000.0, annual_growth=0.0),
...     variable=VariableIncome(base=200_000.0, sigma=0.0),
... )
>>> cfg = ScenarioConfig(months=24, start=date(2025, 9, 1))
>>> engine = SimulationEngine(income, cfg)
>>> results = engine.run_three_cases()
>>> results["base"].wealth.tail()
"""

from .income import FixedIncome, VariableIncome, IncomeModel
from .investment import (
    simulate_capital,
    simulate_portfolio,
)
from .scenario import ScenarioConfig, ScenarioResult, SimulationEngine
from . import utils

__all__ = [
    # Income
    "FixedIncome",
    "VariableIncome",
    "IncomeModel",
    # Investment
    "simulate_capital",
    "simulate_portfolio",
    "fixed_rate_path",
    "lognormal_iid",
    "PortfolioMetrics",
    "compute_metrics",
    # Simulation
    "ScenarioConfig",
    "ScenarioResult",
    "SimulationEngine",
    # Utils (as submodule)
    "utils",
]
