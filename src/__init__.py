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

"""

from .income import FixedIncome, VariableIncome, IncomeModel
from . import utils


