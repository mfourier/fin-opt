# Welcome to FinOpt

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://mfourier.github.io/fin-opt/)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/mfourier/fin-opt/blob/main/LICENSE)

**FinOpt** is a modular Python framework designed for intelligent financial planning. It combines stochastic simulation of income and investment returns with convex optimization to help users achieve their financial goals under uncertainty.

## Key Features

- **Stochastic Income Modeling**: Simulate fixed and variable income with growth, seasonality, and noise.
- **Wealth Dynamics**: Model the evolution of investment accounts using affine wealth representations.
- **Goal-Oriented Optimization**: Find the minimum time to achieve multiple financial goals (e.g., emergency funds, housing) with a specified level of confidence.
- **Bilevel Optimization**: Solve complex problems that minimize time while maximizing terminal wealth.
- **Extensible Architecture**: Modular design allows for custom return models, optimizers, and goal types.

## Installation

```bash
# Using Conda (recommended)
conda env create -f environment.yml
conda activate finance

# Install CVXPY for convex optimization
pip install cvxpy
```

## Quick Example

```python
from finopt import FinancialModel, Account, IncomeModel, FixedIncome
from finopt.goals import TerminalGoal
from finopt.optimization import CVaROptimizer

# Define income and accounts
income = IncomeModel(fixed=FixedIncome(base=1_500_000, annual_growth=0.03))
accounts = [
    Account.from_annual("Conservative", annual_return=0.08, annual_volatility=0.09),
    Account.from_annual("Aggressive", annual_return=0.14, annual_volatility=0.15)
]

# Create model and optimize
model = FinancialModel(income, accounts)
goals = [TerminalGoal(account="Aggressive", threshold=5_000_000, confidence=0.80)]

optimizer = CVaROptimizer(n_accounts=2, objective="balanced")
result = model.optimize(goals=goals, optimizer=optimizer, T_max=120, n_sims=500, seed=42)

# Visualize results
model.plot("wealth", result=result, show_trajectories=True)
```

## Documentation Roadmap

Explore the technical components of the FinOpt framework:

### Core Components
- [Stochastic Returns](returns.md): Generating correlated lognormal returns for simulation.
- [Income Module](income.md): Modeling fixed and variable cash flows with growth and seasonality.
- [Portfolio Dynamics](portfolio.md): The mathematics of wealth evolution and affine representations.
- [Scheduled Withdrawals](withdrawal.md): Integration of planned cash outflows into the wealth equation.

### Optimization & Logic
- [Goals Framework](goals.md): Defining financial milestones as probabilistic chance constraints.
- [Optimization](optimization.md): Technical details on CVaR reformulation and Sample Average Approximation (SAA) solvers.

### Integration & Architecture
- [Unified Model](model.md): The `FinancialModel` facade that orchestrates all system components.
- [Technical Framework](framework.md): A deep dive into the system architecture, design principles, and mathematical foundations.

### Advanced & Utilities
- [Exceptions](exceptions.md): Error handling and custom exception classes.
- [Serialization](serialization.md): Saving and loading models and scenarios.
- [Shared Utilities](utils.md): Common helper functions and numerical routines.
